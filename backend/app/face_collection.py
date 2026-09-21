"""
face_collection.py

Backend-owned continuous face-capture collection, independent of any
browser live-view. Reuses camera_stream.CameraStream / get_stream() exactly
the way a real viewer would — subscribe()/unsubscribe() — so there is no
second RTSP connection and no new frame-reading loop: the same background
thread that already streams JPEGs to any real websocket viewers is what
runs feed_frame() and therefore produces captures. A "phantom" subscriber's
only job is to occupy CameraStream._subscribers so that thread keeps
running with zero real viewers attached.

Start/stop is explicit and API-driven only (see
face_training_routes.py's /collection/* endpoints) — collection never
starts on its own just because a camera exists, and the frontend is never
responsible for keeping it alive (this lives entirely server-side).

Because it reuses CameraStream unchanged, it automatically inherits that
class's existing RTSP reconnect handling (camera_stream.py's _run() loop)
— nothing new was needed for "survive a disconnect".

Multi-day sessions (added for the 7-day collection plan): start_session()
records a persistent row in face_db.collection_sessions (start time,
planned end time, which cameras, running activity) and starts each camera
via the same start() below. resume_if_needed() — called once at app
startup — detects an unfinished session from before a restart and
re-subscribes its cameras; because start() is idempotent and this process's
in-memory `_active` dict is always empty on a fresh process start, this can
never create a second worker for a camera that's already running (there is
no such thing right after a restart — the old process's threads died with
it). A background thread (_expiry_loop) checks periodically whether the
running session's planned end time has passed and stops it automatically —
this is what makes "runs for 7 days without me restarting anything" true
even if nobody ever polls the status endpoint.
"""

import logging
import os
import threading
import time

from . import camera_db, camera_stream, face_db
from .face_pipeline import DEFAULT_COLLECTION_DAYS, SESSION_CHECK_INTERVAL_SECONDS, TRAINING_CAPTURE_DIR

log = logging.getLogger("face_collection")


class _CollectorSink:
    """Discards frame bytes. CameraStream._run() calls put_nowait() on every
    subscriber for every broadcast frame; this one just drops them — its
    only purpose is being present in _subscribers so subscribe()/
    unsubscribe()'s existing thread-lifecycle logic keeps the RTSP read
    loop (and therefore feed_frame()) running without a real viewer."""

    def put_nowait(self, item):
        pass


_active: dict[int, _CollectorSink] = {}
_lock = threading.Lock()


def start(camera_id: int) -> bool:
    """Idempotent: calling start() on an already-collecting camera is a
    no-op that still returns True. Returns False only if the camera isn't
    configured (no host), same "not configured" check the live-view
    websocket already uses."""
    cam = camera_db.get_camera(camera_id)
    if not cam or not cam.get("host"):
        return False
    with _lock:
        if camera_id in _active:
            return True
        sink = _CollectorSink()
        camera_stream.get_stream(camera_id).subscribe(sink, is_collector=True)
        _active[camera_id] = sink
    log.info("face collection started for camera %s", camera_id)
    return True


def stop(camera_id: int) -> bool:
    """Returns True if collection was running and has now been stopped,
    False if it wasn't running (not an error — safe to call unconditionally)."""
    with _lock:
        sink = _active.pop(camera_id, None)
    if sink is None:
        return False
    camera_stream.get_stream(camera_id).unsubscribe(sink)
    log.info("face collection stopped for camera %s", camera_id)
    return True


def stop_all() -> list[int]:
    with _lock:
        camera_ids = list(_active.keys())
    return [cid for cid in camera_ids if stop(cid)]


def status() -> dict:
    with _lock:
        active_ids = sorted(_active.keys())
    return {"active_camera_ids": active_ids, "count": len(active_ids)}


# ---------------------------------------------------------------------------
# Multi-day sessions
# ---------------------------------------------------------------------------

def start_session(camera_ids: list[int], days: float | None = None) -> dict:
    """Starts (or extends, if one is already running) a persisted
    collection session, then starts each camera via start() above —
    idempotent per camera, so calling this again with an overlapping camera
    list is safe and just leaves those cameras running."""
    days = DEFAULT_COLLECTION_DAYS if days is None else days
    now = time.time()
    planned_end = now + days * 86400

    existing = face_db.get_running_collection_session()
    if existing:
        # Don't create a second concurrent "running" row — extend the
        # existing one (new camera list, pushed-out end time) instead.
        merged_cameras = sorted(set(existing["camera_ids"]) | set(camera_ids))
        face_db.update_collection_session_plan(existing["id"], merged_cameras, planned_end)
        session_id = existing["id"]
    else:
        session_id = face_db.create_collection_session(now, planned_end, camera_ids)

    ensure_expiry_watcher_started()

    started, failed = [], []
    for cid in camera_ids:
        (started if start(cid) else failed).append(cid)
    if failed:
        log.warning("collection session %s: cameras not configured, skipped: %s", session_id, failed)

    return {**face_db.get_collection_session(session_id), "failed_camera_ids": failed}


def stop_session(reason: str = "stopped") -> dict | None:
    """Manual early stop: stops every currently-active camera and marks the
    running session finished. Returns the finished session dict, or None if
    none was running."""
    session = face_db.get_running_collection_session()
    stop_all()
    if session:
        face_db.finish_collection_session(session["id"], reason)
        return face_db.get_collection_session(session["id"])
    return None


def resume_if_needed() -> None:
    """Call once at app startup. If a 'running' session exists and hasn't
    passed its planned end time, re-subscribes its cameras — safe to call
    even though this is a fresh process with an empty `_active` dict, since
    that's exactly the situation (old process's threads died with it, so
    there is no existing worker to duplicate). If the session's end time
    already passed while the backend was down, marks it completed instead
    of resuming, so a long outage doesn't silently restart an already-over
    session."""
    session = face_db.get_running_collection_session()
    if session is None:
        return
    if time.time() >= session["planned_end_at"]:
        face_db.finish_collection_session(session["id"], "completed")
        log.info(
            "collection session %s's planned end time already passed while the backend was down; "
            "marked completed, not resuming",
            session["id"],
        )
        return
    for cid in session["camera_ids"]:
        start(cid)
    log.info("resumed collection session %s (%d camera(s))", session["id"], len(session["camera_ids"]))


_expiry_thread_started = False
_expiry_thread_lock = threading.Lock()


def _expiry_loop() -> None:
    while True:
        time.sleep(SESSION_CHECK_INTERVAL_SECONDS)
        try:
            session = face_db.get_running_collection_session()
            if session and time.time() >= session["planned_end_at"]:
                stop_all()
                face_db.finish_collection_session(session["id"], "completed")
                log.info("collection session %s reached its planned end time; stopped automatically", session["id"])
        except Exception:
            log.exception("collection session expiry check failed")


def ensure_expiry_watcher_started() -> None:
    """Idempotent — safe to call from both app startup and every
    start_session() call. One daemon thread for the process's lifetime,
    checking every SESSION_CHECK_INTERVAL_SECONDS regardless of whether
    anyone is polling /collection/status — this is what makes the session
    stop itself after 7 days without requiring anyone to watch it."""
    global _expiry_thread_started
    with _expiry_thread_lock:
        if not _expiry_thread_started:
            threading.Thread(target=_expiry_loop, daemon=True).start()
            _expiry_thread_started = True


def get_disk_usage_bytes() -> int:
    """Total size of everything under the training-capture directory
    (unlabeled staging + every <employee_id>/ folder) — informational only,
    for the monitoring endpoint; not itself an enforcement mechanism (the
    capture-count limits are)."""
    total = 0
    for root, _dirs, files in os.walk(TRAINING_CAPTURE_DIR):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total
