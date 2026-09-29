"""
desks.py

Glue between face recognition and desk analytics (desk_tracker.py).

face_pipeline.CameraFacePipeline calls observe() on every processed frame.
For a camera that has desk outlines, about once a second, each detected face
whose centre sits inside a desk is identified and passed to the tracker. An
identity is cached per face track (ByteTrack's track_id) for
IDENTITY_CACHE_SECONDS, so a person sitting still is embedded and classified
once every so often rather than every second, and at most
MAX_NEW_IDS_PER_CYCLE fresh identifications run per camera per cycle to cap
CPU. Every desk identification also marks attendance.

Also keeps every camera with face recognition switched on streaming with no
one watching (same approach as footfall.py's gates), so desk time and
attendance are recorded all day rather than only while the Live Feed is open.
"""

import logging
import os
import threading
import time

from app import analytics_settings, attendance, camera_db
from app.desk_tracker import DESK_SESSION_GRACE_SECONDS, DeskTracker, in_polygon
from app import lifecycle

log = logging.getLogger("desks")

DESK_MIN_CONFIDENCE = float(os.environ.get("DESK_MIN_CONFIDENCE", "0.7"))
DESK_CYCLE_SECONDS = 1.0
IDENTITY_CACHE_SECONDS = 30.0
MAX_NEW_IDS_PER_CYCLE = 3
SYNC_INTERVAL_SECONDS = 15


class _KeepAliveSink:
    def put_nowait(self, item):
        pass


class DeskService:
    def __init__(self):
        self.tracker: DeskTracker | None = None
        self._lock = threading.Lock()
        self._last_cycle: dict[int, float] = {}
        self._ids: dict[tuple[int, int], tuple[str, float, float]] = {}  # (camera, track) -> (employee, score, when)
        self._sinks: dict[int, _KeepAliveSink] = {}
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.tracker = DeskTracker()
        threading.Thread(target=self._sync_loop, daemon=True, name="desks-keepalive").start()

    def refresh_zones(self) -> None:
        if self.tracker:
            with self._lock:
                self.tracker.refresh_zones()

    def has_zones(self, camera_id: int) -> bool:
        return bool(self.tracker and self.tracker.zones_for(camera_id))

    # --- keep face-recognition cameras streaming ---------------------------

    def _sync_loop(self) -> None:
        while True:
            try:
                self._sync()
            except Exception:
                log.exception("desks: keep-alive sync failed")
            if lifecycle.wait(SYNC_INTERVAL_SECONDS):
                return
    def _sync(self) -> None:
        from app import camera_stream  # lazy: camera_stream imports face_pipeline, which imports this module

        wanted = ({c["id"] for c in camera_db.list_cameras() if camera_db.is_streamable(c) and c.get("attendance_tracking")}
                  if analytics_settings.enabled("face_recognition") else set())
        for cid in wanted - self._sinks.keys():
            sink = _KeepAliveSink()
            camera_stream.get_stream(cid).subscribe(sink, is_collector=True, wants_frames=False)
            self._sinks[cid] = sink
            log.info("desks: keeping camera %s streaming for face recognition", cid)
        for cid in self._sinks.keys() - wanted:
            camera_stream.get_stream(cid).unsubscribe(self._sinks.pop(cid))

    # --- per-frame hook ----------------------------------------------------

    def observe(self, camera_id: int, frame_shape: tuple, tracked_faces: list[tuple[int, list[float]]], identify) -> None:
        """tracked_faces: [(track_id, [x1, y1, x2, y2]), ...] in pixels.
        identify(bbox) -> (employee_id | None, score)."""
        if not self.has_zones(camera_id) or not analytics_settings.enabled("desk_analytics"):
            return
        now = time.time()
        if now - self._last_cycle.get(camera_id, 0) < DESK_CYCLE_SECONDS:
            return
        self._last_cycle[camera_id] = now

        h, w = frame_shape[:2]
        zones = self.tracker.zones_for(camera_id)
        faces, fresh = [], 0
        for track_id, bbox in tracked_faces:
            cx, cy = (bbox[0] + bbox[2]) / 2 / w, (bbox[1] + bbox[3]) / 2 / h
            if not any(in_polygon(cx, cy, z["polygon"]) for z in zones):
                continue
            cached = self._ids.get((camera_id, track_id))
            if cached and now - cached[2] < IDENTITY_CACHE_SECONDS:
                emp, score = cached[0], cached[1]
            elif fresh < MAX_NEW_IDS_PER_CYCLE:
                fresh += 1
                emp, score = identify(bbox)
                if not emp or score < DESK_MIN_CONFIDENCE:
                    continue
                self._ids[(camera_id, track_id)] = (emp, score, now)
            else:
                continue
            faces.append({"employee_id": emp, "bbox": bbox, "score": score})
            attendance.record(emp, camera_id, score, now)

        with self._lock:
            self.tracker.process_frame(camera_id, faces, w, h, now)
        for key in [k for k, v in self._ids.items() if now - v[2] > IDENTITY_CACHE_SECONDS * 4]:
            del self._ids[key]

    # --- report ------------------------------------------------------------

    def report(self, date: str) -> dict:
        from datetime import date as _date

        from app import desk_db, employee_directory

        today = date == _date.today().isoformat()
        live = self.tracker.live_status() if (self.tracker and today) else {}
        now = time.time()
        away_now = {emp: now for emp, st in live.items() if st["status"] == "away"}
        rows = []
        for r in desk_db.get_daily_report(date, open_away_until=away_now):
            emp = employee_directory.get_employee(r["employee_id"]) or {}
            st = live.get(r["employee_id"])
            rows.append({
                **r,
                "name": emp.get("name") or r["employee_id"],
                "company": emp.get("company") or "No company",
                "status": ("At desk" if st["status"] == "at_desk" else "Away") if st else "Away",
                "current_desk": (st or {}).get("zone_label") or "-",
            })
        return {"date": date, "grace_seconds": DESK_SESSION_GRACE_SECONDS, "employees": rows}


service = DeskService()
