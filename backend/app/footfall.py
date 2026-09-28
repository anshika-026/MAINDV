"""
footfall.py

Unique footfall across every entry gate, using the body-appearance Re-ID
engine vendored in app/reid/ (see unique-footfall-export/UNIQUE_FOOTFALL.md
for how the engine itself works and its real-world limits).

Gate cameras are simply cameras whose purpose is "Entry/Exit" — set it in
Camera Management, no separate config. All gates share ONE identity
gallery: a person first seen at gate 1 who leaves through gate 3, or comes
back in through gate 2, matches the same PERSON_NNN and is counted once.
That's the one real change from the export's own wiring, where every
camera worker held its own private gallery copy and only picked up other
cameras' new identities on an explicit reload signal. Here all gates run
in this one process, so they read the same gallery object, and it's
rebuilt the moment any gate enrolls someone.

Also owns keeping gate streams alive 24/7: CameraStream only reads RTSP
while it has a subscriber, and footfall has to count whether or not anyone
has the Live Feed page open. Same phantom-subscriber approach as
face_collection.py.
"""

import concurrent.futures
import logging
import threading
import time
from collections import Counter

import numpy as np

from . import camera_db, camera_stream
from .reid import config as reid_config
from .reid import peopleid_gallery, reid_db, reid_worker

log = logging.getLogger("footfall")

# How often the gate list is re-read from the cameras table, so marking a
# camera "Entry/Exit" (or un-marking it) takes effect without a restart.
GATE_SYNC_INTERVAL_SECONDS = 15

# A track's pending_new_person samples are checked against the shared
# gallery once more before a new identity is created. If at least this
# fraction of them already match one existing person, it's that person —
# typically someone another gate enrolled in the seconds since this track's
# samples were collected.
ENROLL_RECHECK_MATCH_FRACTION = 0.5


def is_gate(cam: dict) -> bool:
    return "entry" in (cam.get("purpose") or "").lower()


class _SharedGallery:
    """The single identity gallery every gate matches against. reload()
    builds a complete new VectorGallery and swaps the reference in one
    assignment, so a gate thread mid-search never sees a half-rebuilt
    matrix (VectorGallery.reload itself updates two arrays separately)."""

    def __init__(self):
        self._gallery = self._build()

    @staticmethod
    def _build() -> peopleid_gallery.VectorGallery:
        return peopleid_gallery.VectorGallery(
            embedding_loader=reid_db.load_all_embeddings,
            similarity_threshold=reid_config.REID_SIMILARITY_THRESHOLD,
            min_margin=reid_config.REID_MIN_MARGIN,
            duplicate_identities_expected=True,  # auto-enrolled — see VectorGallery.__init__
        )

    def reload(self) -> None:
        self._gallery = self._build()

    def best_match(self, embedding, similarity_threshold=None):
        return self._gallery.best_match(embedding, similarity_threshold=similarity_threshold)

    @property
    def person_count(self) -> int:
        return self._gallery.person_count


class _KeepAliveSink:
    """Occupies a CameraStream subscriber slot so the RTSP loop keeps
    running with no viewer; drops every frame it's handed."""

    def put_nowait(self, item):
        pass


class _GateRunner:
    """Per-gate Re-ID state plus a single-worker executor, so a slow
    detection pass delays only this gate's counting — never the video
    broadcast, and never another gate."""

    def __init__(self, camera_id: int, gallery: _SharedGallery):
        self.camera_id = camera_id
        self.state = reid_worker.ReidWorkerState(camera_config=reid_db.get_camera_config(camera_id))
        self.state.gallery = gallery  # replace the private copy with the shared one
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"footfall-{camera_id}")
        self.future: concurrent.futures.Future | None = None
        self.last_submit = 0.0
        self.failed = False


class FootfallService:
    def __init__(self):
        self._gallery: _SharedGallery | None = None
        self._runners: dict[int, _GateRunner] = {}
        self._sinks: dict[int, _KeepAliveSink] = {}
        self._lock = threading.Lock()
        # Serializes every reid_db write + gallery rebuild across gates, so
        # two gates can't both enroll the same newly-arrived person.
        self._write_lock = threading.Lock()
        # (camera_id, track_id) -> (person_id, last_seen): one sighting
        # event per person per track, not one per processed frame.
        self._sighted: dict[tuple[int, int], tuple[int, float]] = {}
        self._started = False

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        reid_db.init_db()
        self._gallery = _SharedGallery()
        if not reid_config.REID_MODEL_PATH:
            log.warning(
                "footfall: Re-ID model osnet_x0_25_msmt17.pth not found in backend/models — using the "
                "ImageNet fallback, which over-counts badly. Run `python -m scripts.fetch_reid_model` from backend/."
            )
        threading.Thread(target=self._sync_loop, daemon=True, name="footfall-gate-sync").start()

    def _sync_loop(self) -> None:
        while True:
            try:
                self._sync_gates()
            except Exception:
                log.exception("footfall: gate sync failed")
            time.sleep(GATE_SYNC_INTERVAL_SECONDS)

    def _sync_gates(self) -> None:
        gate_ids = {c["id"] for c in camera_db.list_cameras() if is_gate(c) and c.get("host")}
        with self._lock:
            added = gate_ids - self._sinks.keys()
            removed = self._sinks.keys() - gate_ids
            for cid in added:
                sink = _KeepAliveSink()
                camera_stream.get_stream(cid).subscribe(sink, is_collector=True)
                self._sinks[cid] = sink
            for cid in removed:
                camera_stream.get_stream(cid).unsubscribe(self._sinks.pop(cid))
                self._runners.pop(cid, None)
        for cid in added:
            log.info("footfall: counting on gate camera %s", cid)
        for cid in removed:
            log.info("footfall: camera %s is no longer a gate, stopped counting", cid)

    # --- per-frame hook (called from camera_stream's read loop) -----------

    def feed(self, camera_id: int, frame: np.ndarray) -> None:
        if self._gallery is None or camera_id not in self._sinks:
            return
        runner = self._runners.get(camera_id)
        if runner is None:
            with self._lock:
                runner = self._runners.setdefault(camera_id, _GateRunner(camera_id, self._gallery))
        if runner.failed or (runner.future is not None and not runner.future.done()):
            return
        # Cheap pre-check of process_frame's own cadence gate, so frames it
        # would ignore anyway aren't pushed through the executor.
        now = time.time()
        if now - runner.last_submit < reid_config.REID_MOT_INTERVAL_SECONDS:
            return
        runner.last_submit = now
        runner.future = runner.executor.submit(self._process, runner, frame)

    def _process(self, runner: _GateRunner, frame: np.ndarray) -> None:
        try:
            result = reid_worker.process_frame(runner.camera_id, frame, runner.state)
            if result:
                self._record(runner.camera_id, result)
        except Exception:
            # Most likely a model that failed to load. Log once and stop
            # counting on this gate rather than retrying every second; the
            # live video is unaffected either way.
            log.exception("footfall: Re-ID failed on camera %s, disabling footfall for it", runner.camera_id)
            runner.failed = True

    # --- turning track results into durable events ------------------------

    def _record(self, camera_id: int, result: dict) -> None:
        now = time.time()
        with self._write_lock:
            for track in result["tracks"]:
                pending = track.get("pending_new_person")
                if pending:
                    person_id = self._enroll(camera_id, track, pending, now)
                    self._sighted[(camera_id, track["track_id"])] = (person_id, now)
                elif track["state"] == reid_db.TRACK_STATE_CONFIRMED and track["person_id"] is not None:
                    key = (camera_id, track["track_id"])
                    prev = self._sighted.get(key)
                    if prev is None or prev[0] != track["person_id"]:
                        reid_db.log_event(camera_id, track["track_id"], reid_db.EVENT_SIGHTING,
                                          track["person_id"], track["confidence"], now)
                        reid_db.touch_person(track["person_id"], now)
                    self._sighted[key] = (track["person_id"], now)
            cutoff = now - reid_config.REID_TRACK_TIMEOUT_SECONDS * 3
            for key in [k for k, (_pid, seen) in self._sighted.items() if seen < cutoff]:
                del self._sighted[key]

    def _enroll(self, camera_id: int, track: dict, samples: list[dict], now: float) -> int:
        embeddings = [np.asarray(s["embedding"], dtype=np.float32) for s in samples]

        votes = Counter()
        for emb in embeddings:
            pid, _score, _margin = self._gallery.best_match(emb)
            if pid is not None:
                votes[pid] += 1
        if votes:
            pid, n = votes.most_common(1)[0]
            if n >= len(embeddings) * ENROLL_RECHECK_MATCH_FRACTION:
                reid_db.log_event(camera_id, track["track_id"], reid_db.EVENT_SIGHTING, pid, track["confidence"], now)
                reid_db.touch_person(pid, now)
                return pid

        person_id = reid_db.create_person(now=now)
        for s, emb in zip(samples, embeddings):
            reid_db.add_embedding(person_id, emb, s["quality_score"], source_camera=camera_id)
        best = max(samples, key=lambda s: s["quality_score"] or 0)
        if best.get("crop_jpeg"):
            reid_db.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            path = reid_db.SNAPSHOTS_DIR / f"person_{person_id}_{int(now)}.jpg"
            path.write_bytes(best["crop_jpeg"])
            reid_db.add_snapshot(person_id, camera_id, str(path), best["quality_score"])
        reid_db.log_event(camera_id, track["track_id"], reid_db.EVENT_NEW_PERSON, person_id, track["confidence"], now)
        self._gallery.reload()
        log.info("footfall: new person %s at camera %s", person_id, camera_id)
        return person_id

    # --- reporting --------------------------------------------------------

    def summary(self) -> dict:
        now = time.time()
        day_start = reid_db._day_start(now)
        gates = [c for c in camera_db.list_cameras() if is_gate(c)]

        with reid_db.get_connection() as conn:
            per_gate = dict(conn.execute(
                "SELECT camera_id, COUNT(DISTINCT person_id) FROM reid_events "
                "WHERE person_id IS NOT NULL AND ts >= ? GROUP BY camera_id", (day_start,),
            ).fetchall())
            # Arrivals per hour = each person counted once, in the hour they
            # were FIRST seen that day (at any gate).
            hourly = {}
            for label, start, end in (("today", day_start, now + 1), ("yesterday", day_start - 86400, day_start)):
                hourly[label] = dict(conn.execute(
                    "SELECT CAST(strftime('%H', first_ts, 'unixepoch', 'localtime') AS INTEGER), COUNT(*) FROM ("
                    "  SELECT person_id, MIN(ts) AS first_ts FROM reid_events"
                    "  WHERE person_id IS NOT NULL AND ts >= ? AND ts < ? GROUP BY person_id"
                    ") GROUP BY 1", (start, end),
                ).fetchall())
            visitors = conn.execute(
                "SELECT p.label, MIN(e.ts), MAX(e.ts), GROUP_CONCAT(DISTINCT e.camera_id) "
                "FROM reid_events e JOIN reid_persons p ON p.id = e.person_id "
                "WHERE e.ts >= ? GROUP BY e.person_id ORDER BY MAX(e.ts) DESC LIMIT 200", (day_start,),
            ).fetchall()

        names = {c["id"]: c["name"] for c in camera_db.list_cameras()}
        gate_rows = [
            {
                "camera_id": g["id"],
                "name": g["name"],
                "unique_today": per_gate.get(g["id"], 0),
                "counting": g["id"] in self._sinks and not getattr(self._runners.get(g["id"]), "failed", False),
            }
            for g in gates
        ]
        # Hours 0-23 so the frontend chart has a stable x-axis; trimmed to
        # the span that actually has traffic (either day), min 8am-6pm.
        active_hours = [h for h in range(24) if hourly["today"].get(h) or hourly["yesterday"].get(h)]
        first_h = min(active_hours + [8])
        last_h = max(active_hours + [18])
        hourly_rows = [
            {"hour": h, "today": hourly["today"].get(h, 0), "yesterday": hourly["yesterday"].get(h, 0)}
            for h in range(first_h, last_h + 1)
        ]
        busiest = max(hourly["today"].items(), key=lambda kv: kv[1])[0] if hourly["today"] else None

        return {
            "unique_today": reid_db.count_unique_today(now),
            "new_today": reid_db.count_new_today(now),
            "returning_today": reid_db.count_returning_today(now),
            "avg_per_gate": round(sum(r["unique_today"] for r in gate_rows) / len(gate_rows), 1) if gate_rows else 0,
            "busiest_hour": busiest,
            "gates": gate_rows,
            "hourly": hourly_rows,
            "visitors": [
                {
                    "person": label,
                    "first_seen": first,
                    "last_seen": last,
                    "gates": [names.get(int(cid), f"Camera {cid}") for cid in (cams or "").split(",") if cid],
                }
                for label, first, last, cams in visitors
            ],
            "model_ready": bool(reid_config.REID_MODEL_PATH),
        }


service = FootfallService()
