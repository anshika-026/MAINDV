"""
service.py

Runs Staff Count on live cameras. camera_stream.py hands every frame to
feed(); for each camera configured as an entrance (staff_camera_config, set
on the Staff Count page) about `process_fps` frames a second go through:

  YOLOv8 person detection + ByteTrack (shared per camera, person_detection.py)
  -> StaffCameraTracker (office ROI, entry-line crossings, identity votes)
  -> StaffOccupancyManager (one for all cameras; the source of truth)

Identity: the existing face-recognition classifier (face_pipeline.py) is run
on the head area of each tracked body, only while that track is unidentified
and capped per frame. Body appearance for matching exits comes from the
existing Re-ID model (reid/reid_embedding.py). With face recognition switched
off, or no trained classifier, counting runs anonymously (everyone inside is
counted, nobody is named).

Controlled by the "staff_count" analytics switch; entrance cameras are kept
streaming only while it's on. Each camera runs on its own worker thread and
skips a frame if the previous one is still being processed, so a slow pass
never delays the video.
"""

import concurrent.futures
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from app import analytics_settings, camera_db, lifecycle, resilience
from app.staff import config as staff_config
from app.staff import occupancy
from app.staff.occupancy import StaffOccupancyManager
from app.staff.tracker import StaffCameraTracker

log = logging.getLogger("staff")

MODEL_PATH = str(Path(__file__).resolve().parent.parent.parent / "models" / "yolov8n.pt")
SYNC_INTERVAL_SECONDS = 15


class _KeepAliveSink:
    def put_nowait(self, item):
        pass


def get_camera_config(camera_id: int) -> dict | None:
    with occupancy._conn() as conn:
        row = conn.execute("SELECT * FROM staff_camera_config WHERE camera_id = ?", (camera_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["line"] = json.loads(d["line"]) if d["line"] else None
    d["roi"] = json.loads(d["roi"]) if d["roi"] else None
    d["enabled"] = bool(d["enabled"])
    return d


def list_camera_configs() -> list[dict]:
    with occupancy._conn() as conn:
        ids = [r[0] for r in conn.execute("SELECT camera_id FROM staff_camera_config")]
    return [get_camera_config(i) for i in ids]


def set_camera_config(camera_id: int, enabled: bool, line, inside_sign: int, roi) -> dict:
    with occupancy._conn() as conn:
        conn.execute(
            "INSERT INTO staff_camera_config (camera_id, enabled, line, inside_sign, roi, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(camera_id) DO UPDATE SET enabled=excluded.enabled, line=excluded.line, inside_sign=excluded.inside_sign, "
            "roi=excluded.roi, updated_at=excluded.updated_at",
            (camera_id, int(enabled), json.dumps(line) if line else None, 1 if inside_sign >= 0 else -1,
             json.dumps(roi) if roi else None, time.time()),
        )
    service.reload_camera(camera_id)
    return get_camera_config(camera_id)


class _Identity:
    """Face identity (existing classifier) and body appearance (existing Re-ID
    model), loaded lazily and shared by all cameras."""

    def __init__(self):
        self._embedder = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        from app.face_pipeline import CameraFacePipeline as P

        return analytics_settings.enabled("face_recognition") and P._get_classifier() is not None

    def identify(self, frame: np.ndarray, bbox) -> tuple[str | None, float]:
        from app.face_pipeline import CameraFacePipeline as P

        clf = P._get_classifier()
        if clf is None:
            return None, 0.0
        P._ensure_arcface_loaded()
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame.shape[:2]
        head_bottom = y1 + int((y2 - y1) * 0.45)          # head and shoulders
        pad = int((x2 - x1) * 0.15)
        crop = frame[max(0, y1 - pad):min(h, head_bottom), max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size == 0 or min(crop.shape[:2]) < 24:
            return None, 0.0
        faces = P._arcface.get(crop)
        if not faces:
            return None, 0.0
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        proba = clf.predict_proba(face.normed_embedding.reshape(1, -1))[0]
        best = int(np.argmax(proba))
        return str(clf.classes_[best]), float(proba[best])

    def embed(self, frame: np.ndarray, bbox):
        with self._lock:
            if self._embedder is None:
                from app.reid import config as reid_config
                from app.reid.reid_embedding import BodyReIdEmbedder

                self._embedder = BodyReIdEmbedder(reid_config.REID_MODEL_NAME, reid_config.REID_MODEL_PATH, reid_config.REID_DEVICE)
        return self._embedder.embed(frame, list(bbox))


class _Camera:
    def __init__(self, camera_id: int, manager, cfg: dict, cam_cfg: dict):
        self.camera_id = camera_id
        self.tracker = StaffCameraTracker(camera_id, manager, cam_cfg["line"], cam_cfg["inside_sign"], cam_cfg["roi"], cfg)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"staff-{camera_id}")
        self.future = None
        self.last_processed = None


class StaffService:
    def __init__(self):
        self.cfg = staff_config.SETTINGS
        self.manager: StaffOccupancyManager | None = None
        self.identity = _Identity()
        self._cams: dict[int, _Camera] = {}
        self._health: dict[int, "resilience.FeatureHealth"] = {}
        self._sinks: dict[int, _KeepAliveSink] = {}
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        occupancy.init_db()
        self.manager = StaffOccupancyManager(self.cfg)
        for cc in list_camera_configs():
            self.reload_camera(cc["camera_id"])
        threading.Thread(target=self._loop, daemon=True, name="staff-sync").start()

    def reload_camera(self, camera_id: int) -> None:
        if self.manager is None:
            return
        cc = get_camera_config(camera_id)
        with self._lock:
            self._cams.pop(camera_id, None)
            if cc and cc["enabled"] and cc["line"]:
                self._cams[camera_id] = _Camera(camera_id, self.manager, self.cfg, cc)
        try:
            self.sync()
        except Exception:
            log.exception("staff: sync after config change failed")

    def anonymous_mode(self) -> bool:
        try:
            return not self.identity.available()
        except Exception:
            return True

    def _loop(self) -> None:
        while True:
            try:
                self.sync()
                if self.manager:
                    self.manager.maybe_end_of_day()
            except Exception:
                log.exception("staff: background pass failed")
            if lifecycle.wait(SYNC_INTERVAL_SECONDS):
                return
    def sync(self) -> None:
        from app import camera_stream

        on = analytics_settings.enabled("staff_count")
        wanted = {cid for cid in self._cams if on and camera_db.is_streamable(camera_db.get_camera(cid))}
        for cid in wanted - self._sinks.keys():
            sink = _KeepAliveSink()
            camera_stream.get_stream(cid).subscribe(sink, is_collector=True, wants_frames=False)
            self._sinks[cid] = sink
            log.info("staff: counting at entrance camera %s", cid)
        for cid in self._sinks.keys() - wanted:
            camera_stream.get_stream(cid).unsubscribe(self._sinks.pop(cid))

    # ---- per frame ----------------------------------------------------------------

    # Tracked people come from the shared per-camera detector
    # (person_detection.py), paced at process_fps for entrance cameras.
    def wants(self, camera_id: int) -> float:
        cam = self._cams.get(camera_id)
        if cam is None or not analytics_settings.enabled("staff_count") or camera_id not in self._sinks:
            return 0.0
        if not self.health(camera_id).allow():
            return 0.0
        return float(self.cfg["process_fps"])

    def deliver(self, camera_id: int, frame: np.ndarray, people: list[dict], ts: float) -> None:
        cam = self._cams.get(camera_id)
        if cam is None or (cam.future is not None and not cam.future.done()):
            return  # previous frame still being processed: skip this one
        cam.future = cam.executor.submit(self._process, cam, frame, ts, people)

    def _process(self, cam: _Camera, frame: np.ndarray, ts: float, people: list[dict]) -> None:
        try:
            people = [p for p in people if p["track_id"] is not None and p["confidence"] >= self.cfg["detection_confidence"]]
            identify = (lambda b: self.identity.identify(frame, b)) if not self.anonymous_mode() else None
            cam.tracker.update(people, frame.shape, ts, identify, lambda b: self.identity.embed(frame, b))
            # Faces confirmed at the entrance teach the appearance hand-off
            # (appearance.py), so these people can be named on desk cameras
            # that only see their backs. learn() rate-limits per employee.
            if analytics_settings.enabled("appearance_handoff"):
                from app import appearance

                for t in cam.tracker.tracks.values():
                    if t.employee_id and t.last_seen == ts and t.bbox:
                        appearance.gallery.learn(t.employee_id, frame, t.bbox, ts)
            cam.last_processed = ts
        except Exception as e:
            # Paused on a backoff, never permanently; tracker state is kept
            # (occupancy lives in the DB-backed manager, not the tracker).
            self.health(cam.camera_id).failure(e)
        else:
            self.health(cam.camera_id).success()

    def health(self, camera_id: int):
        h = self._health.get(camera_id)
        if h is None:
            h = self._health.setdefault(camera_id, resilience.health_for("staff_count", camera_id))
        return h

    # ---- reporting ----------------------------------------------------------------

    def count(self, camera_ids: set[int] | None = None) -> dict:
        return self.manager.counts(anonymous_mode=self.anonymous_mode(), camera_ids=camera_ids) if self.manager else {}

    def debug(self, camera_id: int) -> dict:
        from app import employee_directory

        cam = self._cams.get(camera_id)
        cc = get_camera_config(camera_id)
        tracks = cam.tracker.debug() if cam else []
        for t in tracks:
            emp = employee_directory.get_employee(t["employee_id"]) if t["employee_id"] else None
            t["name"] = emp["name"] if emp else None
        w, h = (cam.tracker.frame_size if cam and cam.tracker.frame_size else (None, None))
        return {
            "camera_id": camera_id, "configured": bool(cam), "failed": self.health(camera_id).failed if cam else False,
            "line": cc["line"] if cc else None, "inside_sign": cc["inside_sign"] if cc else 1, "roi": cc["roi"] if cc else None,
            "frame_w": w, "frame_h": h, "tracks": tracks, "count": self.count(),
            "processing": bool(cam and cam.last_processed and time.time() - cam.last_processed < 3),
        }

    def status(self) -> dict:
        return {
            "switch_on": analytics_settings.enabled("staff_count"),
            "anonymous_mode": self.anonymous_mode(),
            "cameras": [{"camera_id": cid, "failed": self.health(cid).failed, "state": self.health(cid).state,
                         "processing": bool(c.last_processed and time.time() - c.last_processed < 3)} for cid, c in self._cams.items()],
            "settings": self.cfg,
        }


service = StaffService()

from app import person_detection  # noqa: E402

person_detection.service.register("staff_count", service.wants, service.deliver)
