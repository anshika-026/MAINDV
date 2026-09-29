"""
person_detection.py

One shared person detector + tracker per camera, feeding every analytic that
needs tracked people on it: unique footfall (footfall.py), Staff Count
(staff/service.py) and intrusion zones (intrusion.py).

Before this, each of those ran its own YOLO on the same frames, so an
entrance camera doing footfall + staff count + intrusion paid for three
person-detection passes per frame. On this CPU-bound laptop a footfall look
was measured taking ~5 s instead of ~0.5 s, so walkers were seen once
instead of the 3 times needed to be counted.

Now camera_stream.py hands each frame here; per camera:
  - YOLOv8n + ByteTrack (ultralytics, persistent track ids) runs ONCE, at the
    highest rate any active consumer wants for that camera (staff count 5 fps,
    footfall 2 fps, intrusion 1 fps), on its own worker thread, skipping a
    frame if the previous one is still being detected.
  - The tracked people ({track_id, bbox, confidence}) are delivered to each
    consumer when its own interval is due. Consumers do their per-person work
    (Re-ID, face identity, zone checks) on their own worker threads, so a slow
    consumer never delays detection or the others.
  - No consumer wants a camera (features switched off, no gates / entrances /
    zones on it): nothing runs for it.

A consumer is registered with register(name, wants, deliver):
  wants(camera_id) -> frames per second it wants from that camera (0 = none)
  deliver(camera_id, frame, people, ts) -> called when due
"""

import concurrent.futures
import logging
import threading
import time
from pathlib import Path

import numpy as np

from . import config, resilience

log = logging.getLogger("person_detection")

MODEL_PATH = str(Path(config.MODEL_DIR) / "yolov8n.pt")
# Lowest confidence any consumer uses (footfall's PEOPLEID_MOT_CONFIDENCE);
# consumers that want stricter filter the delivered people themselves.
DETECT_CONF = 0.4
IMGSZ = 640
TRACKER = "bytetrack.yaml"


class _Cam:
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.model = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"persons-{camera_id}")
        self.future = None
        self.last_run = 0.0
        self.last_delivery: dict[str, float] = {}  # consumer -> next due time
        # Replaces a permanent `failed` flag: a failure pauses detection on a
        # backoff and the model is reloaded after repeated failures, so a
        # transient error can't silently stop footfall, Staff Count and
        # intrusion on this camera for the rest of the process lifetime.
        self.health = resilience.health_for("person_detection", camera_id)
        self.ms = []  # recent detection times, for status

    @property
    def failed(self) -> bool:
        return self.health.failed


class SharedPersonDetection:
    def __init__(self):
        self._consumers: list[tuple[str, object, object]] = []
        self._cams: dict[int, _Cam] = {}
        self._lock = threading.Lock()

    def register(self, name: str, wants, deliver) -> None:
        self._consumers.append((name, wants, deliver))

    def _wanted(self, camera_id: int) -> dict[str, float]:
        out = {}
        for name, wants, _ in self._consumers:
            try:
                fps = float(wants(camera_id) or 0)
            except Exception:
                fps = 0.0
            if fps > 0:
                out[name] = fps
        return out

    def feed(self, camera_id: int, frame: np.ndarray) -> None:
        wanted = self._wanted(camera_id)
        if not wanted:
            return
        cam = self._cams.get(camera_id)
        if cam is None:
            with self._lock:
                cam = self._cams.setdefault(camera_id, _Cam(camera_id))
        if not cam.health.allow():
            return
        now = time.time()
        if now - cam.last_run < 1.0 / max(wanted.values()):
            return
        if cam.future is not None and not cam.future.done():
            return  # still detecting the previous frame: skip, don't queue
        cam.last_run = now
        cam.future = cam.executor.submit(self._run, cam, frame, now, wanted)

    def _run(self, cam: _Cam, frame: np.ndarray, ts: float, wanted: dict[str, float]) -> None:
        try:
            if cam.model is None:
                from ultralytics import YOLO

                from app import models

                cam.model = YOLO(str(models.require("yolo_person")))
            t = time.time()
            res = cam.model.track(frame, persist=True, tracker=TRACKER, classes=[0], conf=DETECT_CONF,
                                  imgsz=IMGSZ, verbose=False)[0]
            cam.ms = (cam.ms + [round((time.time() - t) * 1000)])[-20:]
        except Exception as e:
            if cam.health.failure(e):
                cam.model = None  # reload the model (and a fresh tracker) next time
            return
        cam.health.success()
        people = []
        if res.boxes is not None and len(res.boxes):
            ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(res.boxes)
            for tid, box, conf in zip(ids, res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                # track_id None: detected this frame but not (yet) a confirmed
                # track. Zone checks can use it; trackers skip it.
                people.append({"track_id": None if tid is None else int(tid), "bbox": [float(v) for v in box], "confidence": float(conf)})
        # Each consumer is due every 1/fps on a fixed schedule and gets the
        # detection frame nearest its due time (within half a detection step),
        # so e.g. 2 fps from a 5 fps detector really is 2 fps, not the 1.7 a
        # "0.5 s since the last one" rule rounds down to.
        half_step = 0.5 / max(wanted.values())
        for name, _, deliver in self._consumers:
            fps = wanted.get(name)
            if not fps:
                continue
            due = cam.last_delivery.get(name)
            if due is not None and ts + half_step < due:
                continue
            interval = 1.0 / fps
            cam.last_delivery[name] = ts + interval if due is None else max(due + interval, ts)  # next due; never a backlog
            try:
                deliver(cam.camera_id, frame, people, ts)
            except Exception:
                log.exception("person consumer %s failed on camera %s", name, cam.camera_id)

    def shutdown(self) -> None:
        with self._lock:
            cams = list(self._cams.values())
        for c in cams:
            c.executor.shutdown(wait=False, cancel_futures=True)

    def status(self) -> list[dict]:
        return [
            {"camera_id": cid, "failed": c.failed, "state": c.health.state, "wanted": self._wanted(cid),
             "detect_ms": round(sum(c.ms) / len(c.ms)) if c.ms else None,
             "running": time.time() - c.last_run < 3}
            for cid, c in self._cams.items()
        ]


service = SharedPersonDetection()
