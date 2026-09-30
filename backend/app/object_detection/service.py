"""
service.py — runs the vendored detector (detector.run_full_detection) as
one more analytics consumer of the live camera frames.

  - camera_stream.py hands every decoded frame to feed(); only the newest
    frame per camera is kept, and only when that camera is due
    (OBJECT_DETECTION_FPS per camera, default one look every 5 s).
  - ONE worker thread runs detection for all cameras, oldest-due first.
    The detector keeps its models and caches in module globals and costs
    ~0.6 s per frame on CPU, so serial processing is both the thread-safe
    and the CPU-bounded choice: it can never use more than one core, and a
    slow cycle only delays the next look, it never queues work up.
  - Only the four target classes are reported: backpack, handbag, bottle,
    laptop. PPE, phone and fixture paths of the module stay off.
  - Models come from MODEL_DIR (models.py specs od_general, od_specialist);
    the module's zero-shot fallbacks (YOLO-World, RT-DETR, CLIP) download
    weights on first use, so they stay off unless OBJECT_DETECTION_FALLBACK
    is enabled and offline mode is off.
  - Failures (missing model, inference error) pause detection on the
    shared backoff (resilience.py) and the detector is reloaded after
    repeated failures; they never switch the feature off for good.
  - Off by default: the "object_detection" analytics switch must be turned
    on (it costs CPU on every analytics camera).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from types import SimpleNamespace

from app import analytics_settings, config, lifecycle, models, resilience

log = logging.getLogger("object_detection")

TARGET_CLASSES = ("backpack", "handbag", "bottle", "laptop")

# Looks per second per camera. 0.2 = every 5 s; the module's authors ran it
# at 0.15 on a CPU-only box. Raise on a GPU host.
FPS = float(os.environ.get("OBJECT_DETECTION_FPS", "0.2"))
FALLBACK_ENABLED = os.environ.get("OBJECT_DETECTION_FALLBACK", "false").strip().lower() in ("1", "true", "yes", "on")
# A camera whose last result is older than this many intervals is reported stale.
STALE_AFTER_INTERVALS = 4

# Tuned values from the module's own .env.example (each justified there by a
# real-footage measurement). setdefault: the environment still wins.
_DETECTOR_DEFAULTS = {
    "IMC_DEMO_MODE": "1",               # report only the target classes
    "DETECT_DEBUG": "0",                # no per-frame debug blocks
    "IMC_SPECIALIST_CONFIDENCE": "0.15",
    "GENERAL_CONFIDENCE": "0.45",
    "SMALL_OBJECT_CONFIDENCE": "0.32",
    "GENERAL_IMGSZ": "640",
    "GENERAL_IOU": "0.30",
    "YOLOS_FALLBACK_CONFIDENCE": "0.5",
    "YOLOS_FALLBACK_MIN_INTERVAL_S": "6",
    "PPE_MODE": "crop",
}


def _load_detector() -> SimpleNamespace:
    """Imports the vendored module (loads its models). Raises
    ModelMissingError with the fix if the required model isn't provisioned."""
    general = models.require("od_general")
    os.environ.setdefault("OBJECT_DETECTION_MODEL_DIR", str(config.MODEL_DIR))
    os.environ.setdefault("GENERAL_MODEL", str(general))
    fallback = FALLBACK_ENABLED and not config.MODEL_OFFLINE_MODE
    os.environ["YOLOS_FALLBACK_ENABLED"] = "1" if fallback else "0"
    for k, v in _DETECTOR_DEFAULTS.items():
        os.environ.setdefault(k, v)

    from . import detector, object_tracker

    if not detector.GENERAL_MODEL_AVAILABLE:
        detector.load_general_model()  # retried here on every recovery attempt
    if not detector.GENERAL_MODEL_AVAILABLE:
        raise models.ModelMissingError(f"object detection model could not be loaded: {detector.GENERAL_LOAD_ERROR}")
    log.info("object detection ready (general model %s, specialist %s, fallback %s)",
             os.path.basename(str(detector.GENERAL_MODEL_PATH)),
             "present" if models.present(models.get("od_specialist")) else "missing (general model only)",
             "on" if fallback else "off")
    return SimpleNamespace(run=detector.run_full_detection, new_tracker=object_tracker.ObjectTracker)


def _clean(obj: dict, frame_w: int, frame_h: int) -> dict | None:
    name = str(obj.get("name", "")).lower()
    if name not in TARGET_CLASSES:
        return None
    box = obj.get("bbox") or [obj.get("x1"), obj.get("y1"), obj.get("x2"), obj.get("y2")]
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    except (TypeError, ValueError):
        return None
    x1, x2 = max(0, min(x1, frame_w)), max(0, min(x2, frame_w))
    y1, y2 = max(0, min(y1, frame_h)), max(0, min(y2, frame_h))
    if x2 <= x1 or y2 <= y1:
        return None
    color = obj.get("color")
    return {
        "class": name,
        "confidence": round(float(obj.get("confidence") or 0.0), 3),
        "bbox": [x1, y1, x2, y2],
        "track_id": obj.get("track_id"),
        "color": color if isinstance(color, str) else None,
    }


class ObjectDetectionService:
    def __init__(self, loader=_load_detector, clock=time.time):
        self._loader = loader
        self._clock = clock
        self._detector: SimpleNamespace | None = None
        self._lock = threading.Lock()
        self._pending: dict[int, object] = {}       # camera_id -> newest frame waiting
        self._last_run: dict[int, float] = {}
        self._results: dict[int, dict] = {}
        self._trackers: dict[int, object] = {}
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self.health = resilience.health_for("object_detection")

    # --- input -------------------------------------------------------------

    def enabled(self) -> bool:
        return analytics_settings.enabled("object_detection")

    def feed(self, camera_id: int, frame) -> None:
        """Called by the camera read loop for every frame; cheap when not due."""
        if frame is None or not self.enabled():
            return
        if self._clock() - self._last_run.get(camera_id, 0.0) < 1.0 / max(FPS, 1e-6):
            return
        with self._lock:
            self._pending[camera_id] = frame
        self._wake.set()

    # --- worker ------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="object-detection")
        self._thread.start()

    def _loop(self) -> None:
        while not lifecycle.stopping():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            self.run_pending()

    def run_pending(self) -> int:
        """Processes every waiting frame, least recently served camera first.
        Returns how many cameras were processed."""
        with self._lock:
            batch = sorted(self._pending.items(), key=lambda kv: self._last_run.get(kv[0], 0.0))
            self._pending.clear()
        done = 0
        for camera_id, frame in batch:
            if lifecycle.stopping() or not self.enabled():
                break
            if not self.health.allow():
                break  # backing off; newer frames will arrive when due
            self._last_run[camera_id] = self._clock()
            try:
                result = self._detect(camera_id, frame)
            except Exception as e:
                if self.health.failure(e):
                    self._detector = None          # reload models on the next attempt
                    self._trackers.clear()
                continue
            self.health.success()
            with self._lock:
                self._results[camera_id] = result
            done += 1
        return done

    def _detect(self, camera_id: int, frame) -> dict:
        if self._detector is None:
            self._detector = self._loader()
        tracker = self._trackers.get(camera_id)
        if tracker is None:
            tracker = self._trackers[camera_id] = self._detector.new_tracker()
        t0 = time.perf_counter()
        raw = self._detector.run(frame, object_tracker=tracker, run_ppe=False) or {}
        h, w = frame.shape[:2]
        objects = [o for o in (_clean(x, w, h) for x in raw.get("objects", []) if isinstance(x, dict)) if o]
        counts = {c: 0 for c in TARGET_CLASSES}
        for o in objects:
            counts[o["class"]] += 1
        return {"camera_id": camera_id, "ts": self._clock(), "objects": objects, "counts": counts,
                "frame_w": int(w), "frame_h": int(h), "latency_ms": round((time.perf_counter() - t0) * 1000)}

    # --- output ------------------------------------------------------------

    def latest(self, camera_ids: set[int] | None = None) -> list[dict]:
        now = self._clock()
        stale_after = STALE_AFTER_INTERVALS / max(FPS, 1e-6)
        with self._lock:
            rows = [dict(r) for cid, r in sorted(self._results.items()) if camera_ids is None or cid in camera_ids]
        for r in rows:
            r["stale"] = now - r["ts"] > stale_after
        return rows

    def forget(self, camera_id: int) -> None:
        with self._lock:
            self._results.pop(camera_id, None)
            self._pending.pop(camera_id, None)
        self._trackers.pop(camera_id, None)

    def status(self) -> dict:
        return {"enabled": self.enabled(), "fps_per_camera": FPS, "fallback": FALLBACK_ENABLED and not config.MODEL_OFFLINE_MODE,
                "loaded": self._detector is not None, "classes": list(TARGET_CLASSES),
                "cameras": sorted(self._results), "health": self.health.snapshot()}


service = ObjectDetectionService()
