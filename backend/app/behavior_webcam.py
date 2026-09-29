"""
behavior_webcam.py

Server-side port of the standalone `src/real_time_emotion.py` from the
facial-expression-recognition branch, for the Behavior Analytics page.

This is deliberately a FAITHFUL port, not a reinterpretation — same model
file, same 48x48 grayscale input, same Haar cascades, same label list,
same smoothing windows, same attention rules, same green/red meaning:

    emotion model   models/emotion_model_v3.keras   (unchanged, from the branch)
    smoothing       last 10 predictions, averaged
    attention       last 15 frames, distracted when mean < 0.5
    box colour      green when engaged, red when distracted
    label           "Distracted", else "<Emotion>: <confidence>%"

Two things had to change, both forced by this environment:

1. Frame source. The original opened its own cv2.VideoCapture(0) on the
   machine running it, which is useless for a browser page. Frames now
   arrive from the user's laptop webcam via the Behavior Analytics
   endpoint.

2. Face and eye detection. The original used the Haar cascades
   `haarcascade_frontalface_default.xml` and
   `haarcascade_eye_tree_eyeglasses.xml`, but this project runs OpenCV
   5.0, which removed `cv2.CascadeClassifier` altogether — that code
   cannot run here at all. InsightFace (already installed, already used by
   the live pipeline) replaces both: it returns the face box AND 5-point
   landmarks, two of which are the eye centres. The attention rules and
   their thresholds below are unchanged — only where the eye coordinates
   come from is different, and landmark eyes are more reliable than a
   cascade sweep over the upper face.

Nothing in face_pipeline.py is touched and no RTSP camera is involved —
this path is entirely separate from the live-camera recognition pipeline.

State (the smoothing deques) is kept per face key so two people in frame
smooth independently rather than averaging into each other.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque

import cv2
import numpy as np

log = logging.getLogger("behavior_webcam")

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
EMOTION_KERAS_MODEL = os.environ.get(
    "EMOTION_KERAS_MODEL", os.path.join(_MODELS_DIR, "emotion_model_v3.keras")
)

# Exactly the branch's label order — the model's output indices mean
# nothing without it, so this must not be reordered or "cleaned up".
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral & Engaged",
    "Sad",
    "Surprise",
]

# Attention thresholds, verbatim from the branch.
MAX_EYE_OFFSET = 0.12   # eye-midpoint drift from face centre -> head turned
MAX_EYE_TILT = 0.10     # one eye higher than the other -> head tilted
PREDICTION_HISTORY = 10  # frames averaged before naming an emotion
ATTENTION_HISTORY = 15   # frames averaged before calling someone distracted

_model = None
_model_lock = threading.Lock()
_load_health = None  # resilience.FeatureHealth, created on first use

# Per-face rolling state, keyed the same way the endpoint keys a face.
_history: dict[int, dict] = {}
_history_lock = threading.Lock()


def _ensure_model():
    """Loads the Keras model once, lazily. A missing model file or missing
    TensorFlow degrades to "no emotion" rather than breaking the page."""
    global _model, _load_health
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if _load_health is None:
            from app import resilience

            _load_health = resilience.health_for("webcam_emotion_model")
        # Retried on a backoff rather than given up on for good.
        if not _load_health.allow():
            return None
        try:
            import tensorflow as tf

            _model = tf.keras.models.load_model(EMOTION_KERAS_MODEL)
            log.info("emotion model loaded: %s", EMOTION_KERAS_MODEL)
            _load_health.success()
        except Exception as e:
            _load_health.failure(e)
        return _model


def facing_straight(box, kps) -> bool:
    """The branch's facing_straight() rules, applied to InsightFace's eye
    landmarks instead of a Haar eye sweep (see the module docstring).

    Someone counts as facing straight when both eyes sit evenly either
    side of the face centre and level with each other. Eye contact is not
    required. Thresholds are the branch's own values, unchanged.
    """
    if kps is None or len(kps) < 2:
        return False  # no eyes located -> head turned, looking down, or closed

    fx1, fy1, fx2, fy2 = box
    w, h = max(1.0, fx2 - fx1), max(1.0, fy2 - fy1)
    # InsightFace keypoint order: left eye, right eye, nose, mouth corners.
    (x1, y1), (x2, y2) = kps[0], kps[1]

    if abs(x1 - x2) < w * 0.2:
        return False  # both points on the same side = bad landmark fit
    eyes_midpoint = (x1 + x2) / 2
    face_centre_x = fx1 + w / 2
    if abs(eyes_midpoint - face_centre_x) > w * MAX_EYE_OFFSET:
        return False  # head turned left/right
    if abs(y1 - y2) > h * MAX_EYE_TILT:
        return False  # head tilted
    return True


def _state(key: int) -> dict:
    with _history_lock:
        return _history.setdefault(
            key,
            {
                "predictions": deque(maxlen=PREDICTION_HISTORY),
                "attention": deque(maxlen=ATTENTION_HISTORY),
            },
        )


def analyze(frame_bgr: np.ndarray) -> list[dict]:
    """Detect faces in one webcam frame and return, per face, the same
    result the standalone script would have drawn on screen.

    Each entry: bbox, whether they're engaged, the smoothed emotion and
    confidence, and the ready-made label/colour so the page renders the
    identical thing the original window did.
    """
    from app.face_pipeline import CameraFacePipeline

    CameraFacePipeline._ensure_arcface_loaded()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detected = CameraFacePipeline._arcface.get(frame_bgr)

    model = _ensure_model()
    out = []
    seen_keys = set()

    for face in detected:
        x1, y1, x2, y2 = (int(max(0, v)) for v in face.bbox[:4])
        x, y, w, h = x1, y1, max(1, x2 - x1), max(1, y2 - y1)
        key = (int(y + h / 2) // 80) * 1000 + (int(x + w / 2) // 80)
        seen_keys.add(key)
        st = _state(key)
        face_gray = gray[y : y + h, x : x + w]
        if face_gray.size == 0:
            continue

        # --- attention, smoothed over the last ATTENTION_HISTORY frames ---
        st["attention"].append(
            1 if facing_straight((x1, y1, x2, y2), getattr(face, "kps", None)) else 0
        )
        distracted = float(np.mean(st["attention"])) < 0.5

        # --- emotion, averaged over the last PREDICTION_HISTORY frames ---
        emotion, confidence = None, 0.0
        if model is not None:
            face = cv2.resize(face_gray, (48, 48)).astype("float32") / 255.0
            face = face.reshape(1, 48, 48, 1)
            try:
                # model(x) rather than model.predict(x): predict() rebuilds a
                # tf.data pipeline per call, which measured ~6.9s per frame
                # here for a single 48x48 sample. Calling the model directly
                # is the same maths on the same weights, without that
                # per-call machinery.
                st["predictions"].append(np.asarray(model(face, training=False))[0])
                averaged = np.mean(st["predictions"], axis=0)
                idx = int(np.argmax(averaged))
                emotion = EMOTION_LABELS[idx]
                confidence = float(averaged[idx]) * 100.0
            except Exception:
                log.exception("emotion prediction failed for one face")

        # Distracted replaces the emotion entirely, and turns the box red —
        # same precedence as the original.
        if distracted:
            label = "Distracted"
        elif emotion is not None:
            label = f"{emotion}: {confidence:.1f}%"
        else:
            label = None

        out.append({
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "track_id": key,
            "distracted": bool(distracted),
            "emotion": emotion,
            "confidence": round(confidence, 1),
            "label": label,
            "color": "red" if distracted else "green",
        })

    # Drop rolling state for faces that have left, so histories don't grow
    # forever or get re-used by a different person later.
    with _history_lock:
        for gone in [k for k in _history if k not in seen_keys]:
            _history.pop(gone, None)

    return out
