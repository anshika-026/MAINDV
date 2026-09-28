"""
OFFICE FIXTURES DETECTOR
==========================

Optional second general-object model for things COCO's 80 classes do
not cover: AC units, cabinets, ceiling fans, the CCTV camera itself,
glass doors/partitions, whiteboards.

Trained by scripts/train_fixtures_model.py on hand-labelled frames from
these specific office cameras (see app/build_fixtures_dataset.py).

Optional by design: if app/models/office_fixtures.pt has not been
trained yet, this quietly returns nothing rather than erroring, so the
rest of detection keeps working.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "office_fixtures.pt"

FIXTURES_CONFIDENCE = float(os.getenv("FIXTURES_CONFIDENCE", "0.35"))

fixtures_model = None
FIXTURES_MODEL_AVAILABLE = False
FIXTURES_CLASS_NAMES: dict[int, str] = {}


def load_fixtures_model():
    global fixtures_model, FIXTURES_MODEL_AVAILABLE, FIXTURES_CLASS_NAMES

    fixtures_model = None
    FIXTURES_MODEL_AVAILABLE = False
    FIXTURES_CLASS_NAMES = {}

    if not MODEL_PATH.is_file():
        print(f"[FIXTURES] Not trained yet - {MODEL_PATH} not found. "
              f"Run app.build_fixtures_dataset then scripts/train_fixtures_model.py.")
        return False

    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
    except Exception as exc:
        print(f"[FIXTURES] Failed to load: {type(exc).__name__}: {exc}")
        return False

    fixtures_model = model
    FIXTURES_MODEL_AVAILABLE = True
    FIXTURES_CLASS_NAMES = {
        int(k): str(v) for k, v in getattr(model, "names", {}).items()
    }

    print(f"[FIXTURES] Loaded {MODEL_PATH.name}, classes: {list(FIXTURES_CLASS_NAMES.values())}")
    return True


load_fixtures_model()


def detect_fixtures(frame):
    """
    Run the fixtures model once. Returns a list of detection dicts in
    the same shape detector.py's other object lists use.
    """
    detections = []

    if frame is None or fixtures_model is None or not FIXTURES_MODEL_AVAILABLE:
        return detections

    try:
        results = fixtures_model(
            frame,
            conf=FIXTURES_CONFIDENCE,
            verbose=False,
        )
    except Exception as exc:
        print(f"[FIXTURES] inference error: {type(exc).__name__}: {exc}")
        return detections

    height, width = frame.shape[:2]

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            try:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                name = FIXTURES_CLASS_NAMES.get(class_id, f"class_{class_id}")

                detections.append({
                    "name": name,
                    "class_name": name,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "track_id": None,
                    "hits": 999,  # static fixture, always "confirmed"
                    "from_fixtures_model": True,
                })

            except Exception as exc:
                print(f"[FIXTURES] skipped malformed box: {type(exc).__name__}: {exc}")

    return detections


def get_fixtures_status():
    return {
        "fixtures_model_available": FIXTURES_MODEL_AVAILABLE,
        "fixtures_model_path": str(MODEL_PATH),
        "fixtures_model_exists": MODEL_PATH.is_file(),
        "fixtures_classes": list(FIXTURES_CLASS_NAMES.values()),
        "fixtures_confidence": FIXTURES_CONFIDENCE,
    }
