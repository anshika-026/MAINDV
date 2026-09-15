"""
verify_face_detector.py

Standalone sanity check for the YOLO face-detection weight file — run this
BEFORE touching any live camera. Loads the detector from
app.face_pipeline.YOLO_FACE_WEIGHTS, runs it against a single local image,
prints what it found, and confirms the resulting crop still produces a
valid ArcFace embedding (i.e. the existing InsightFace/ArcFace pipeline is
untouched and still works end to end with a real detection as input,
not just a hand-picked crop).

Usage:
    cd backend
    python scripts/verify_face_detector.py path/to/image.jpg

If no path is given, looks for any image directly under backend/data/ or
backend/data/_test_samples/ as a convenience for repeat runs.

Does NOT touch any camera, the database, or the training pipeline — this
is purely "does the detector load and does its output still work",
step-by-step, so a failure points at exactly one thing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from app.face_pipeline import CameraFacePipeline, YOLO_FACE_WEIGHTS


def find_test_image() -> Path | None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for candidate_dir in (data_dir, data_dir / "_test_samples"):
            matches = sorted(candidate_dir.glob(pattern)) if candidate_dir.exists() else []
            if matches:
                return matches[0]
    return None


def main():
    print(f"YOLO_FACE_WEIGHTS resolves to: {YOLO_FACE_WEIGHTS}")

    weights_path = Path(YOLO_FACE_WEIGHTS)
    if not weights_path.exists():
        print(f"FAIL: weight file not found at {weights_path}")
        print("This is expected until a real yolov8n-face.pt checkpoint is placed there.")
        sys.exit(1)

    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = find_test_image()
        if image_path is None:
            print("FAIL: no test image given and none found under backend/data/")
            print("Usage: python scripts/verify_face_detector.py path/to/image.jpg")
            sys.exit(1)
    print(f"Using test image: {image_path}")

    if not image_path.exists():
        print(f"FAIL: image not found at {image_path}")
        sys.exit(1)

    # --- Step 1: load YOLO -------------------------------------------------
    print("\n[1/4] Loading YOLO face detector...")
    try:
        CameraFacePipeline._ensure_yolo_loaded()
    except Exception as e:
        print(f"FAIL: YOLO failed to load — {type(e).__name__}: {e}")
        sys.exit(1)
    print("OK: YOLO loaded successfully.")

    # --- Step 2: run detection on the test image ---------------------------
    print("\n[2/4] Running detection on test image...")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"FAIL: could not read image at {image_path} (corrupt file or unsupported format)")
        sys.exit(1)

    # Bypass __init__ (which builds a ByteTrack instance and expects a real
    # camera_id) — this script only needs the two model-calling methods, not
    # a full per-camera pipeline. _yolo/_arcface are class-level, so a bare
    # instance sees them once loaded above.
    pipeline = CameraFacePipeline.__new__(CameraFacePipeline)
    detections = pipeline._detect_faces(img)
    print(f"OK: {len(detections)} face(s) detected.")
    for i, (x1, y1, x2, y2, conf) in enumerate(detections):
        print(f"  face {i}: bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) confidence={conf:.3f}")

    if len(detections) == 0:
        print("FAIL: no faces detected in the test image — try a clearer/frontal test photo.")
        sys.exit(1)

    # --- Step 3: crop the best detection ------------------------------------
    # Same padded crop feed_frame() now produces (CameraFacePipeline._pad_bbox)
    # — not a bbox-tight crop — so this test reflects the real pipeline.
    print("\n[3/4] Cropping best detection (with the same margin the live pipeline uses)...")
    best = max(detections, key=lambda d: d[4])
    x1, y1, x2, y2 = [max(0, int(v)) for v in best[:4]]
    x1, y1, x2, y2 = pipeline._pad_bbox(x1, y1, x2, y2, img.shape)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        print("FAIL: best detection produced an empty crop.")
        sys.exit(1)
    print(f"OK: crop shape = {crop.shape}")

    # --- Step 4: feed the crop through the EXISTING, UNCHANGED ArcFace path ---
    print("\n[4/4] Running the crop through InsightFace/ArcFace (unchanged pipeline)...")
    try:
        CameraFacePipeline._ensure_arcface_loaded()
        embedding = pipeline._embed(crop)
    except Exception as e:
        print(f"FAIL: ArcFace embedding step raised — {type(e).__name__}: {e}")
        sys.exit(1)

    if embedding is None:
        print("FAIL: ArcFace could not find/align a face in the YOLO crop (crop may be too tight/blurry).")
        sys.exit(1)

    print(f"OK: embedding produced — shape={embedding.shape}, dtype={embedding.dtype}, "
          f"L2 norm={float((embedding ** 2).sum() ** 0.5):.4f} (should be ~1.0, it's normalized)")

    print("\nALL CHECKS PASSED: detector loads, detects a face, and the crop produces a valid "
          "ArcFace embedding through the existing, unmodified recognition path.")


if __name__ == "__main__":
    main()
