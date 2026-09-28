"""
DETECTOR
========

    frame
      -> general object detection (yolov8s.pt, 80 COCO classes)
      -> object tracker (gives each object a stable track_id)
      -> split into people and other objects
      -> clothing / object colour
      -> PPE model (best.pt, 14 classes) - ONE inference
      -> turn PPE classes into helmet / mask / specs statuses
      -> attach each PPE item to the correct person
      -> hold the status for a few seconds so it stops flickering
      -> analytics (counted once per person)
      -> result


THREE CHANGES FOR RELIABLE LIVE WEBCAM DETECTION
------------------------------------------------

1. THE HELMET REGION NOW STARTS ABOVE THE PERSON BOX

   A hard hat sticks up above the head. YOLO's person box often starts
   at the hairline or the shoulders, so the helmet box sits partly
   OUTSIDE it and scored 0% overlap - the detection was found and then
   thrown away. The helmet region now begins 10% of the person's height
   above the top of their box.

2. THE MATCH THRESHOLD IS 0.35, NOT 0.45

   On a close-range webcam the boxes are large and jittery. 45% overlap
   was too strict and dropped correct matches.

3. PPE STATUS IS REMEMBERED FOR A FEW SECONDS

   This is the big one for live video. Your own measurements:

       NO-Hardhat  0.18  0.27  0.34  0.41

   The model does not detect on every single frame - it comes and goes.
   Without memory the label flashes on and off and the counts jump
   around, which is what "not working" looks like on screen.

   So when a person has a confident PPE result, it is remembered against
   their track_id for PPE_MEMORY_SECONDS. If the next frame misses, the
   remembered value is used instead of dropping to "unknown".

   This is NOT inventing detections. Nothing is remembered unless the
   model actually reported it for that person first, and it expires.


WHAT IS DELIBERATELY NOT DONE
-----------------------------
No guessing. If the model has never said anything about a person's mask,
that person reads "unknown", not "no mask". Your best.pt has real
NO-Mask / NO-Hardhat / NO-Goggles classes, so guessing is unnecessary.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any os.getenv() call below (and before importing
# ppe_detector, which reads its own env vars at import time). Without
# this, whichever of detector.py / ppe_detector.py / config.py happens
# to be imported first "wins" and every PPE_*, GENERAL_*, HELMET_* env
# override in .env is silently ignored - the app just uses hardcoded
# defaults with no error. This bit main.py's real import order:
# `from .detector import ...` runs before `from . import ... config`,
# so .env had never been loaded yet when this file's env reads ran.
load_dotenv()

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

from ultralytics import YOLO

from .ppe_detector import (
    run_ppe_detection,
    run_face_ppe_detection,
    get_ppe_model_status,
    get_face_ppe_status,
    FACE_PPE_MODEL_AVAILABLE,
    PPE_MODEL_PATH,
    PPE_MODEL_AVAILABLE,
    PPE_CAPABILITIES,
)
from .fixtures_detector import (
    detect_fixtures,
    get_fixtures_status,
    FIXTURES_MODEL_AVAILABLE,
    FIXTURES_CLASS_NAMES,
)
from .color_detector import (
    get_person_color,
    get_object_color,
    get_person_color_cached,
    get_object_color_cached,
    get_dominant_color,
    normalize_color_name,
    reset_color_cache,
    COLOR_NAMES,
)


# =====================================================================
# 1. SETTINGS
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_flag(name, default="0"):
    return os.getenv(name, default) == "1"


# --- general object model (COCO) --------------------------------------
# Searched in order, first match wins. Newer or larger weights placed at
# the top are picked up without renaming anything.
#
# WHICH ONE TO USE
#
#   model      params    COCO mAP    CPU
#   yolov8n     3.2 M      37.3       95 ms
#   yolov8s    11.2 M      44.9      280 ms     <- the current one
#   yolov8m    25.9 M      50.2      700 ms
#   yolov8l    43.7 M      52.9     1300 ms
#
# Going up a SIZE buys about 5 mAP. Going up a VERSION at the same size
# usually buys 2 to 4. So yolov8s -> yolov8m helps more than
# yolov8s -> yolo26s, though nothing stops you trying both.
#
# Point at any file without renaming:
#     $env:GENERAL_MODEL="C:\\path\\to\\yolo26s.pt"
#
# Ultralytics downloads a known name automatically the first time:
#     $env:GENERAL_MODEL="yolov8m.pt"
GENERAL_MODEL_CANDIDATES = [
    BASE_DIR / "models" / "yolo26s.pt",
    BASE_DIR / "models" / "yolo11s.pt",
    BASE_DIR / "models" / "yolov8m.pt",
    BASE_DIR / "models" / "yolov8s.pt",          # yours is here
    BASE_DIR.parent / "yolov8s.pt",
    BASE_DIR / "yolov8s.pt",
    BASE_DIR.parent / "models" / "yolov8s.pt",
]

GENERAL_CONFIDENCE = _env_float("GENERAL_CONFIDENCE", 0.25)

# =====================================================================
# HOW MANY TIMES AN OBJECT MUST BE SEEN BEFORE IT IS REPORTED
# =====================================================================
#
# Measured on two consecutive frames of the same, motionless room:
#
#     frame A    Chair 11   Tv 4   Laptop 3   Bowl 1
#     frame B    Chair 10   Tv 4   Laptop 3   Bottle 1
#
# Nothing moved. A chair vanished, and one object went from a bowl to a
# bottle. That is YOLO making a fresh independent guess every frame, and
# every wobble going straight to the dashboard.
#
# The tracker already counts how many frames it has seen each object -
# its "hits". Nothing was using that number.
#
# Now an object must be seen this many times before it is reported. A
# one-frame hallucination never reaches the screen; a real chair is well
# past three within a few seconds.
#
# PEOPLE ARE EXEMPT. They walk in and out, and waiting three frames at
# 1 FPS would mean noticing someone three seconds late.
# Default 2, not 3.
#
# Three turned out to be too strict. Small office things - a bottle, a
# keyboard, a mouse - are not found in every single frame, so they never
# reached three hits and disappeared from the dashboard entirely. The
# count went from seven kinds of object down to three.
#
# Two still blocks a one-frame hallucination, which was the point, while
# letting a real object through on its second sighting.
#
#     0 or 1   everything, instantly, wobble included
#     2        one-frame ghosts blocked            <- default
#     3+       very steady, but small things vanish
OBJECT_MIN_HITS = _env_int("OBJECT_MIN_HITS", 2)

# Small objects need a lower bar than large ones. A chair fills hundreds
# of pixels; a mouse at four metres is barely twenty across, and YOLO is
# never as sure about it.
#
# So anything in this list is kept at a lower confidence than the rest.
#
# "clock" was here and is not anymore - measured on real office footage
# it consistently misfired at 0.20-0.44 on a wall switch-plate and a
# ceiling fan (both round/white like a clock face), which a low floor
# let straight through. Moved to NOISY_CLASSES below instead.
SMALL_OBJECT_CLASSES = {
    "mouse", "keyboard", "cell phone", "remote", "book",
    "bottle", "cup", "bowl", "scissors", "tie", "vase",
    # Added after measuring real footage: on this fisheye wide shot,
    # genuine laptops/backpacks were being found at 0.22-0.27 confidence
    # (angled screens, partly behind monitors/desk dividers) but the
    # general 0.45 floor - raised to kill unrelated "clock"/"tv" noise -
    # was silently cutting all of them. 4 real backpacks/laptops were
    # invisible on the dashboard because of this.
    "laptop", "backpack",
}

SMALL_OBJECT_CONFIDENCE = _env_float("SMALL_OBJECT_CONFIDENCE", 0.18)

# The opposite problem: classes that regularly misfire on ordinary
# office furniture and need a HIGHER bar than normal, not a lower one.
# Measured on real footage: "dining table" on a plain white desk
# partition. "clock" used to be here too, but it misfired on the same
# ceiling fan in every single test frame (0.20 up to 0.65) and this
# room has no real clock for it to ever be right about - moved to
# BLOCKED_CLASSES instead of trying to find a threshold that saves it.
NOISY_CLASSES = {
    "dining table": _env_float("NOISY_CONF_DINING_TABLE", 0.65),
}

# Classes that will never legitimately appear in an office and are only
# ever a wrong guess here - filtered out regardless of confidence.
BLOCKED_CLASSES = {
    "airplane", "bus", "train", "boat", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "skis",
    "snowboard", "surfboard", "kite", "baseball bat", "baseball glove",
    "frisbee", "sports ball", "tennis racket", "hot dog", "donut",
    "toilet", "oven", "toaster", "hair drier", "parking meter",
    "stop sign", "fire hydrant",
    # Never once correct on this camera - always the ceiling fan.
    "clock",
}

# IMC stall demo scope: show ONLY these object classes on the dashboard
# for the event, instead of the full COCO set. There is no separate
# "laptop bag" class in COCO - a laptop bag is what COCO calls
# "backpack" or "handbag" depending on its shape, so those two classes
# already cover it without any new training. "person" is always kept
# internally even when this is on, because PPE cropping and the object
# tracker both depend on person detections - this only limits which
# OBJECT classes get returned to the caller/dashboard.
IMC_DEMO_MODE = _env_bool("IMC_DEMO_MODE", False)
IMC_ALLOWED_CLASSES = {"backpack", "handbag", "bottle"}

# Comes straight from the general COCO model, no specialist/fallback
# ensemble needed - verified reliable on this exact office's live
# footage (0.86 confidence), so it's added here rather than tuned/
# retrained like the 3 IMC classes above. "chair" was tried too but
# dropped - correct far more often than not, but every chair in frame
# (8-11 per camera here) lights up at once, which reads as visual
# clutter/noise even when individually accurate.
IMC_ALLOWED_CLASSES |= {"laptop"}

# A held phone is only ~15-20px wide on this camera's full frame (~1.5%
# of frame width) - whole-frame detection plateaued at 84-86% mAP50 no
# matter how much data/epochs/resolution/model size was thrown at it.
# Same fix as PPE_MODE=crop: crop tightly around each detected PERSON,
# upscale that crop, and run a dedicated single-class model on it, where
# the same phone now fills a much bigger fraction of the image. That hit
# 95.4% mAP50 on real held-out camera footage - the whole-frame model
# never got close to that no matter what was tried.
PHONE_CROP_MODEL_CANDIDATES = [
    BASE_DIR / "models" / "phone_crop_best.pt",
    BASE_DIR.parent / "models" / "phone_crop_best.pt",
]
PHONE_CROP_SIZE = _env_int("PHONE_CROP_SIZE", 512)
PHONE_CROP_CONFIDENCE = _env_float("PHONE_CROP_CONFIDENCE", 0.25)
PHONE_CROP_MARGIN = _env_float("PHONE_CROP_MARGIN", 1.6)

# Dedicated backpack/handbag/bottle specialist, fine-tuned on real
# CAM2/CAM3 footage: 99.5% / 99.2% / 99.4% mAP50 on held-out frames from
# these same cameras, comfortably above the 90% target and well above
# the base COCO model's recall for backpack specifically on this
# camera's angle/lighting (the original complaint that started this
# whole retrain effort - it was missing real, visible backpacks
# entirely). Trained as its own 4-class-only model (nc=4, no "person"),
# so it CANNOT replace the main GENERAL_MODEL outright - that would
# lose person detection, which PPE, the tracker and the phone-crop model
# above all depend on. Instead it runs alongside the base model, same
# specialist-overrides-generalist pattern as the fixtures model.
IMC_SPECIALIST_MODEL_CANDIDATES = [
    BASE_DIR / "models" / "imc_general_best.pt",
    BASE_DIR.parent / "models" / "imc_general_best.pt",
]
IMC_SPECIALIST_CLASSES = ("backpack", "handbag", "bottle")
IMC_SPECIALIST_CONFIDENCE = _env_float("IMC_SPECIALIST_CONFIDENCE", 0.30)
IMC_SPECIALIST_IMGSZ = int(_env_float("IMC_SPECIALIST_IMGSZ", 640))

# Crowded cameras (7-9 people) were running a separate phone-crop
# inference PER PERSON every cycle with no cap, unlike PPE which is
# already capped at PPE_MAX_CROPS - this alone pushed cycles to
# 16-43s against a 6.67s target on this CPU. Same fix: cap how many
# people get a phone-crop pass per cycle.
PHONE_CROP_MAX_PEOPLE = _env_int("PHONE_CROP_MAX_PEOPLE", 4)

# Plausible box area as a fraction of the full frame, per class - a
# sanity filter independent of confidence (a box outside this range is
# wrong regardless of how sure the model was).
IMC_AREA_BOUNDS = {
    "backpack": (0.003, 0.35),
    "handbag": (0.0015, 0.25),
    "bottle": (0.0003, 0.12),
    "cell phone": (0.00005, 0.35),  # already cropped tight around a person
}

# Same idea, but for boxes already in FULL-FRAME coordinates (WBF-fused
# and raw ensemble-fallback boxes are stored full-frame, not crop-relative,
# so IMC_AREA_BOUNDS's cell-phone 0.35 ceiling - sized for a tight person
# crop - would let a phone box covering a third of the whole camera frame
# through unfiltered; this is the "phone box too big" bug reported live).
IMC_FULLFRAME_AREA_BOUNDS = {
    "backpack": (0.003, 0.30),
    "handbag": (0.0015, 0.20),
    "bottle": (0.0003, 0.10),
    "cell phone": (0.00005, 0.03),
}

imc_specialist_model = None
IMC_SPECIALIST_MODEL_AVAILABLE = False
_imc_specialist_load_attempted = False


def _load_imc_specialist_model():
    global imc_specialist_model, IMC_SPECIALIST_MODEL_AVAILABLE, _imc_specialist_load_attempted

    if _imc_specialist_load_attempted:
        return
    _imc_specialist_load_attempted = True

    found = _find_first_existing(IMC_SPECIALIST_MODEL_CANDIDATES)
    if found is None:
        print("[IMC-SPECIALIST] imc_general_best.pt not found - backpack/handbag/bottle stay on the general model.")
        return

    try:
        imc_specialist_model = YOLO(str(found))
        IMC_SPECIALIST_MODEL_AVAILABLE = True
        print(f"[IMC-SPECIALIST] Loaded {found}")
    except Exception as exc:
        print(f"[IMC-SPECIALIST] Failed to load: {type(exc).__name__}: {exc}")


def detect_imc_specialist_objects(frame):
    """
    Run the dedicated backpack/handbag/bottle model on the whole frame.
    Returns detections in the same dict shape as detect_objects().
    """
    _load_imc_specialist_model()
    if not IMC_SPECIALIST_MODEL_AVAILABLE or imc_specialist_model is None or frame is None:
        return []

    try:
        results = imc_specialist_model(
            frame, conf=IMC_SPECIALIST_CONFIDENCE, imgsz=IMC_SPECIALIST_IMGSZ,
            device=DEVICE, verbose=False,
        )
    except Exception as exc:
        print(f"[IMC-SPECIALIST] ERROR during inference: {type(exc).__name__}: {exc}")
        return []

    names = getattr(imc_specialist_model, "names", {}) or {}
    detections = []
    frame_h, frame_w = frame.shape[:2]
    frame_area = max(1, frame_w * frame_h)

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = str(names.get(class_id, "")).lower()
            if class_name not in IMC_SPECIALIST_CLASSES:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            # Pixel-size sanity filter, independent of confidence: a
            # box whose area is wildly implausible for its class (a
            # "bottle" covering half the frame, a "backpack" a few
            # pixels wide) is a bad detection even if the model was
            # confident about it.
            box_area_frac = ((x2 - x1) * (y2 - y1)) / frame_area
            min_frac, max_frac = IMC_AREA_BOUNDS.get(class_name, (0.0, 1.0))
            if not (min_frac <= box_area_frac <= max_frac):
                continue

            detections.append({
                "name": class_name,
                "bbox": (x1, y1, x2, y2),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": confidence,
                "track_id": None,
                "status": None,
                "color": None,
            })

    return detections


YOLOS_FALLBACK_ENABLED = _env_bool("YOLOS_FALLBACK_ENABLED", True)
YOLOS_FALLBACK_CONFIDENCE = _env_float("YOLOS_FALLBACK_CONFIDENCE", 0.25)
# Running RT-DETR every detection cycle on 3 concurrent camera pipelines
# pushed cycle latency to 95-206s (measured live) - way past our real-time
# budget. Throttling it to at most once per this many seconds PER CAMERA
# (keyed by that camera's own ObjectTracker instance) keeps it as a
# periodic accuracy correction instead of a per-cycle cost every camera
# pays. A single process-wide timestamp was tried first and caused a
# reported bug: with 3 cameras sharing one slot, only one camera at a time
# actually got the ensemble boost, so the SAME real object showed up on
# one camera's feed but not another's, purely because of which camera
# happened to win the shared throttle that cycle - not a real detection
# difference.
YOLOS_FALLBACK_MIN_INTERVAL_S = _env_float("YOLOS_FALLBACK_MIN_INTERVAL_S", 30.0)
_yolos_fallback_last_run_ts_by_scope = {}
yolos_fallback_model = None
yolos_fallback_processor = None
YOLOS_FALLBACK_AVAILABLE = False
_yolos_fallback_load_attempted = False


def _load_yolos_fallback_model():
    """
    Open-vocabulary YOLO-World-S, used to fill in whatever our own
    fine-tuned specialist misses on a given frame - it is not fine-tuned
    for our cameras, but it never learned the specialist's specific failure
    modes either, so it is a genuinely independent second opinion. Measured
    directly on our own live camera frames against RT-DETR-L (the previous
    fallback): YOLO-World found ALL 4 classes in a single whole-frame pass
    (backpack 0.70, handbag 0.58, bottle 0.36, cell phone 0.29) where
    RT-DETR-L found zero cell phone at whole-frame scale (phones are only
    a few pixels wide in a full 1920x1080 overhead shot) - and it did this
    in 5.3s vs RT-DETR-L's 10.8s on the same frame.
    """
    global yolos_fallback_model, YOLOS_FALLBACK_AVAILABLE, _yolos_fallback_load_attempted

    if _yolos_fallback_load_attempted:
        return
    _yolos_fallback_load_attempted = True

    if not YOLOS_FALLBACK_ENABLED:
        return

    try:
        from ultralytics import YOLO
        yolos_fallback_model = YOLO("yolov8s-world.pt")
        yolos_fallback_model.set_classes(list(IMC_SPECIALIST_CLASSES))
        YOLOS_FALLBACK_AVAILABLE = True
        print("[YOLOS-FALLBACK] Loaded YOLO-World-S (zero-shot ensemble fallback)")
    except Exception as exc:
        print(f"[YOLOS-FALLBACK] Failed to load: {type(exc).__name__}: {exc}")


PHONE_FALLBACK_MIN_INTERVAL_S = _env_float("PHONE_FALLBACK_MIN_INTERVAL_S", 30.0)
_phone_fallback_last_run_ts_by_scope = {}

rtdetr_phone_model = None
RTDETR_PHONE_AVAILABLE = False
_rtdetr_phone_load_attempted = False


def _load_rtdetr_phone_model():
    """
    RT-DETR-L specifically for the phone-crop fallback. Measured directly:
    on the SAME person crops, RT-DETR found cell phone at up to 0.89
    confidence while YOLO-World-S found ZERO cell phone hits at any
    threshold down to 0.02 - YOLO-World is excellent for backpack/handbag/
    bottle but its zero-shot text-embedding for "cell phone" specifically
    doesn't fire reliably. Using the right tool per class instead of
    forcing one model to cover everything.
    """
    global rtdetr_phone_model, RTDETR_PHONE_AVAILABLE, _rtdetr_phone_load_attempted
    if _rtdetr_phone_load_attempted:
        return
    _rtdetr_phone_load_attempted = True
    if not YOLOS_FALLBACK_ENABLED:
        return
    try:
        from ultralytics import RTDETR
        rtdetr_phone_model = RTDETR("rtdetr-l.pt")
        RTDETR_PHONE_AVAILABLE = True
        print("[PHONE-FALLBACK] Loaded RT-DETR-L (dedicated phone-crop model)")
    except Exception as exc:
        print(f"[PHONE-FALLBACK] Failed to load RT-DETR: {type(exc).__name__}: {exc}")


def detect_phone_fallback_in_crops(frame, people, scope=None):
    """
    RT-DETR-L run on upscaled person crops, same reasoning as the golden-set
    finding: on a full 1920x1080 overhead frame a phone is only a few pixels
    wide and general/zero-shot detectors find ~zero of them at whole-frame
    scale. Cropping and upscaling around each person the way our own
    phone_crop_model already does gives it enough pixels to actually see a
    phone. `scope` (this camera's own tracker instance) keeps the throttle
    below independent per camera - see the note on YOLOS_FALLBACK_MIN_INTERVAL_S
    above for why a shared clock caused inconsistent detections between
    cameras.
    """
    if "cell phone" not in IMC_ALLOWED_CLASSES:
        return []
    if not YOLOS_FALLBACK_ENABLED or frame is None or not people:
        return []
    _load_rtdetr_phone_model()
    if not RTDETR_PHONE_AVAILABLE:
        return []

    scope_key = id(scope) if scope is not None else "_global"
    now = time.time()
    last_run = _phone_fallback_last_run_ts_by_scope.get(scope_key, 0.0)
    if now - last_run < PHONE_FALLBACK_MIN_INTERVAL_S:
        return []
    _phone_fallback_last_run_ts_by_scope[scope_key] = now

    frame_h, frame_w = frame.shape[:2]
    detections = []
    for person in people[:PHONE_CROP_MAX_PEOPLE]:
        try:
            x1, y1, x2, y2 = person["bbox"]
        except (KeyError, ValueError, TypeError):
            continue
        pw, ph = x2 - x1, y2 - y1
        if pw <= 0 or ph <= 0:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(pw, ph) * PHONE_CROP_MARGIN / 2
        cx1 = max(0, int(cx - half))
        cy1 = max(0, int(cy - half))
        cx2 = min(frame_w, int(cx + half))
        cy2 = min(frame_h, int(cy + half))
        if cx2 - cx1 < 20 or cy2 - cy1 < 20:
            continue

        crop = frame[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop.shape[:2]
        upscaled = cv2.resize(crop, (PHONE_CROP_SIZE, PHONE_CROP_SIZE), interpolation=cv2.INTER_CUBIC)

        try:
            results = rtdetr_phone_model(upscaled, conf=YOLOS_FALLBACK_CONFIDENCE, verbose=False)
        except Exception as exc:
            print(f"[PHONE-FALLBACK] ERROR during inference: {type(exc).__name__}: {exc}")
            continue

        names = getattr(rtdetr_phone_model, "names", {}) or {}
        sx = crop_w / PHONE_CROP_SIZE
        sy = crop_h / PHONE_CROP_SIZE
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = str(names.get(class_id, "")).lower()
                if class_name != "cell phone":
                    continue
                confidence = float(box.conf[0])
                bx1, by1, bx2, by2 = [float(v) for v in box.xyxy[0]]
                fx1 = cx1 + bx1 * sx
                fy1 = cy1 + by1 * sy
                fx2 = cx1 + bx2 * sx
                fy2 = cy1 + by2 * sy
                detections.append({
                    "name": "cell phone",
                    "bbox": (fx1, fy1, fx2, fy2),
                    "x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2,
                    "confidence": confidence,
                    "track_id": None,
                    "status": None,
                    "color": None,
                    "source": "phone_fallback",
                })

    # People sitting close together have overlapping crops (crops extend
    # past each person's own bbox by PHONE_CROP_MARGIN), so the SAME
    # physical phone on the desk between them gets picked up once per
    # crop it falls inside - this is the "lots of false phone detections"
    # bug reported live: not wrong boxes, duplicate boxes on one real
    # phone. Dedup by keeping the highest-confidence box in each
    # overlapping cluster.
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    deduped = []
    for det in detections:
        if any(calculate_iou(det["bbox"], kept["bbox"]) > 0.3 for kept in deduped):
            continue
        deduped.append(det)
    return deduped


def detect_yolos_fallback_objects(frame, only_classes=None, scope=None):
    """
    Run RT-DETR-L on the frame and return detections for our 4 target
    classes only. `only_classes` restricts to a subset (the classes the
    specialist missed entirely this cycle) to avoid paying the inference
    cost when it isn't needed. `scope` is this camera's own tracker
    instance (or any stable per-camera object) so the throttle below is
    tracked independently per camera instead of one shared clock.
    """
    _load_yolos_fallback_model()
    if not YOLOS_FALLBACK_AVAILABLE or frame is None:
        return []
    if only_classes is not None and not only_classes:
        return []

    scope_key = id(scope) if scope is not None else "_global"
    now = time.time()
    last_run = _yolos_fallback_last_run_ts_by_scope.get(scope_key, 0.0)
    if now - last_run < YOLOS_FALLBACK_MIN_INTERVAL_S:
        return []
    _yolos_fallback_last_run_ts_by_scope[scope_key] = now

    try:
        # imgsz=1024 measured both faster AND higher-confidence than 640 on
        # our own live frames for the small cell-phone class (0.73 vs 0.57
        # top confidence, 11.6s vs 21.8s) - counterintuitive but verified
        # directly, not a default guess.
        results = yolos_fallback_model(frame, conf=YOLOS_FALLBACK_CONFIDENCE, imgsz=1024, verbose=False)
    except Exception as exc:
        print(f"[YOLOS-FALLBACK] ERROR during inference: {type(exc).__name__}: {exc}")
        return []

    names = getattr(yolos_fallback_model, "names", {}) or {}
    frame_h, frame_w = frame.shape[:2]
    frame_area = max(1, frame_w * frame_h)
    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = str(names.get(class_id, "")).lower()
            if class_name not in IMC_SPECIALIST_CLASSES:
                continue
            if only_classes is not None and class_name not in only_classes:
                continue
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            box_area_frac = ((x2 - x1) * (y2 - y1)) / frame_area
            min_frac, max_frac = IMC_AREA_BOUNDS.get(class_name, (0.0, 1.0))
            if not (min_frac <= box_area_frac <= max_frac):
                continue
            detections.append({
                "name": class_name,
                "bbox": (x1, y1, x2, y2),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": confidence,
                "track_id": None,
                "status": None,
                "color": None,
                "source": "yolos_fallback",
            })
    return detections


def _wbf_fuse_same_class(specialist_items, fallback_items, frame_w, frame_h):
    """
    Weighted Boxes Fusion between our specialist's detections and the
    RT-DETR fallback's detections, run separately per class. Returns
    (fused_items, consumed_fallback_indices) - consumed_fallback_indices
    marks which fallback_items were folded into a fused box so the caller
    doesn't also add them again as a separate "new" detection.
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except Exception:
        return list(specialist_items), set()

    fused_items = []
    consumed = set()

    by_class = {}
    for item in specialist_items:
        by_class.setdefault(item["name"], {"specialist": [], "fallback": []})["specialist"].append(item)
    for idx, item in enumerate(fallback_items):
        by_class.setdefault(item["name"], {"specialist": [], "fallback": []})["fallback"].append((idx, item))

    for class_name, group in by_class.items():
        specialist_group = group["specialist"]
        fallback_group = group["fallback"]

        if not fallback_group:
            fused_items.extend(specialist_group)
            continue

        boxes_list, scores_list, labels_list = [], [], []
        for item in specialist_group:
            x1, y1, x2, y2 = item["bbox"]
            boxes_list.append([x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h])
            scores_list.append(item["confidence"])
            labels_list.append(0)
        for _, item in fallback_group:
            x1, y1, x2, y2 = item["bbox"]
            boxes_list.append([x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h])
            scores_list.append(item["confidence"])
            labels_list.append(0)
        # No specialist boxes for this class - still run WBF on the
        # fallback-only boxes so RT-DETR's own overlapping duplicate boxes
        # for the same physical object (it doesn't NMS as aggressively as
        # our specialist) get merged into one instead of all being added.

        try:
            fused_boxes, fused_scores, _ = weighted_boxes_fusion(
                [boxes_list], [scores_list], [labels_list],
                weights=None, iou_thr=0.5, skip_box_thr=0.0001,
            )
        except Exception:
            fused_items.extend(specialist_group)
            continue

        for box, score in zip(fused_boxes, fused_scores):
            x1, y1, x2, y2 = box
            fused_items.append({
                "name": class_name,
                "bbox": (x1 * frame_w, y1 * frame_h, x2 * frame_w, y2 * frame_h),
                "x1": x1 * frame_w, "y1": y1 * frame_h, "x2": x2 * frame_w, "y2": y2 * frame_h,
                "confidence": float(score),
                "track_id": None,
                "status": None,
                "color": None,
                "source": "wbf_fused",
            })
        for idx, _ in fallback_group:
            consumed.add(idx)

    return fused_items, consumed


clip_model = None
clip_processor = None
CLIP_AVAILABLE = False
_clip_load_attempted = False
CLIP_LABELS = {
    "backpack": "a photo of a backpack",
    "handbag": "a photo of a handbag",
    "bottle": "a photo of a water bottle",
    "cell phone": "a photo of a cell phone",
}


def _load_clip_model():
    global clip_model, clip_processor, CLIP_AVAILABLE, _clip_load_attempted
    if _clip_load_attempted:
        return
    _clip_load_attempted = True
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = clip_model.to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        CLIP_AVAILABLE = True
        print(f"[CLIP] Loaded openai/clip-vit-base-patch32 on {DEVICE} (backpack/handbag tiebreaker)")
    except Exception as exc:
        print(f"[CLIP] Failed to load: {type(exc).__name__}: {exc}")


def _clip_disambiguate(frame, box, candidate_names):
    """
    Zero-shot re-classification of a crop between two confused IMC classes
    (verified case: RT-DETR gave the SAME box backpack=0.33 vs handbag=0.28
    - CLIP resolved it clearly as handbag 0.58 vs backpack 0.39). Returns
    the winning class name, or None if CLIP isn't available/fails.
    """
    _load_clip_model()
    if not CLIP_AVAILABLE or frame is None:
        return None
    try:
        import torch
        from PIL import Image
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        labels = [CLIP_LABELS[n] for n in candidate_names]
        inputs = clip_processor(text=labels, images=pil_img, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        best_idx = int(probs.argmax())
        return candidate_names[best_idx]
    except Exception as exc:
        print(f"[CLIP] ERROR during disambiguation: {type(exc).__name__}: {exc}")
        return None


phone_crop_model = None
PHONE_CROP_MODEL_AVAILABLE = False
_phone_crop_load_attempted = False


def _load_phone_crop_model():
    global phone_crop_model, PHONE_CROP_MODEL_AVAILABLE, _phone_crop_load_attempted

    if _phone_crop_load_attempted:
        return
    _phone_crop_load_attempted = True

    found = _find_first_existing(PHONE_CROP_MODEL_CANDIDATES)
    if found is None:
        print("[PHONE-CROP] phone_crop_best.pt not found - cell phone stays on the general model.")
        return

    try:
        phone_crop_model = YOLO(str(found))
        PHONE_CROP_MODEL_AVAILABLE = True
        print(f"[PHONE-CROP] Loaded {found}")
    except Exception as exc:
        print(f"[PHONE-CROP] Failed to load: {type(exc).__name__}: {exc}")


def detect_phones_in_person_crops(frame, people):
    """
    For each detected person, crop + upscale around them and run the
    dedicated phone-crop model. Returns full-frame "cell phone"
    detections (same dict shape as detect_objects()), mapped back from
    crop coordinates to the original frame.
    """
    if "cell phone" not in IMC_ALLOWED_CLASSES:
        return []
    _load_phone_crop_model()
    if not PHONE_CROP_MODEL_AVAILABLE or phone_crop_model is None or frame is None or not people:
        return []

    frame_h, frame_w = frame.shape[:2]
    results = []

    for person in people[:PHONE_CROP_MAX_PEOPLE]:
        try:
            x1, y1, x2, y2 = person["bbox"]
        except (KeyError, ValueError, TypeError):
            continue

        pw, ph = x2 - x1, y2 - y1
        if pw <= 0 or ph <= 0:
            continue

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(pw, ph) * PHONE_CROP_MARGIN / 2
        cx1 = max(0, int(cx - half))
        cy1 = max(0, int(cy - half))
        cx2 = min(frame_w, int(cx + half))
        cy2 = min(frame_h, int(cy + half))

        if cx2 - cx1 < 20 or cy2 - cy1 < 20:
            continue

        crop = frame[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop.shape[:2]
        upscaled = cv2.resize(crop, (PHONE_CROP_SIZE, PHONE_CROP_SIZE), interpolation=cv2.INTER_CUBIC)

        try:
            crop_results = phone_crop_model(
                upscaled, conf=PHONE_CROP_CONFIDENCE, device=DEVICE, verbose=False,
            )
        except Exception as exc:
            print(f"[PHONE-CROP] ERROR during inference: {type(exc).__name__}: {exc}")
            continue

        # scale factors from the CROP_SIZE x CROP_SIZE upscaled image
        # back to the original crop, then offset by the crop's own
        # position in the full frame.
        sx = crop_w / PHONE_CROP_SIZE
        sy = crop_h / PHONE_CROP_SIZE

        crop_area = max(1, PHONE_CROP_SIZE * PHONE_CROP_SIZE)
        min_frac, max_frac = IMC_AREA_BOUNDS.get("cell phone", (0.0, 1.0))

        for result in crop_results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf[0])
                bx1, by1, bx2, by2 = [float(v) for v in box.xyxy[0]]

                box_area_frac = ((bx2 - bx1) * (by2 - by1)) / crop_area
                if not (min_frac <= box_area_frac <= max_frac):
                    continue

                fx1 = cx1 + bx1 * sx
                fy1 = cy1 + by1 * sy
                fx2 = cx1 + bx2 * sx
                fy2 = cy1 + by2 * sy

                # The crop-relative check above only catches a box that's
                # a huge fraction of the tight person-crop - a box that's
                # a plausible fraction of a LARGE crop (a person filling
                # much of the frame) can still be an implausibly big phone
                # once mapped back to the full frame. Check that too.
                fullframe_frac = ((fx2 - fx1) * (fy2 - fy1)) / max(1, frame_w * frame_h)
                ff_min, ff_max = IMC_FULLFRAME_AREA_BOUNDS.get("cell phone", (0.0, 1.0))
                if not (ff_min <= fullframe_frac <= ff_max):
                    continue

                results.append({
                    "name": "cell phone",
                    "bbox": (fx1, fy1, fx2, fy2),
                    "x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2,
                    "confidence": confidence,
                    "track_id": None,
                    "status": None,
                    "color": None,
                })

    return results

# Missing a PERSON is the worst failure mode for a surveillance system -
# worse than the occasional false positive elsewhere. Measured on real
# footage: a genuinely present, visible person was found at 0.44
# confidence, just under the general 0.45 floor, and dropped entirely.
PERSON_CONFIDENCE = _env_float("PERSON_CONFIDENCE", 0.30)

# Names that skip the wait.
INSTANT_CLASSES = {"person"}
GENERAL_IOU = _env_float("GENERAL_IOU", 0.45)
GENERAL_IMAGE_SIZE = _env_int("GENERAL_IMGSZ", 640)

# --- dedicated helmet model (optional, NOT used in the live loop) ------
HELMET_MODEL_CANDIDATES = [
    BASE_DIR / "models" / "hard_hat_best.pt",
    BASE_DIR.parent / "models" / "hard_hat_best.pt",
]
# Measured on this camera: NO-Hardhat came back at 0.18 / 0.27 / 0.34 /
# 0.39. hard_hat_detector.py used 0.55, which rejected every one of them.
HELMET_CONFIDENCE = _env_float("HELMET_CONFIDENCE", 0.20)
HELMET_IMAGE_SIZE = _env_int("HELMET_IMGSZ", 640)

# --- PPE association ---------------------------------------------------
# How much of a PPE box must sit inside the person's relevant region.
# Was 0.45; lowered because close-range webcam boxes are large and jittery.
PPE_MATCH_THRESHOLD = _env_float("PPE_MATCH_THRESHOLD", 0.30)

# Second-chance threshold.
#
# Stage 1 checks the PPE box against the right part of the body: a helmet
# against the head, a mask against the face. That is correct for CCTV and
# it stops one person's helmet being credited to the person next to them.
#
# But best.pt does not always put its box where you would expect. Measured
# on a close-range webcam, a NO-Hardhat at 0.39 confidence came back as
# [129, 272, 367, 479] - over the torso, not the head. Against the head
# region that scores 0% and the detection was thrown away.
#
# So stage 2 gives anything unmatched a second chance against the person's
# WHOLE box. Still the best-matching person, so multi-person accuracy is
# kept, but a good detection is no longer lost because the model framed it
# differently from the training data.
#
# Set PPE_FALLBACK_MATCH=0 to switch stage 2 off and keep only strict
# region matching.
PPE_FALLBACK_MATCH_THRESHOLD = _env_float("PPE_FALLBACK_MATCH", 0.20)

# Should the dedicated 2-class hard_hat_best.pt run as well as best.pt?
#
# OFF by default. best.pt already has Hardhat AND NO-Hardhat, so running
# both would put two helmet boxes on the same head and count that head
# twice. Turn it on only if you specifically want the second opinion -
# merge_helmet_results() below then removes the duplicates.
USE_DEDICATED_HELMET_MODEL = _env_flag("USE_DEDICATED_HELMET_MODEL", "0")

# Two helmet boxes overlapping more than this are treated as the same
# helmet, so only the more confident one is kept.
HELMET_DUPLICATE_IOU = _env_float("HELMET_DUPLICATE_IOU", 0.50)

# Which slice of a person's height each PPE item should be found in.
# 0.0 = top of the person's box, 1.0 = their feet.
#
# The helmet region starts at -0.10, i.e. ABOVE the box, because a hard
# hat sticks up past the head and YOLO's person box often starts at the
# hairline. Without this the helmet scores 0% overlap and is discarded.
# Every PPE category the pipeline handles end to end: detection ->
# association -> memory -> analytics -> result. Gloves included.
#
# The PPE model's own "person" class is deliberately NOT here. yolov8s
# already detects people and the tracker gives them stable ids; taking
# people from best.pt as well would create a second, competing set of
# person boxes.
PPE_CATEGORIES = ("helmet", "mask", "specs", "vest", "gloves")

# Categories absence may be inferred for, when that is switched on.
PPE_INFER_CATEGORIES = ("helmet", "mask", "specs")

PPE_REGIONS = {
    "helmet": (-0.10, 0.38),
    "specs": (0.00, 0.34),
    "mask": (0.04, 0.44),
    "vest": (0.15, 0.75),
    "gloves": (0.30, 0.95),
}

# How long a confident PPE result is held for a person after the model
# stops reporting it. Set PPE_MEMORY_SECONDS=0 to switch this off.
# At 0.1 FPS - about ten seconds a frame on this machine - a three
# second memory expires before the next frame even arrives. Fifteen
# seconds keeps a person's status steady between updates.
PPE_MEMORY_SECONDS = _env_float("PPE_MEMORY_SECONDS", 15.0)

# Anything older than this is dropped entirely, so the memory cannot grow
# without limit as people come and go.
PPE_MEMORY_MAX_AGE = 60.0

# Your best.pt has explicit NO- classes, but measured on this webcam they
# barely fire, while positive classes (Goggles) work fine. So absence can
# be worked out instead: if a person is clearly visible and no helmet was
# found on them, they are not wearing one.
#
#   "off"     never guess
#   "auto"    guess only where the model has no NO- class of its own
#   "always"  guess whenever nothing was detected   <- default
#
# Positive detections ALWAYS win. Nothing is guessed for a person the
# model has already said something about.
PPE_ABSENCE_MODE = os.getenv("PPE_ABSENCE_MODE", "off").strip().lower()

# The old on/off switch is still honoured, so PPE_INFER_ABSENCE=0 works.
if os.getenv("PPE_INFER_ABSENCE") == "0":
    PPE_ABSENCE_MODE = "off"

PPE_INFER_ABSENCE = PPE_ABSENCE_MODE != "off"

# Conditions that must hold before we are willing to guess. These stop a
# tiny or half-visible person being reported as a safety violation.
PPE_INFER_MIN_PERSON_CONF = _env_float("PPE_INFER_MIN_PERSON_CONF", 0.40)
PPE_INFER_MIN_PERSON_HEIGHT = _env_int("PPE_INFER_MIN_PERSON_HEIGHT", 60)

# For helmet we also need the top of the head inside the picture. If the
# person's box is jammed against the top edge their head is cut off, and
# we cannot honestly say whether they are wearing a helmet.
PPE_INFER_TOP_MARGIN = _env_int("PPE_INFER_TOP_MARGIN", 4)

# --- device ------------------------------------------------------------
if torch is not None and torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# --- debug -------------------------------------------------------------
DETECT_DEBUG = _env_flag("DETECT_DEBUG", "1")
_DEBUG_INTERVAL_SECONDS = 1.0
_last_debug_time = 0.0


def _debug(lines):
    """Print a block of debug lines at most once per second."""
    global _last_debug_time
    if not DETECT_DEBUG:
        return
    now = time.time()
    if now - _last_debug_time < _DEBUG_INTERVAL_SECONDS:
        return
    _last_debug_time = now
    for line in lines:
        print(line)


# =====================================================================
# 2. MODEL LOADING
# =====================================================================

model = None
GENERAL_MODEL_PATH = GENERAL_MODEL_CANDIDATES[0]
GENERAL_MODEL_AVAILABLE = False
GENERAL_LOAD_ERROR = None

helmet_model = None
HELMET_LOCAL_PATH = HELMET_MODEL_CANDIDATES[0]
HELMET_MODEL_AVAILABLE = False
HELMET_LOAD_ERROR = None
_helmet_load_attempted = False


def _find_first_existing(candidates):
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _find_general_model():
    """
    GENERAL_MODEL wins if set - a full path, or a name ultralytics knows
    and will download ("yolov8m.pt", "yolo11s.pt"). Otherwise the first
    candidate that exists.
    """
    from_env = os.getenv("GENERAL_MODEL")

    if from_env:
        candidate = Path(from_env)
        if candidate.is_file():
            return candidate
        # Not a path on disk. If it looks like a known weights name, hand
        # it to ultralytics and let it fetch it.
        if from_env.endswith(".pt") and "/" not in from_env and "\\" not in from_env:
            print(f"[OBJECT] GENERAL_MODEL={from_env} is not on disk - "
                  f"ultralytics will try to download it.")
            return Path(from_env)
        print(f"[OBJECT] GENERAL_MODEL points at a missing file: {candidate}")

    return _find_first_existing(GENERAL_MODEL_CANDIDATES)


def load_general_model():
    """Load the COCO object model. Reports honestly if it fails."""
    global model, GENERAL_MODEL_PATH, GENERAL_MODEL_AVAILABLE, GENERAL_LOAD_ERROR

    model = None
    GENERAL_MODEL_AVAILABLE = False
    GENERAL_LOAD_ERROR = None

    found = _find_general_model()

    if found is None:
        GENERAL_LOAD_ERROR = "yolov8s.pt not found"
        print("=" * 60)
        print("[OBJECT] GENERAL MODEL NOT FOUND")
        print("=" * 60)
        for candidate in GENERAL_MODEL_CANDIDATES:
            print(f"[OBJECT]   looked in: {candidate}")
        print("[OBJECT] Object detection is DISABLED.")
        print("=" * 60)
        return

    GENERAL_MODEL_PATH = found

    try:
        loaded = YOLO(str(found))
        try:
            loaded.to(DEVICE)
        except Exception as exc:
            print(f"[OBJECT] Could not move model to {DEVICE}: {exc}")

        model = loaded
        GENERAL_MODEL_AVAILABLE = True

        print("=" * 60)
        print("[OBJECT] GENERAL MODEL LOADED")
        print("=" * 60)
        print(f"[OBJECT] File       : {found}")
        print(f"[OBJECT] Device     : {DEVICE}")
        print(f"[OBJECT] Confidence : {GENERAL_CONFIDENCE}")
        print(f"[OBJECT] Image size : {GENERAL_IMAGE_SIZE}")
        print(f"[OBJECT] Classes    : {len(getattr(loaded, 'names', {}) or {})}")
        print("=" * 60)

    except Exception as exc:
        GENERAL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        print("=" * 60)
        print("[OBJECT] GENERAL MODEL FAILED TO LOAD")
        print(f"[OBJECT] File  : {found}")
        print(f"[OBJECT] Reason: {GENERAL_LOAD_ERROR}")
        print("=" * 60)
        model = None
        GENERAL_MODEL_AVAILABLE = False


def load_helmet_model():
    """
    Load the 2-class hard hat model.

    NOT used by the live pipeline. best.pt already detects Hardhat AND
    NO-Hardhat, so running a second model on every frame would just waste
    CPU and create duplicate helmet results. Kept for direct calls only,
    and loaded lazily the first time it is actually needed.
    """
    global helmet_model, HELMET_LOCAL_PATH
    global HELMET_MODEL_AVAILABLE, HELMET_LOAD_ERROR, _helmet_load_attempted

    _helmet_load_attempted = True
    helmet_model = None
    HELMET_MODEL_AVAILABLE = False
    HELMET_LOAD_ERROR = None

    found = _find_first_existing(HELMET_MODEL_CANDIDATES)

    if found is None:
        HELMET_LOAD_ERROR = "hard_hat_best.pt not found"
        print(f"[HELMET] Model not found in "
              f"{[str(c) for c in HELMET_MODEL_CANDIDATES]}")
        return

    HELMET_LOCAL_PATH = found

    try:
        loaded = YOLO(str(found))
        try:
            loaded.to(DEVICE)
        except Exception:
            pass
        helmet_model = loaded
        HELMET_MODEL_AVAILABLE = True
        print(f"[HELMET] Model loaded: {found}")
        print(f"[HELMET] Classes: {getattr(loaded, 'names', {})}")
    except Exception as exc:
        HELMET_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        helmet_model = None
        HELMET_MODEL_AVAILABLE = False
        print(f"[HELMET] Model FAILED to load: {HELMET_LOAD_ERROR}")


load_general_model()

print(f"[DETECTOR] PPE model available : {PPE_MODEL_AVAILABLE}")
print(f"[DETECTOR] PPE model path      : {PPE_MODEL_PATH}")
print(f"[DETECTOR] PPE match threshold : {PPE_MATCH_THRESHOLD}")
print(f"[DETECTOR] PPE memory          : {PPE_MEMORY_SECONDS}s")
print(f"[DETECTOR] Absence inference   : "
      f"{PPE_ABSENCE_MODE}")


# =====================================================================
# 3. SMALL HELPERS
# =====================================================================

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_bbox(frame, x1, y1, x2, y2):
    """Keep a box inside the picture."""
    if frame is None:
        return 0, 0, 0, 0
    height, width = frame.shape[:2]
    x1 = max(0, min(safe_int(x1), width - 1))
    y1 = max(0, min(safe_int(y1), height - 1))
    x2 = max(0, min(safe_int(x2), width - 1))
    y2 = max(0, min(safe_int(y2), height - 1))
    return x1, y1, x2, y2


def calculate_iou(box_a, box_b):
    """Intersection over union of two boxes."""
    try:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - overlap

        return overlap / union if union > 0 else 0.0
    except Exception:
        return 0.0


def intersection_over_area(inner_box, outer_box):
    """
    How much of inner_box sits inside outer_box, as a fraction of
    inner_box's own area. The right measure for attaching a small PPE box
    to a big person box - plain IoU would always be tiny.
    """
    try:
        ax1, ay1, ax2, ay2 = inner_box
        bx1, by1, bx2, by2 = outer_box

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        inner_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)

        return overlap / inner_area if inner_area > 0 else 0.0
    except Exception:
        return 0.0


def _box_of(detection):
    """Read a bounding box out of a detection dict, whatever style it uses."""
    if not isinstance(detection, dict):
        return None
    bbox = detection.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [safe_int(v) for v in bbox]
    if all(key in detection for key in ("x1", "y1", "x2", "y2")):
        return [safe_int(detection["x1"]), safe_int(detection["y1"]),
                safe_int(detection["x2"]), safe_int(detection["y2"])]
    return None


# =====================================================================
# 4. GENERAL OBJECT DETECTION
# =====================================================================

def detect_objects(frame, conf=None, with_color=True):
    """
    Run the COCO model and return every object it finds.

    with_color=False skips colour so run_full_detection can compute it
    once after tracking instead of once per raw detection.
    """
    detections = []

    if frame is None or model is None or not GENERAL_MODEL_AVAILABLE:
        return detections

    threshold = GENERAL_CONFIDENCE if conf is None else float(conf)

    # Run the model at the LOWEST bar any class might need, then filter
    # per class below. Same forward pass, but a small object is no
    # longer thrown away inside the model before we get to see it.
    inference_floor = min(threshold, SMALL_OBJECT_CONFIDENCE, PERSON_CONFIDENCE)

    try:
        results = model(
            frame,
            conf=inference_floor,
            iou=GENERAL_IOU,
            imgsz=GENERAL_IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )
    except Exception as exc:
        print(f"[OBJECT] ERROR during inference: {type(exc).__name__}: {exc}")
        return detections

    names = getattr(model, "names", {}) or {}

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            try:
                confidence = float(box.conf[0])

                class_id = int(box.cls[0])
                class_name = str(names.get(class_id, f"class_{class_id}")).lower()

                if class_name in BLOCKED_CLASSES:
                    continue

                if IMC_DEMO_MODE and class_name not in IMC_ALLOWED_CLASSES and class_name != "person":
                    continue

                # Small things get a lower bar; noisy things get a
                # higher one. Both judged AFTER the class is known, so
                # the model still runs once at the lowest floor either
                # side needs.
                if class_name in NOISY_CLASSES:
                    needed = max(threshold, NOISY_CLASSES[class_name])
                elif class_name == "person":
                    needed = min(threshold, PERSON_CONFIDENCE)
                elif class_name in SMALL_OBJECT_CLASSES:
                    needed = min(threshold, SMALL_OBJECT_CONFIDENCE)
                else:
                    needed = threshold

                if confidence < needed:
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                x1, y1, x2, y2 = clamp_bbox(frame, x1, y1, x2, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                color = "unknown"
                if with_color:
                    if class_name == "person":
                        color = get_person_color(frame, x1, y1, x2, y2)
                    else:
                        color = get_object_color(frame, x1, y1, x2, y2)

                detections.append({
                    "name": class_name,
                    "class_id": class_id,
                    "color": color,
                    "confidence": round(confidence, 3),
                    "status": None,
                    "helmet": "unknown",
                    "helmet_confidence": 0.0,
                    "mask": "unknown",
                    "mask_confidence": 0.0,
                    "specs": "unknown",
                    "specs_confidence": 0.0,
                    "vest": "unknown",
                    "gloves": "unknown",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "bbox": [x1, y1, x2, y2],
                })

            except Exception as exc:
                print(f"[OBJECT] Skipped one bad box: {type(exc).__name__}: {exc}")

    return detections


# =====================================================================
# 5. DEDICATED HELMET MODEL (optional, not used in the live loop)
# =====================================================================

def classify_helmet_status(class_name):
    """Turn any helmet-ish class name into 'helmet' or 'no helmet'."""
    value = str(class_name or "").strip().lower()
    value = value.replace("-", "_").replace(" ", "_")

    is_negative = (
        value.startswith("no_")
        or value.startswith("non_")
        or value.startswith("not_")
        or value.startswith("without_")
    )
    return "no helmet" if is_negative else "helmet"


def _is_reasonable_helmet_box(frame, x1, y1, x2, y2):
    """Reject boxes obviously too big or the wrong shape for a head."""
    height, width = frame.shape[:2]
    box_width, box_height = x2 - x1, y2 - y1

    if box_width <= 0 or box_height <= 0:
        return False
    if box_height > height * 0.6 or box_width > width * 0.6:
        return False

    aspect = box_width / max(box_height, 1)
    return 0.3 <= aspect <= 5.0



def get_helmet_color(frame, x1, y1, x2, y2):
    """
    What colour is the hard hat?

    Taken from hard_hat_detector.py, which was never actually wired into
    the pipeline. Useful on a site where colour means role - white for
    supervisors, yellow for workers, and so on.

    Very dark and very washed-out pixels are dropped first, then the
    median hue decides.
    """
    try:
        crop = frame[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Keep only bright, colourful pixels for the HUE decision.
        usable = (hsv[:, :, 2] > 60) & (hsv[:, :, 1] > 35)

        if usable.any():
            hue = float(np.median(hsv[:, :, 0][usable]))
            sat = float(np.median(hsv[:, :, 1][usable]))
            val = float(np.median(hsv[:, :, 2][usable]))
        else:
            # Nothing survived - which is exactly what a WHITE or BLACK
            # helmet looks like. White has almost no saturation, black
            # has almost no value, so the filter above removes every
            # pixel of both. hard_hat_detector.py returned "Unknown"
            # here, so white and black helmets never got a colour.
            # Judge those two from the whole crop instead.
            hue = float(np.median(hsv[:, :, 0]))
            sat = float(np.median(hsv[:, :, 1]))
            val = float(np.median(hsv[:, :, 2]))

            if val < 70:
                return "black"
            if sat < 60:
                return "white"

        if sat < 60 and val > 150:
            return "white"
        if val < 70:
            return "black"
        if hue < 10 or hue >= 170:
            return "red"
        if 10 <= hue < 20:
            return "orange"
        if 20 <= hue < 40:
            return "yellow"
        if 40 <= hue < 85:
            return "green"
        if 85 <= hue < 135:
            return "blue"

        return "unknown"

    except Exception as exc:
        print(f"[HELMET] colour error: {type(exc).__name__}: {exc}")
        return "unknown"


def run_helmet_model(image, conf=None, imgsz=None):
    """
    Run hard_hat_best.pt on ONE image - a full frame or a person crop.

    Returns detections in that image's own coordinates, already
    normalised to "helmet" / "no helmet".
    """
    global _helmet_load_attempted

    detections = []

    if image is None or not USE_DEDICATED_HELMET_MODEL:
        return detections

    if not _helmet_load_attempted:
        load_helmet_model()

    if helmet_model is None or not HELMET_MODEL_AVAILABLE:
        return detections

    threshold = HELMET_CONFIDENCE if conf is None else float(conf)
    image_size = HELMET_IMAGE_SIZE if imgsz is None else int(imgsz)

    try:
        results = helmet_model(
            image,
            conf=threshold,
            iou=0.45,
            imgsz=image_size,
            device=DEVICE,
            verbose=False,
        )
    except Exception as exc:
        print(f"[HELMET] inference error: {type(exc).__name__}: {exc}")
        return detections

    names = getattr(helmet_model, "names", {}) or {}

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            try:
                confidence = float(box.conf[0])
                if confidence < threshold:
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                x1, y1, x2, y2 = clamp_bbox(image, x1, y1, x2, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                class_id = int(box.cls[0])
                status = classify_helmet_status(
                    str(names.get(class_id, "helmet"))
                )

                colour = (
                    get_helmet_color(image, x1, y1, x2, y2)
                    if status == "helmet" else "none"
                )

                detections.append({
                    "name": "helmet",
                    "category": "helmet",
                    "status": status,
                    "helmet": status,
                    "class_name": str(names.get(class_id, "helmet")),
                    "confidence": round(confidence, 3),
                    "helmet_confidence": round(confidence, 3),
                    "helmet_color": colour,
                    "color": colour,
                    "from_helmet_model": True,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "bbox": [x1, y1, x2, y2],
                })

            except Exception as exc:
                print(f"[HELMET] bad box: {type(exc).__name__}: {exc}")

    return detections


def detect_helmets(frame):
    """
    Run the dedicated 2-class hard hat model.

    The live pipeline does NOT call this - best.pt already covers helmets,
    and running both would produce duplicate helmet results. Kept because
    main.py imports it. Loads the model on first use.
    """
    global _helmet_load_attempted

    detections = []

    if frame is None:
        return detections

    # best.pt already reports Hardhat and NO-Hardhat. Running this model
    # too would duplicate every helmet. Off unless explicitly enabled.
    if not USE_DEDICATED_HELMET_MODEL:
        return detections

    if not _helmet_load_attempted:
        load_helmet_model()

    if helmet_model is None or not HELMET_MODEL_AVAILABLE:
        return detections

    try:
        results = helmet_model(
            frame,
            conf=HELMET_CONFIDENCE,
            imgsz=HELMET_IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )
    except Exception as exc:
        print(f"[HELMET] ERROR during inference: {type(exc).__name__}: {exc}")
        return detections

    names = getattr(helmet_model, "names", {}) or {}

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            try:
                confidence = float(box.conf[0])
                if confidence < HELMET_CONFIDENCE:
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                x1, y1, x2, y2 = clamp_bbox(frame, x1, y1, x2, y2)

                if not _is_reasonable_helmet_box(frame, x1, y1, x2, y2):
                    continue

                class_id = int(box.cls[0])
                class_name = str(names.get(class_id, "helmet"))
                status = classify_helmet_status(class_name)

                detections.append({
                    "name": "helmet",
                    "category": "helmet",
                    "status": status,
                    "helmet": status,
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "helmet_confidence": round(confidence, 3),
                    "color": "unknown",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "bbox": [x1, y1, x2, y2],
                })

            except Exception as exc:
                print(f"[HELMET] Skipped one bad box: {type(exc).__name__}: {exc}")

    return detections


# =====================================================================
# 6. TURNING RAW PPE DETECTIONS INTO CATEGORY LISTS
# =====================================================================

def split_ppe_by_category(raw_ppe):
    """
    Sort raw PPE detections into one list per category.

    ppe_detector already worked out 'category' and 'status' from the real
    model class names, so there is no guessing to do here.
    """
    buckets = {
        "helmet": [], "mask": [], "specs": [],
        "vest": [], "gloves": [], "person": [], "other": [],
    }

    for detection in raw_ppe or []:
        if not isinstance(detection, dict):
            continue

        category = detection.get("category", "other")

        # Ignore the PPE model's own "person" class. yolov8s already
        # detects people and the tracker gives them stable ids. Letting
        # best.pt add its own person boxes would create a second,
        # competing set of people with no track ids.
        if category == "person":
            continue

        if category not in buckets:
            category = "other"
        buckets[category].append(detection)

    return buckets


def detect_ppe(frame):
    """Backward-compatible wrapper: raw PPE detections for one frame."""
    return run_ppe_detection(frame)


def merge_helmet_results(primary, extra):
    """
    Merge two lists of helmet detections without double counting.

    If the dedicated hard hat model and best.pt both put a box on the same
    head, that is ONE helmet, not two. Boxes overlapping by more than
    HELMET_DUPLICATE_IOU are treated as the same helmet and only the more
    confident one survives.
    """
    merged = list(primary or [])

    for candidate in extra or []:
        if not isinstance(candidate, dict):
            continue

        candidate_box = _box_of(candidate)
        if candidate_box is None:
            continue

        duplicate_index = -1

        for index, existing in enumerate(merged):
            existing_box = _box_of(existing)
            if existing_box is None:
                continue
            if calculate_iou(candidate_box, existing_box) >= HELMET_DUPLICATE_IOU:
                duplicate_index = index
                break

        if duplicate_index < 0:
            merged.append(candidate)
            continue

        # Same helmet seen twice - keep whichever is more confident.
        existing = merged[duplicate_index]
        if safe_float(candidate.get("confidence")) > safe_float(
            existing.get("confidence")
        ):
            merged[duplicate_index] = candidate

    return merged


# =====================================================================
# 7. ATTACHING PPE TO THE CORRECT PERSON
# =====================================================================

def _person_region(person_box, category):
    """
    The slice of a person's box where this PPE item should appear.

    The helmet fraction is negative at the top, so the region extends
    ABOVE the person's box. A hard hat sticks up past the head and YOLO's
    person box often starts at the hairline; without this the helmet box
    falls outside and scores 0% overlap.
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1

    top_fraction, bottom_fraction = PPE_REGIONS.get(category, (0.0, 1.0))

    region_y1 = y1 + int(height * top_fraction)
    region_y2 = y1 + int(height * bottom_fraction)

    if region_y2 <= region_y1:
        region_y2 = region_y1 + 1

    return [x1, region_y1, x2, region_y2]


def _box_centre(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _centre_inside(inner_box, outer_box, margin=0.0):
    """Is the middle of inner_box inside outer_box?"""
    if outer_box is None:
        return False

    cx, cy = _box_centre(inner_box)
    x1, y1, x2, y2 = outer_box

    pad_x = (x2 - x1) * margin
    pad_y = (y2 - y1) * margin

    return (x1 - pad_x) <= cx <= (x2 + pad_x) and (y1 - pad_y) <= cy <= (y2 + pad_y)


def _best_person_for(ppe_box, boxes, require_centre_in=None):
    """
    Which of these boxes does the PPE box sit inside the most?

    require_centre_in: when given, a person only qualifies if the PPE
    box's CENTRE falls inside their box. Overlap alone is not enough when
    two people stand close together - one person's shoulder can clip into
    their neighbour's box and steal the detection. The centre point
    cannot be in two people at once.
    """
    best_index = -1
    best_overlap = 0.0

    for index, box in enumerate(boxes):
        if box is None:
            continue

        if require_centre_in is not None:
            if not _centre_inside(ppe_box, require_centre_in[index], 0.05):
                continue

        overlap = intersection_over_area(ppe_box, box)
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index

    return best_index, best_overlap


def _record_ppe(person, category, detection, overlap, source):
    """Store a PPE result on a person, keeping the most confident one."""
    confidence = safe_float(detection.get("confidence", 0.0))

    confidence_key = f"{category}_confidence"
    current_confidence = safe_float(person.get(confidence_key, 0.0))
    current_status = person.get(category, "unknown")

    # If the model reports both "Hardhat" and "NO-Hardhat" on one head,
    # the more confident one wins.
    if current_status != "unknown" and confidence <= current_confidence:
        return False

    person[category] = detection.get("status", "unknown")
    person[confidence_key] = round(confidence, 3)

    # Hard hat colour, when the helmet model provided one.
    if category == "helmet":
        helmet_colour = detection.get("helmet_color")
        if helmet_colour and helmet_colour not in ("unknown", "none"):
            person["helmet_color"] = helmet_colour
    person[f"{category}_bbox"] = _box_of(detection)
    person[f"{category}_source"] = source
    person[f"{category}_overlap"] = round(overlap, 3)
    return True


def assign_ppe_to_people(people, ppe_detections, category):
    """
    Give each PPE detection to exactly one person, in two stages.

    STAGE 1 - strict.
        The PPE box must sit inside the right part of the body: a helmet
        on the head, a mask on the face. This is what stops one person's
        helmet being credited to the person standing next to them.

    STAGE 2 - second chance.
        Anything stage 1 could not place is retried against the person's
        WHOLE box.

        Measured on a close-range webcam: best.pt returned NO-Hardhat at
        0.39 confidence with the box at [129, 272, 367, 479] - over the
        torso, not the head. Against the head region that scores 0%, so a
        perfectly good detection was thrown away.

        Stage 2 still picks the BEST matching person, so with several
        people in shot the detection goes to the right one.
    """
    if not people or not ppe_detections:
        return

    person_boxes = [_box_of(person) for person in people]
    regions = [
        _person_region(box, category) if box else None
        for box in person_boxes
    ]

    unmatched = []

    # ---- stage 1: the right part of the body ----
    for detection in ppe_detections:
        ppe_box = _box_of(detection)
        if ppe_box is None:
            continue

        index, overlap = _best_person_for(ppe_box, regions)

        if index >= 0 and overlap >= PPE_MATCH_THRESHOLD:
            _record_ppe(people[index], category, detection, overlap, "model")
        else:
            unmatched.append(detection)

    if not unmatched or PPE_FALLBACK_MATCH_THRESHOLD <= 0:
        return

    # ---- stage 2: anywhere on the person ----
    for detection in unmatched:
        ppe_box = _box_of(detection)
        if ppe_box is None:
            continue

        index, overlap = _best_person_for(
            ppe_box, person_boxes, require_centre_in=person_boxes
        )

        if index < 0 or overlap < PPE_FALLBACK_MATCH_THRESHOLD:
            # Not on anybody at all. Leave it unassigned rather than
            # forcing it onto the nearest person.
            continue

        _record_ppe(
            people[index], category, detection, overlap, "model (loose)"
        )



# =====================================================================
# PPE ON PERSON CROPS
# =====================================================================
#
# THE PROBLEM WITH RUNNING PPE ON THE WHOLE FRAME
#
# Measured on this project's own camera: a person 129 px tall means a
# face about 51 px, and a mask on that face about 22 px. YOLO needs
# roughly 30x30 px to recognise a small object. So mask and goggles were
# never going to be found, at any confidence threshold.
#
# THE FIX
#
# Take each person's box, crop it, and run the PPE model on THAT crop at
# 416x416. A 243x180 crop scaled to 416 makes the face ~87 px and the
# mask ~39 px. Now there are enough pixels.
#
# It also fixes association for free: PPE found inside person A's crop
# belongs to person A. There is no spatial guessing at all.
#
# WHICH ONE ACTUALLY WORKS DEPENDS ON HOW THE MODEL WAS TRAINED.
#
# Cropping gives the model more pixels, and the arithmetic for that is
# right: a 22 px mask in a full frame becomes 107 px in a crop.
#
# But a model only handles pictures that look like the ones it learned
# from. This model was trained on Roboflow images of whole scenes -
# rooms, desks, several people. A tight crop of one person is a kind of
# picture it has never seen, and it falls over on it.
#
# Measured on this project's own camera:
#
#     ppe_live.py, whole frame   goggles 0.74, in 95 of 124 frames
#     dashboard, person crops    nothing at all, not even at 0.10
#
# Same model, same camera, opposite results. So "frame" is the default:
# it is what the model was trained for, and it is one inference instead
# of up to eight.
#
#   PPE_MODE = "frame"  one pass over the whole frame   <- default
#              "crop"   a pass per person (more pixels, different
#                       kind of picture - only helps if the model was
#                       trained on crops)
#              "both"   run both and keep whatever fires. Slowest, but
#                       it cannot miss because of this choice.
PPE_MODE = os.getenv("PPE_MODE", "frame").strip().lower()

# How much to grow the person's box before cropping. Extra on top so a
# hard hat sticking above the head is inside the crop.
PPE_CROP_MARGIN = _env_float("PPE_CROP_MARGIN", 0.18)
PPE_CROP_TOP_MARGIN = _env_float("PPE_CROP_TOP_MARGIN", 0.35)

# What size the crop is fed to the model at. Larger sees more detail and
# costs more CPU.
PPE_CROP_IMGSZ = _env_int("PPE_CROP_IMGSZ", 640)

# One inference per person, so cap it. Biggest people first - they are
# closest and most likely to give a usable result.
PPE_MAX_CROPS = _env_int("PPE_MAX_CROPS", 4)

# Skip anyone too small to be worth cropping.
PPE_CROP_MIN_HEIGHT = _env_int("PPE_CROP_MIN_HEIGHT", 40)

# =====================================================================
# HOW BIG A PERSON HAS TO BE BEFORE PPE IS EVEN ATTEMPTED
# =====================================================================
#
# This is what stops confident nonsense.
#
# YOLO needs roughly 30 px on an object to recognise it. Working back
# from how large each item is relative to a person:
#
#     helmet   22% of person height  ->  person must be 136 px
#     mask     18%                   ->  person must be 166 px
#     specs    14%                   ->  person must be 214 px
#
# Measured on this office:
#
#     camera 1, someone 1-2 m away    300 px   all three fine
#     camera 1, someone 4 m away      130 px   none reliable
#     camera 2, people seated 4-5 m    90 px   none reliable
#
# The old floor was 40 px for everything, so the model was being asked
# about masks on 90 px people - and answered "mask 79%" for people who
# were not wearing one. It was not lying; it simply cannot see at that
# size, and a model always returns its best guess.
#
# Below these sizes the answer is now "unknown". An honest "do not know"
# is worth far more than a confident wrong answer.
#
# Lower them if you would rather have weak guesses than blanks.
# One switch to turn the whole size gate off.
#
#     $env:PPE_SIZE_GATE="0"
#
# The gate was added when the OLD model was reporting "mask 79%" on
# people 90 px tall who were not wearing one - it could not see at that
# size and a model always returns its best guess.
#
# The model you trained is far better: 0.81 on no_helmet where the old
# one managed 0.09. With a model this good the gate mostly gets in the
# way, blocking people it could have judged correctly.
#
# Off, everyone detected gets checked. Voting across three seconds and
# the per-class confidence thresholds are what keep nonsense out now.
PPE_SIZE_GATE = _env_flag("PPE_SIZE_GATE", "1")

# Lowered from 130 / 160 / 200. Those were set for a model that could
# not see, and on a 720p office camera they blocked nearly everybody -
# "no mask" came back empty because every person was under 160 px.
PPE_MIN_PERSON_HEIGHT = {
    "helmet": _env_int("PPE_MIN_PERSON_HELMET", 80),
    "mask": _env_int("PPE_MIN_PERSON_MASK", 90),
    "specs": _env_int("PPE_MIN_PERSON_SPECS", 140),
    "vest": _env_int("PPE_MIN_PERSON_VEST", 80),
    "gloves": _env_int("PPE_MIN_PERSON_GLOVES", 110),
}


if PPE_SIZE_GATE:
    print(f"[DETECTOR] PPE size gate       : ON   {PPE_MIN_PERSON_HEIGHT}")
else:
    print("[DETECTOR] PPE size gate       : OFF  (everyone is checked)")


def ppe_categories_for_person(person_height):
    """Which PPE items are worth asking about for a person this tall."""
    if not PPE_SIZE_GATE:
        return set(PPE_CATEGORIES)

    return {
        category
        for category, minimum in PPE_MIN_PERSON_HEIGHT.items()
        if person_height >= minimum
    }

# Alternate the two crops instead of running both every frame.
#
# Body crop finds helmet / vest / gloves. Face crop finds mask / specs.
# Running both on every frame doubles the PPE cost.
#
# With this on, one frame does bodies and the next does faces. PPE memory
# holds each result for PPE_MEMORY_SECONDS, so at 3 FPS every category is
# still refreshed about 1.5 times a second - far faster than anyone can
# put a helmet on - while the per-frame cost is halved.
PPE_ALTERNATE_CROPS = _env_flag("PPE_ALTERNATE_CROPS", "1")

# Counts detection cycles, so the two crops can take turns.
_ppe_cycle = 0

# The FACE crop, used for mask and specs.
#
# A full-body crop scaled to 640 still leaves the face small - the face
# is only 15-20% of body height. A separate, tighter head crop scaled to
# the same 640 makes the face itself fill the input.
#
# Width matters as much as height. YOLO letterboxes to a square, so the
# scale comes from the longer side; keeping full body width throws the
# zoom away.
# Which categories the FACE crop is allowed to contribute.
#
# Default is mask and specs, because best.pt cannot do helmets from a
# face crop. Once you train your own model on HEAD crops - which include
# the hard hat - add helmet here and the face crop handles all three:
#
#     $env:PPE_FACE_CATEGORIES="mask,specs,helmet"
PPE_FACE_CATEGORIES = tuple(
    part.strip().lower()
    for part in os.getenv("PPE_FACE_CATEGORIES", "mask,specs").split(",")
    if part.strip()
)

PPE_FACE_WIDTH = _env_float("PPE_FACE_WIDTH", 0.55)   # of person width
PPE_FACE_TOP = _env_float("PPE_FACE_TOP", 0.10)       # above the box
PPE_FACE_BOTTOM = _env_float("PPE_FACE_BOTTOM", 0.42) # down the box


def _expand_box(box, frame_shape, margin, top_margin):
    """Grow a box, keeping it inside the picture."""
    x1, y1, x2, y2 = box
    height, width = frame_shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1

    ex1 = int(x1 - box_width * margin)
    ex2 = int(x2 + box_width * margin)
    ey1 = int(y1 - box_height * top_margin)
    ey2 = int(y2 + box_height * margin)

    ex1 = max(0, min(ex1, width - 1))
    ey1 = max(0, min(ey1, height - 1))
    ex2 = max(0, min(ex2, width))
    ey2 = max(0, min(ey2, height))

    if ex2 <= ex1 or ey2 <= ey1:
        return None

    return [ex1, ey1, ex2, ey2]


def run_ppe_on_person_crops(frame, people):
    """
    Run PPE model on each person: ONE full-body crop (helmet/vest/gloves)
    PLUS one zoomed-in FACE crop (mask/specs).

    Face crop fixes this: a full-body crop resized to 640px still leaves
    the face tiny (it's ~15-20% of body height), so mask/goggles never
    had enough pixels to be detected. A dedicated head-region crop,
    zoomed in and resized to 640px on its own, makes the face itself
    large enough to detect.
    """
    global _ppe_cycle

    results = {}

    if frame is None or not people:
        return results

    _ppe_cycle += 1

    # Take turns. Both still run when alternating is switched off.
    if PPE_ALTERNATE_CROPS:
        do_body = (_ppe_cycle % 2) == 1
        do_face = not do_body
    else:
        do_body = True
        do_face = True

    order = sorted(
        range(len(people)),
        key=lambda i: -(
            (_box_of(people[i]) or [0, 0, 0, 0])[3]
            - (_box_of(people[i]) or [0, 0, 0, 0])[1]
        ),
    )[:PPE_MAX_CROPS]

    for index in order:
        box = _box_of(people[index])
        if box is None:
            continue

        person_height = box[3] - box[1]

        if person_height < PPE_CROP_MIN_HEIGHT:
            continue

        # Which items are worth asking about at this size?
        allowed = ppe_categories_for_person(person_height)

        # None of them: do not run the model on this person at all.
        #
        # enforce_ppe_size_limits() would throw the answers away
        # afterwards anyway, so running the inference just burns CPU for
        # a result nobody sees. Measured on camera 13: nine people at
        # about 90 px each, two crops per person, 350 ms per crop - that
        # is over 5 seconds a frame spent producing nothing usable, and
        # it is why a frame was taking 7.66 seconds.
        if not allowed:
            people[index]["ppe_skipped"] = "person too small for any PPE"
            people[index]["person_height"] = person_height
            continue

        # Only run the crop that can actually answer something.
        want_body = bool(allowed & {"helmet", "vest", "gloves"})
        want_face = bool(allowed & {"mask", "specs"})

        all_detections = []
        body_face_fallback = []

        # ---- CROP 1: full body (helmet, vest, gloves) ----
        expanded = _expand_box(
            box, frame.shape, PPE_CROP_MARGIN, PPE_CROP_TOP_MARGIN
        ) if (do_body and want_body) else None

        if expanded is not None:
            cx1, cy1, cx2, cy2 = expanded
            crop = frame[cy1:cy2, cx1:cx2]

            if crop is not None and crop.size > 0:
                try:
                    detections = run_ppe_detection(crop, imgsz=PPE_CROP_IMGSZ)
                except Exception as exc:
                    print(f"[PPE] body crop inference failed: {type(exc).__name__}: {exc}")
                    detections = []

                # The dedicated hard hat model, on the SAME crop.
                # Running it on the whole frame has the same pixel
                # problem as best.pt did; on a person crop it gets a
                # proper look at the head.
                if USE_DEDICATED_HELMET_MODEL:
                    try:
                        detections = detections + run_helmet_model(
                            crop, imgsz=PPE_CROP_IMGSZ
                        )
                    except Exception as exc:
                        print(f"[HELMET] crop failed: "
                              f"{type(exc).__name__}: {exc}")

                for detection in detections:
                    if detection.get("category") == "person":
                        continue
                    # Mask/specs are preferred from the face crop below,
                    # which has far better pixel density. But hold on to
                    # the body-crop versions as a fallback - if the face
                    # crop finds nothing, losing them entirely would be
                    # worse than using the weaker result.
                    if detection.get("category") in PPE_FACE_CATEGORIES:
                        x1, y1, x2, y2 = detection["bbox"]
                        spare = dict(detection)
                        spare["bbox"] = [x1 + cx1, y1 + cy1, x2 + cx1, y2 + cy1]
                        spare["x1"], spare["y1"] = x1 + cx1, y1 + cy1
                        spare["x2"], spare["y2"] = x2 + cx1, y2 + cy1
                        spare["from_crop"] = True
                        spare["from_body_crop"] = True
                        body_face_fallback.append(spare)
                        continue
                    x1, y1, x2, y2 = detection["bbox"]
                    fx1, fy1 = x1 + cx1, y1 + cy1
                    fx2, fy2 = x2 + cx1, y2 + cy1
                    moved = dict(detection)
                    moved["bbox"] = [fx1, fy1, fx2, fy2]
                    moved["x1"], moved["y1"] = fx1, fy1
                    moved["x2"], moved["y2"] = fx2, fy2
                    moved["from_crop"] = True
                    all_detections.append(moved)

        # ---- CROP 2: face only, zoomed in (mask, specs) ----
        px1, py1, px2, py2 = box
        person_height = py2 - py1
        person_width = px2 - px1

        # NARROW IN X TOO, not just Y.
        #
        # Keeping the full body width wastes the zoom. YOLO letterboxes
        # to a square, so the scale is set by the LONGER side. Measured:
        #
        #   full width  315 x 66  -> 640/315 = 2.03x   (no better than body)
        #   head width  133 x 66  -> 640/133 = 4.81x   (2.4x better)
        #
        # A head is roughly a third of shoulder width. PPE_FACE_WIDTH is
        # generous at 0.55 so a turned head or wide-brimmed hat still fits.
        centre_x = (px1 + px2) // 2
        half_face = max(12, int(person_width * PPE_FACE_WIDTH * 0.5))

        face_y1 = py1 - int(person_height * PPE_FACE_TOP)
        face_y2 = py1 + int(person_height * PPE_FACE_BOTTOM)
        face_x1 = centre_x - half_face
        face_x2 = centre_x + half_face

        # A very wide, very short crop still wastes most of the model's
        # input. Pad the shorter side towards square so more of the 640
        # actually contains the head.
        face_width = face_x2 - face_x1
        face_height = face_y2 - face_y1

        if face_height < face_width * 0.7:
            extra = int((face_width * 0.7 - face_height) / 2)
            face_y1 -= extra
            face_y2 += extra

        fh, fw = frame.shape[:2]
        face_x1 = max(0, min(face_x1, fw - 1))
        face_y1 = max(0, min(face_y1, fh - 1))
        face_x2 = max(0, min(face_x2, fw))
        face_y2 = max(0, min(face_y2, fh))

        if do_face and want_face and face_x2 - face_x1 > 5 and face_y2 - face_y1 > 5:
            face_crop = frame[face_y1:face_y2, face_x1:face_x2]

            if face_crop is not None and face_crop.size > 0:
                try:
                    # Zoomed-in face crop, resized to the SAME 640px
                    # target -> face pixels are now much bigger relative
                    # to the model's input than in the full-body crop.
                    face_detections = run_ppe_detection(
                        face_crop, imgsz=PPE_CROP_IMGSZ
                    )
                except Exception as exc:
                    print(f"[PPE] face crop inference failed: {type(exc).__name__}: {exc}")
                    face_detections = []

                # If a second, face-specialised model is installed, run
                # it on the same crop. best.pt is a construction model
                # and hardly ever reports Mask or Goggles on indoor
                # close-range footage; a face-mask model trained on
                # webcam faces does far better. Both results are kept
                # and the more confident one wins later.
                if FACE_PPE_MODEL_AVAILABLE:
                    try:
                        face_detections = face_detections + (
                            run_face_ppe_detection(
                                face_crop, imgsz=PPE_CROP_IMGSZ
                            )
                        )
                    except Exception as exc:
                        print(f"[FACE-PPE] crop failed: "
                              f"{type(exc).__name__}: {exc}")

                for detection in face_detections:
                    if detection.get("category") not in PPE_FACE_CATEGORIES:
                        continue
                    x1, y1, x2, y2 = detection["bbox"]
                    fx1, fy1 = x1 + face_x1, y1 + face_y1
                    fx2, fy2 = x2 + face_x1, y2 + face_y1
                    moved = dict(detection)
                    moved["bbox"] = [fx1, fy1, fx2, fy2]
                    moved["x1"], moved["y1"] = fx1, fy1
                    moved["x2"], moved["y2"] = fx2, fy2
                    moved["from_crop"] = True
                    moved["from_face_crop"] = True
                    all_detections.append(moved)

        # Face crop found no mask or specs? Use the weaker body-crop
        # versions rather than reporting nothing at all.
        found_face_categories = {
            d.get("category") for d in all_detections
            if d.get("from_face_crop")
        }
        for spare in body_face_fallback:
            if spare.get("category") not in found_face_categories:
                all_detections.append(spare)

        if all_detections:
            results[index] = all_detections

    return results


def assign_crop_ppe(people, crop_results):
    """
    Attach crop detections to the person they were cropped from.

    Nothing spatial here - the crop already identified the person. For
    each category the most confident detection wins, so a head showing
    both Hardhat and NO-Hardhat resolves to the stronger one.

    Anything the person is too small to judge is dropped, however
    confident the model sounded. That is what removes "mask 79%" from
    someone 90 px tall on the far side of the room.
    """
    assigned = 0
    rejected_small = 0

    for index, detections in crop_results.items():
        if index >= len(people):
            continue

        person = people[index]

        box = _box_of(person)
        person_height = (box[3] - box[1]) if box else 0

        allowed = ppe_categories_for_person(person_height)

        # Record what was skipped, so the reason is visible rather than
        # the answer silently going missing.
        too_small = [
            category for category in PPE_CATEGORIES
            if category not in allowed
        ]
        if too_small:
            person["ppe_too_small"] = too_small
            person["person_height"] = person_height

        for detection in detections:
            category = detection.get("category")

            if category not in PPE_CATEGORIES:
                continue

            if category not in allowed:
                rejected_small += 1
                continue

            if _record_ppe(person, category, detection, 1.0, "model (crop)"):
                assigned += 1

    return assigned, rejected_small


# =====================================================================
# 8. PPE MEMORY - stops the live status flickering
# =====================================================================
#
# Your measurements: NO-Hardhat 0.18 / 0.27 / 0.34 / 0.41. The model does
# not fire on every frame. Without memory the label blinks on and off and
# the counts jump about, which on screen looks like "not working".
#
# So a confident result is held against that person's track_id for a few
# seconds. Nothing is ever remembered that the model did not report for
# that person first, and everything expires.

_ppe_memory = {}

# =====================================================================
# VOTING ACROSS FRAMES
# =====================================================================
#
# Until now each frame decided on its own, and memory simply held the
# last answer. That makes one bad frame enough to be wrong:
#
#     frame 1   nothing
#     frame 2   nothing
#     frame 3   mask 0.79     <- one bad frame
#     frame 4   nothing
#
#   -> reported "mask", and it stays for the whole memory window.
#
# A person does not put a mask on and take it off between frames. So
# collect evidence over a few seconds and let it vote:
#
#     three sightings of "no mask" beat one of "mask"
#     one solid "specs" carries when nothing contradicts it
#
# Each observation is weighted by its own confidence, so a 0.8 counts
# for more than a 0.2, and a winner must clear a share of the total
# before it is reported at all. Otherwise: unknown.
#
# This is what turns a flickering feed into a stable answer, and it is
# the single biggest accuracy gain available without retraining.

# How long to keep observations. At 0.2-1 FPS a three second window
# often held less than one frame, so the vote had almost nothing to work
# with. Twenty seconds gives a handful of observations at any frame rate
# this server realistically runs at.
# OFF by default.
#
# Voting earned its place against the OLD model, which produced noise at
# 0.09 to 0.13 and needed several frames to be argued down.
#
# The model you trained reports 0.65 on a correct no_mask. That is not
# noise, and voting was throwing it away: the log showed
#
#     [PPE] kept 2 {'no mask': 2}      model found two
#     [PPE] ... voted: 2               voting changed two
#     [MASK] mask: 0, no mask: 0       nothing survived
#
# Voting needs several observations of the same person to build a
# majority, and at roughly 0.1 FPS a twenty second window holds about
# two frames. There is never enough to agree on, so everything falls
# back to unknown.
#
# The per-class confidence thresholds now do the filtering instead, and
# they judge each detection on its own merit rather than needing a crowd.
#
# Turn it back on if false positives return - it is genuinely useful at
# 2 FPS or more:
#     $env:PPE_VOTE_SECONDS="20"
PPE_VOTE_SECONDS = _env_float("PPE_VOTE_SECONDS", 0)

# A sighting this confident is accepted on its own.
#
# Voting exists to reject noise, and noise is weak - the old model's
# nonsense came in at 0.09 to 0.13. A detection at 0.5+ is not noise.
#
# Without this a correct 0.81 helmet seen in one frame out of five was
# outvoted by the four frames that missed it, and reported as unknown.
# Missing a detection is normal; a confident false positive is not.
PPE_VOTE_STRONG = _env_float("PPE_VOTE_STRONG", 0.30)

# How much of the total weight the winner needs. Below this the
# observations disagree too much to call.
PPE_VOTE_SHARE = _env_float("PPE_VOTE_SHARE", 0.60)

# A single weak sighting should not decide anything.
PPE_VOTE_MIN_WEIGHT = _env_float("PPE_VOTE_MIN_WEIGHT", 0.35)

# "The model looked and found nothing" is evidence too - it is what
# outvotes a single false hit. Weaker than a real sighting, but nine of
# them beat one.
NOT_SEEN = "__not_seen__"
PPE_VOTE_ABSENT_WEIGHT = _env_float("PPE_VOTE_ABSENT_WEIGHT", 0.15)

PPE_VOTE_MAX_AGE = 60.0

# {(scope, track_id, category): [(status, confidence, time), ...]}
_ppe_votes = {}


def _prune_votes(now):
    stale = [
        key for key, entries in _ppe_votes.items()
        if not entries or now - entries[-1][2] > PPE_VOTE_MAX_AGE
    ]
    for key in stale:
        _ppe_votes.pop(key, None)


def record_ppe_vote(scope, track_id, category, status, confidence):
    """Remember one observation."""
    if track_id is None:
        return

    key = (id(scope) if scope is not None else 0, track_id, category)
    now = time.time()

    entries = _ppe_votes.setdefault(key, [])
    entries.append((status, float(confidence), now))

    cutoff = now - PPE_VOTE_SECONDS
    _ppe_votes[key] = [e for e in entries if e[2] >= cutoff][-40:]


def read_ppe_vote(scope, track_id, category):
    """
    What do the last few seconds say?

    Returns (status, confidence, agreement) or (None, 0, 0) when there
    is not enough to call.
    """
    if track_id is None:
        return None, 0.0, 0.0

    key = (id(scope) if scope is not None else 0, track_id, category)
    entries = _ppe_votes.get(key)

    if not entries:
        return None, 0.0, 0.0

    now = time.time()
    cutoff = now - PPE_VOTE_SECONDS
    recent = [e for e in entries if e[2] >= cutoff]

    if not recent:
        return None, 0.0, 0.0

    weights = {}
    best_confidence = {}

    for status, confidence, _when in recent:
        weights[status] = weights.get(status, 0.0) + confidence
        best_confidence[status] = max(
            best_confidence.get(status, 0.0), confidence
        )

    # A confident sighting stands on its own. The model missing something
    # in other frames is expected; it does not make the sighting wrong.
    strong = {
        status: value
        for status, value in best_confidence.items()
        if status != NOT_SEEN and value >= PPE_VOTE_STRONG
    }

    if strong:
        winner = max(strong, key=strong.get)
        return winner, strong[winner], 1.0

    total = sum(weights.values())

    if total < PPE_VOTE_MIN_WEIGHT:
        return None, 0.0, 0.0

    winner = max(weights, key=weights.get)
    share = weights[winner] / total

    if winner == NOT_SEEN:
        # The model kept looking and kept finding nothing.
        return None, 0.0, share

    if share < PPE_VOTE_SHARE:
        # The sightings disagree. Saying nothing is the honest answer.
        return None, 0.0, share

    return winner, best_confidence[winner], share


def apply_ppe_voting(people, scope):
    """
    Replace each person's per-frame answer with the vote.

    Runs after assignment, so it sees this frame's result, records it,
    and then reports whatever the last few seconds agree on.
    """
    if PPE_VOTE_SECONDS <= 0:
        return 0

    now = time.time()
    _prune_votes(now)

    changed = 0

    for person in people:
        track_id = person.get("track_id")

        if track_id is None:
            continue

        box = _box_of(person)
        person_height = (box[3] - box[1]) if box else 0
        allowed = ppe_categories_for_person(person_height)

        for category in PPE_CATEGORIES:
            status = person.get(category, "unknown")

            if status and status != "unknown":
                record_ppe_vote(
                    scope, track_id, category, status,
                    safe_float(person.get(f"{category}_confidence", 0.0)),
                )
            elif category in allowed:
                # The model DID look at this person for this item and
                # came back with nothing. That counts against whatever
                # a stray frame claimed earlier.
                #
                # Only when it actually looked - a person too small to
                # judge tells us nothing either way.
                record_ppe_vote(
                    scope, track_id, category, NOT_SEEN,
                    PPE_VOTE_ABSENT_WEIGHT,
                )

            voted, confidence, agreement = read_ppe_vote(
                scope, track_id, category
            )

            if voted is None:
                if status != "unknown":
                    changed += 1
                person[category] = "unknown"
                person[f"{category}_confidence"] = 0.0
                person[f"{category}_agreement"] = round(agreement, 2)
                continue

            if voted != status:
                changed += 1

            person[category] = voted
            person[f"{category}_confidence"] = round(confidence, 3)
            person[f"{category}_agreement"] = round(agreement, 2)
            person[f"{category}_source"] = "vote"

    return changed


def reset_ppe_votes():
    """Forget every vote. Useful when a camera restarts."""
    _ppe_votes.clear()



def _prune_ppe_memory(now):
    """Drop entries for people who left the frame a while ago."""
    stale = [
        track_id
        for track_id, entry in _ppe_memory.items()
        if now - entry.get("seen", 0.0) > PPE_MEMORY_MAX_AGE
    ]
    for track_id in stale:
        _ppe_memory.pop(track_id, None)


def apply_ppe_memory(people):
    """
    Remember fresh results, and fill in gaps from recent ones.

    A person keeps their last known helmet/mask/specs status for
    PPE_MEMORY_SECONDS after the model last confirmed it.
    """
    if PPE_MEMORY_SECONDS <= 0:
        return 0

    now = time.time()
    _prune_ppe_memory(now)

    restored = 0

    for person in people:
        track_id = person.get("track_id")
        if track_id is None:
            continue

        entry = _ppe_memory.setdefault(track_id, {"seen": now, "ppe": {}})
        entry["seen"] = now
        remembered = entry["ppe"]

        for category in PPE_CATEGORIES:
            status = person.get(category, "unknown")

            if status and status != "unknown":
                # Fresh result from the model. Store it.
                remembered[category] = {
                    "status": status,
                    "confidence": safe_float(
                        person.get(f"{category}_confidence", 0.0)
                    ),
                    "time": now,
                }
                continue

            # Nothing this frame. Use the remembered value if it is recent.
            previous = remembered.get(category)
            if not previous:
                continue

            age = now - previous["time"]
            if age > PPE_MEMORY_SECONDS:
                continue

            person[category] = previous["status"]
            person[f"{category}_confidence"] = previous["confidence"]
            person[f"{category}_source"] = "remembered"
            person[f"{category}_age"] = round(age, 1)
            restored += 1

    return restored


def reset_ppe_memory():
    """Forget everything. Useful when a camera restarts."""
    _ppe_memory.clear()
    _ppe_votes.clear()


def enforce_ppe_size_limits(people):
    """
    Clear any PPE answer for a person too small to judge.

    This runs LAST, after crop assignment, the spatial passes, memory and
    inference. Filtering earlier is not enough: the spatial pass would
    put the detection back, and memory would restore it a frame later.

    It is the difference between "mask 79%" on someone 90 px tall across
    the room, and an honest "unknown".
    """
    cleared = 0

    for person in people:
        box = _box_of(person)
        if box is None:
            continue

        person_height = box[3] - box[1]
        allowed = ppe_categories_for_person(person_height)

        blocked = []

        for category in PPE_CATEGORIES:
            if category in allowed:
                continue

            if person.get(category, "unknown") != "unknown":
                cleared += 1

            person[category] = "unknown"
            person[f"{category}_confidence"] = 0.0
            person[f"{category}_source"] = "too small to judge"
            blocked.append(category)

        if blocked:
            person["ppe_too_small"] = blocked
            person["person_height"] = person_height

        # 'status' mirrors helmet, so keep it honest too.
        if person.get("helmet", "unknown") == "unknown":
            person["status"] = None

    return cleared


def apply_absence_inference(people, frame_shape=None):
    """
    Mark a clearly visible person with no PPE detection as NOT wearing it.

    The logic you asked for, and it is sound: if the person is plainly in
    shot and no helmet was found anywhere on them, they are not wearing a
    helmet.

    It was previously blocked by this line:

        if capability.get("negative"): continue

    which skipped any category where the model HAS a NO- class. Your model
    has all three, so nothing was ever guessed - even though those classes
    barely fire on your webcam. PPE_ABSENCE_MODE="always" now overrides
    that.

    Guessing NEVER overrides the model:
      * a positive detection (Hardhat) wins
      * an explicit negative (NO-Hardhat) wins
      * a value still held in memory wins
      * only a genuine "unknown" is filled in

    And it will not guess when it cannot see properly:
      * the person must be detected confidently
      * they must be big enough in frame
      * for helmet, the top of their head must be inside the picture
    """
    if PPE_ABSENCE_MODE == "off":
        return 0

    frame_height = None
    if frame_shape is not None and len(frame_shape) >= 2:
        frame_height = frame_shape[0]

    inferred_count = 0

    for person in people:
        box = _box_of(person)
        if box is None:
            continue

        x1, y1, x2, y2 = box
        height = y2 - y1

        # Too small to judge - too few pixels for the model to have had a
        # fair chance in the first place.
        if height < PPE_INFER_MIN_PERSON_HEIGHT:
            person["absence_skipped"] = "person too small"
            continue

        # Not a confident person detection.
        if safe_float(person.get("confidence", 0.0)) < PPE_INFER_MIN_PERSON_CONF:
            person["absence_skipped"] = "person confidence too low"
            continue

        head_cut_off = (
            frame_height is not None and y1 <= PPE_INFER_TOP_MARGIN
        )

        for category in PPE_INFER_CATEGORIES:
            # The model already gave an answer, or memory is holding one.
            if person.get(category, "unknown") != "unknown":
                continue

            if PPE_ABSENCE_MODE == "auto":
                # Only fill in where the model has no NO- class at all.
                capability = PPE_CAPABILITIES.get(category, {})
                if capability.get("negative"):
                    continue

            # A helmet sits on top of the head. If the head is cut off by
            # the top of the frame we genuinely cannot tell.
            if category == "helmet" and head_cut_off:
                person["helmet_skipped"] = "head cut off at top of frame"
                continue

            person[category] = f"no {category}"
            person[f"{category}_confidence"] = 0.0
            person[f"{category}_source"] = "inferred"
            inferred_count += 1

    return inferred_count


def detect_ppe_with_absence(frame, people, raw_ppe=None):
    """
    Kept for backward compatibility.

    Returns (helmet_results, specs_results, mask_results) - THREE values,
    exactly as before. The real per-person logic now lives in
    assign_ppe_to_people().
    """
    if frame is None:
        return [], [], []

    if raw_ppe is None:
        raw_ppe = run_ppe_detection(frame)

    buckets = split_ppe_by_category(raw_ppe)
    return buckets["helmet"], buckets["specs"], buckets["mask"]


# =====================================================================
# 9. TRACKER OUTPUT NORMALISATION
# =====================================================================

def _track_to_dict(item):
    """Turn one tracker result into a plain dictionary."""
    if isinstance(item, dict):
        return dict(item)

    result = {}
    for field in ("name", "track_id", "confidence", "color", "bbox",
                  "x1", "y1", "x2", "y2", "class_id", "age", "hits"):
        if hasattr(item, field):
            result[field] = getattr(item, field)

    return result or None


# =====================================================================
# FIXTURES CACHE
# =====================================================================
#
# AC units, cabinets, doors, whiteboards etc. never move on a fixed
# CCTV camera. Running a third model on every single frame to re-find
# the same six boxes was adding real, measured cost - on this CPU,
# per-frame detection time was 12-40 SECONDS against a 0.5s target,
# and stacking fixtures on top of the general model, the dedicated
# helmet model and PPE crops was a big part of why. Re-detecting once
# every 30s per camera is plenty; the wall does not move in that time.
_FIXTURES_CACHE_SECONDS = 30.0
_fixtures_cache: dict[int, tuple[float, list]] = {}


def _get_fixtures_cached(frame, object_tracker):
    import copy
    import time as _time

    key = id(object_tracker) if object_tracker is not None else 0
    now = _time.time()

    cached = _fixtures_cache.get(key)
    if cached is not None and (now - cached[0]) < _FIXTURES_CACHE_SECONDS:
        return copy.deepcopy(cached[1])

    detections = detect_fixtures(frame)
    _fixtures_cache[key] = (now, detections)
    return detections


def run_tracker(object_tracker, detections):
    """
    Feed detections to the tracker and get tracked objects back.

    Returns (tracked_list, description) so the logs say which path ran.
    """
    if object_tracker is None:
        return detections, "no tracker (using raw detections)"

    if not hasattr(object_tracker, "update"):
        return detections, "tracker has no update() (using raw detections)"

    try:
        returned = object_tracker.update(detections)
    except Exception as exc:
        print(f"[TRACKER] update() failed: {type(exc).__name__}: {exc}")
        return detections, "update() failed (using raw detections)"

    candidates = []

    if isinstance(returned, (list, tuple)) and returned:
        candidates.append((list(returned), "update() return value"))

    if hasattr(object_tracker, "get_results"):
        try:
            results = object_tracker.get_results()
            if isinstance(results, (list, tuple)) and results:
                candidates.append((list(results), "get_results()"))
        except Exception as exc:
            print(f"[TRACKER] get_results() failed: {type(exc).__name__}: {exc}")

    for raw_list, description in candidates:
        converted = []
        for item in raw_list:
            as_dict = _track_to_dict(item)
            if as_dict and _box_of(as_dict) is not None:
                converted.append(as_dict)
        if converted:
            return converted, description

    if not detections:
        # Nothing to track in the first place. Normal for an empty frame.
        return [], "no detections in this frame"

    return detections, "tracker returned nothing usable (using raw detections)"


# =====================================================================
# 10. THE MAIN PIPELINE
# =====================================================================

def run_full_detection(frame, object_tracker=None, run_ppe=True):
    """
    Run the whole detection pipeline on one frame.

    object_tracker: an ObjectTracker instance (or None)
    run_ppe:        set False to skip the PPE model and save CPU
    """
    start_ticks = cv2.getTickCount()

    people = []
    other_objects = []
    tracked_objects = []
    raw_ppe = []
    helmets = []
    specs = []
    masks = []
    vests = []
    gloves = []
    tracker_note = "not run"
    unstable = 0
    remembered_count = 0
    crop_assigned = 0
    crop_too_small = 0
    size_cleared = 0
    voted_count = 0
    inferred_count = 0

    if frame is not None:

        # --- 1. general object detection ------------------------------
        raw_objects = detect_objects(frame, with_color=False)

        # --- 2. tracking ----------------------------------------------
        tracked_objects, tracker_note = run_tracker(object_tracker, raw_objects)

        # --- 3. split people from everything else ---------------------
        for item in tracked_objects:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).lower()
            box = _box_of(item)
            if box is None:
                continue

            item["name"] = name
            item["bbox"] = box
            item["x1"], item["y1"], item["x2"], item["y2"] = box

            item.setdefault("track_id", None)
            item.setdefault("confidence", 0.0)
            item.setdefault("status", None)

            # object_tracker.py's Track dataclass fills these in as
            # "Unknown" with a capital U, so setdefault alone is not
            # enough - overwrite anything that is not a real PPE status.
            for field in PPE_CATEGORIES:
                value = str(item.get(field, "") or "").strip().lower()
                if value in ("", "none", "unknown"):
                    value = "unknown"
                item[field] = value
                item.setdefault(f"{field}_confidence", 0.0)

            if name == "person":
                people.append(item)
                continue

            # Everything else has to prove it is really there.
            hits = safe_int(item.get("hits", 1), 1)

            if name not in INSTANT_CLASSES and hits < OBJECT_MIN_HITS:
                unstable += 1
                continue

            other_objects.append(item)

        # --- 3b. office fixtures (AC, cabinet, whiteboard, doors...) ---
        #
        # Static objects on a fixed camera - no tracker/hit-counting
        # needed, they don't flicker the way a fresh YOLO guess on a
        # moving scene does. Optional: quietly does nothing until
        # office_fixtures.pt has been trained.
        if FIXTURES_MODEL_AVAILABLE:
            for fixture in _get_fixtures_cached(frame, object_tracker):
                fixture.setdefault("color", None)
                other_objects.append(fixture)

            # The general COCO model has no classes for AC units,
            # cabinets, whiteboards, doors etc, so on this exact spot it
            # keeps guessing the closest-looking COCO class instead -
            # measured on real footage: the same whiteboard called "tv"
            # AND, separately, "laptop" (a flat rectangle is a flat
            # rectangle to a model that was never taught "whiteboard").
            # The fixtures model was trained specifically on this room
            # and gets these right at 99%+, so wherever a fixture and a
            # general-model guess land on the same spot, trust the
            # specialist and drop the general model's guess - whatever
            # class it picked.
            fixture_names = set(FIXTURES_CLASS_NAMES.values()) if FIXTURES_CLASS_NAMES else set()
            fixture_boxes = [
                _box_of(item) for item in other_objects
                if str(item.get("name", "")).lower() in fixture_names
            ]
            fixture_boxes = [b for b in fixture_boxes if b is not None]

            if fixture_boxes:
                kept = []
                for item in other_objects:
                    name_lower = str(item.get("name", "")).lower()
                    if name_lower not in fixture_names:
                        box = _box_of(item)
                        overlaps_fixture = box is not None and any(
                            calculate_iou(box, fb) > 0.3
                            for fb in fixture_boxes
                        )
                        if overlaps_fixture:
                            continue
                    kept.append(item)
                other_objects = kept

        # --- 3c. cell phone via person-crop model ----------------------
        #
        # The general model's own whole-frame "cell phone" guesses only
        # hit ~84-86% mAP50 on this camera (phone is ~15-20px wide in the
        # full frame) - drop them and use the dedicated crop model's
        # boxes instead, which measured 95.4% on the same real footage.
        specialist_objects = detect_imc_specialist_objects(frame)
        if IMC_SPECIALIST_MODEL_AVAILABLE:
            # Same specialist-overrides-generalist rule as fixtures: drop
            # the general model's own backpack/handbag/bottle guesses
            # wherever the specialist found the same object, keep the
            # specialist's box instead.
            specialist_boxes = [_box_of(item) for item in specialist_objects]
            specialist_boxes = [b for b in specialist_boxes if b is not None]

            def _overlaps_specialist(item):
                if str(item.get("name", "")).lower() not in IMC_SPECIALIST_CLASSES:
                    return False
                box = _box_of(item)
                return box is not None and any(
                    calculate_iou(box, sb) > 0.3 for sb in specialist_boxes
                )

            other_objects = [item for item in other_objects if not _overlaps_specialist(item)]

            for item in specialist_objects:
                item.setdefault("track_id", None)
                item.setdefault("status", None)
                for field in PPE_CATEGORIES:
                    item.setdefault(field, "unknown")
                    item.setdefault(f"{field}_confidence", 0.0)
                other_objects.append(item)

        crop_phones = detect_phones_in_person_crops(frame, people)
        if PHONE_CROP_MODEL_AVAILABLE:
            # Only drop the general model's weak whole-frame guesses once
            # the crop model actually loaded - never leave cell phone
            # with zero detections just because the crop model is missing.
            other_objects = [
                item for item in other_objects
                if str(item.get("name", "")).lower() != "cell phone"
            ]
            for phone in crop_phones:
                phone.setdefault("track_id", None)
                phone.setdefault("status", None)
                for field in PPE_CATEGORIES:
                    phone.setdefault(field, "unknown")
                    phone.setdefault(f"{field}_confidence", 0.0)
                other_objects.append(phone)

        # --- 3c-bis. RT-DETR phone fallback if we still found nothing ---
        if YOLOS_FALLBACK_ENABLED and not any(
            str(item.get("name", "")).lower() == "cell phone" for item in other_objects
        ):
            phone_fallback_hits = detect_phone_fallback_in_crops(frame, people, scope=object_tracker)
            for phone in phone_fallback_hits:
                phone.setdefault("track_id", None)
                phone.setdefault("status", None)
                for field in PPE_CATEGORIES:
                    phone.setdefault(field, "unknown")
                    phone.setdefault(f"{field}_confidence", 0.0)
                other_objects.append(phone)

        # --- 3d. RT-DETR zero-shot fallback + Weighted Boxes Fusion -----
        #
        # Our fine-tuned specialist and the general model both looked at
        # this frame; ask an independent, never-fine-tuned-on-our-cameras
        # model (RT-DETR-L) for a second opinion - it can't repeat the
        # same mistake since it never saw our data. Where it finds a class
        # we found nothing for, add its box outright. Where BOTH models
        # found the same class, fuse the two boxes with Weighted Boxes
        # Fusion instead of just keeping ours - this is what actually
        # fixes an oversized/loose box (e.g. the reported "huge phone
        # box"): WBF averages overlapping boxes weighted by confidence,
        # tightening a loose box using the other model's tighter one,
        # rather than picking one or blindly stacking both.
        if YOLOS_FALLBACK_ENABLED:
            imc_items = [
                item for item in other_objects
                if str(item.get("name", "")).lower() in IMC_SPECIALIST_CLASSES
            ]
            found_classes = {str(item.get("name", "")).lower() for item in imc_items}
            fallback_hits = detect_yolos_fallback_objects(frame, only_classes=set(IMC_SPECIALIST_CLASSES), scope=object_tracker)
            if fallback_hits:
                frame_h, frame_w = frame.shape[:2]
                fused_items, consumed_fallback_idx = _wbf_fuse_same_class(
                    imc_items, fallback_hits, frame_w, frame_h,
                )
                # drop the pre-fusion IMC items, replace with fused versions
                other_objects = [o for o in other_objects if o not in imc_items]
                other_objects.extend(fused_items)
                # any fallback hit for a class we had NOTHING for is a pure add
                for idx, hit in enumerate(fallback_hits):
                    if idx in consumed_fallback_idx:
                        continue
                    if hit["name"] not in found_classes:
                        hit.setdefault("track_id", None)
                        hit.setdefault("status", None)
                        for field in PPE_CATEGORIES:
                            hit.setdefault(field, "unknown")
                            hit.setdefault(f"{field}_confidence", 0.0)
                        other_objects.append(hit)

            # --- cross-class conflict resolution (backpack vs handbag) --
            #
            # RT-DETR sometimes labels the SAME physical bag as both
            # "backpack" and "handbag" at once (verified directly: two
            # boxes at the identical location, 0.33 backpack + 0.28
            # handbag) - WBF above only merges same-class boxes, so this
            # would otherwise show as two objects. Where an IMC-class box
            # heavily overlaps a DIFFERENT IMC-class box, resolve which
            # label is right: if the two detectors are close in confidence
            # (within 0.15 - a genuinely ambiguous case, not one detector
            # clearly wrong), ask CLIP for an independent zero-shot opinion
            # on the crop instead of arbitrarily trusting whichever
            # detector happened to score marginally higher (verified: CLIP
            # correctly called a 0.33-vs-0.28 tie as handbag 0.58 vs
            # backpack 0.39). Otherwise just keep the higher-confidence box.
            imc_now = [
                item for item in other_objects
                if str(item.get("name", "")).lower() in IMC_SPECIALIST_CLASSES
            ]
            to_drop = set()
            for i in range(len(imc_now)):
                for j in range(i + 1, len(imc_now)):
                    a, b = imc_now[i], imc_now[j]
                    if a["name"] == b["name"]:
                        continue
                    box_a, box_b = _box_of(a), _box_of(b)
                    if box_a is None or box_b is None:
                        continue
                    if calculate_iou(box_a, box_b) > 0.5:
                        conf_gap = abs(a["confidence"] - b["confidence"])
                        winner_name = None
                        if conf_gap < 0.35:
                            winner_name = _clip_disambiguate(frame, box_a, [a["name"], b["name"]])
                        if winner_name is not None:
                            loser = a if a["name"] != winner_name else b
                        else:
                            loser = a if a["confidence"] < b["confidence"] else b
                        to_drop.add(id(loser))
            if to_drop:
                other_objects = [o for o in other_objects if id(o) not in to_drop]

            # The overlap check above only fires when TWO competing boxes
            # exist. A lone weak backpack/handbag guess (verified live:
            # "handbag" at 0.15 confidence on what was actually a
            # backpack, with nothing else overlapping it to compare
            # against) sailed straight through un-checked. Give every
            # low-confidence solo guess the same CLIP sanity check instead
            # of trusting it blindly.
            imc_now2 = [
                item for item in other_objects
                if str(item.get("name", "")).lower() in ("backpack", "handbag")
            ]
            relabels = {}
            for item in imc_now2:
                if item.get("confidence", 1.0) >= 0.35:
                    continue
                box = _box_of(item)
                if box is None:
                    continue
                clip_opinion = _clip_disambiguate(frame, box, ["backpack", "handbag"])
                if clip_opinion is not None and clip_opinion != item["name"]:
                    relabels[id(item)] = clip_opinion
            for item in imc_now2:
                new_name = relabels.get(id(item))
                if new_name:
                    item["name"] = new_name

        # Final full-frame sanity pass, applied to EVERY IMC-class box
        # regardless of which path produced it (general COCO model, WBF
        # fusion, phone-crop model, YOLO-World fallback) - this is what
        # actually catches an oversized/implausible box like the reported
        # "phone box too big" bug, since each individual path above only
        # checks its own local area assumption (crop-relative, not
        # frame-relative) or none at all.
        frame_h_final, frame_w_final = frame.shape[:2]
        frame_area_final = frame_w_final * frame_h_final
        if frame_area_final:
            kept_objects = []
            for o in other_objects:
                name_final = str(o.get("name", "")).lower()
                bounds = IMC_FULLFRAME_AREA_BOUNDS.get(name_final)
                if bounds is None:
                    kept_objects.append(o)
                    continue
                ox1, oy1, ox2, oy2 = o.get("bbox", (0, 0, 0, 0))
                area_frac_final = (max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)) / frame_area_final
                if bounds[0] <= area_frac_final <= bounds[1]:
                    kept_objects.append(o)
            other_objects = kept_objects

        # Cross-source duplicate collapse for the 4 IMC classes: the
        # dedicated phone-crop model, the RT-DETR phone-crop fallback, and
        # WBF's own output can all land a box on the SAME physical object
        # when people sit close together (their upscaled crops overlap),
        # each one blind to what the other paths already found. This is
        # what the live "lots of false phone detections" report actually
        # was - not wrong boxes, duplicate boxes on the same real phone.
        # Highest-confidence box wins each overlapping cluster.
        imc_dup_check = [
            o for o in other_objects if str(o.get("name", "")).lower() in IMC_ALLOWED_CLASSES
        ]
        if len(imc_dup_check) > 1:
            imc_dup_check.sort(key=lambda o: o.get("confidence", 0.0), reverse=True)
            keep_ids = []
            dropped_ids = set()
            for o in imc_dup_check:
                o_name = str(o.get("name", "")).lower()
                o_box = o.get("bbox", (0, 0, 0, 0))
                if any(
                    str(k.get("name", "")).lower() == o_name
                    and calculate_iou(o_box, k.get("bbox", (0, 0, 0, 0))) > 0.35
                    for k in keep_ids
                ):
                    dropped_ids.add(id(o))
                    continue
                keep_ids.append(o)
            if dropped_ids:
                other_objects = [o for o in other_objects if id(o) not in dropped_ids]

        # --- 4. colour ------------------------------------------------
        #
        # Cached against the tracker id, refreshed roughly once a second.
        # A shirt does not change colour between frames, and this is the
        # most expensive per-person work in the pipeline.
        #
        # The tracker instance is passed as the SCOPE. Each camera has
        # its own ObjectTracker, so camera 1's track 3 can never be
        # confused with camera 2's track 3.
        for person in people:
            x1, y1, x2, y2 = person["bbox"]
            person["color"] = normalize_color_name(
                get_person_color_cached(
                    frame, x1, y1, x2, y2,
                    scope=object_tracker,
                    track_id=person.get("track_id"),
                )
            )

        for obj in other_objects:
            x1, y1, x2, y2 = obj["bbox"]
            obj["color"] = normalize_color_name(
                get_object_color_cached(
                    frame, x1, y1, x2, y2,
                    scope=object_tracker,
                    track_id=obj.get("track_id"),
                )
            )

        # --- 5. PPE (one inference for helmet + mask + goggles) -------
        #
        # detect_helmets() is the dedicated hard_hat_best.pt model. It
        # returns an empty list unless USE_DEDICATED_HELMET_MODEL=1, so by
        # default only best.pt runs and nothing can be duplicated.
        helmets = detect_helmets(frame)

        specs = []
        masks = []
        vests = []
        gloves = []
        raw_ppe = []

        if run_ppe:

            # --- 5a. PPE on each person's CROP -----------------------
            #
            # This is the method that actually finds small items. The
            # whole frame makes a mask about 22 px across; a person crop
            # blown up to 416 makes it about 39 px. And whatever is found
            # in person A's crop belongs to person A, so association is
            # exact rather than guessed.
            if PPE_MODE in ("crop", "both") and people:
                crop_results = run_ppe_on_person_crops(frame, people)

                crop_assigned, crop_too_small = assign_crop_ppe(
                    people, crop_results
                )

                for detections in crop_results.values():
                    raw_ppe.extend(detections)

            # --- 5b. PPE on the WHOLE FRAME --------------------------
            #
            # Still useful for anything not on a person, and as a
            # fallback when nobody was detected.
            if PPE_MODE in ("frame", "both") or not people:
                frame_ppe = run_ppe_detection(frame)
                raw_ppe.extend(frame_ppe)

            buckets = split_ppe_by_category(raw_ppe)

            # Same three-value contract as before:
            #     helmet_results, specs_results, mask_results
            ppe_helmets = buckets["helmet"]
            specs = buckets["specs"]
            masks = buckets["mask"]
            vests = buckets["vest"]
            gloves = buckets["gloves"]

            # Merge, de-duplicating any helmet both models found.
            helmets = merge_helmet_results(helmets, ppe_helmets)

            # --- 6. attach anything not already placed by a crop ------
            #
            # Crop results were assigned in 5a with source "model (crop)".
            # _record_ppe only overwrites on higher confidence, so these
            # spatial passes fill gaps without undoing exact matches.
            assign_ppe_to_people(people, helmets, "helmet")
            assign_ppe_to_people(people, specs, "specs")
            assign_ppe_to_people(people, masks, "mask")
            assign_ppe_to_people(people, vests, "vest")
            assign_ppe_to_people(people, gloves, "gloves")

            # --- 7. steady the result over time ---------------------
            #
            # Voting and memory both smooth over time, but they must not
            # both run. Memory restores the last answer, voting then
            # treats that restored answer as a FRESH observation and
            # votes for it again - so a single false frame feeds itself
            # and the agreement score climbs instead of falling.
            #
            # Voting is strictly better: it weighs the evidence both
            # ways instead of just repeating the last thing it saw.
            if PPE_VOTE_SECONDS > 0:
                remembered_count = 0
            else:
                # With voting off, memory is what stops the label
                # blinking between frames. It only ever repeats
                # something the model actually reported for that person,
                # and it expires.
                remembered_count = apply_ppe_memory(people)

            inferred_count = apply_absence_inference(people, frame.shape)

            # Vote across the last few seconds. One bad frame can no
            # longer decide anything on its own.
            voted_count = apply_ppe_voting(people, object_tracker)

            # LAST word: anything the person is too small for goes back
            # to unknown, whatever the model, the memory or the vote
            # said.
            size_cleared = enforce_ppe_size_limits(people)

            # Mirror helmet status into 'status' for older frontend code.
            for person in people:
                if person.get("helmet", "unknown") != "unknown":
                    person["status"] = person["helmet"]

    end_ticks = cv2.getTickCount()
    latency_ms = (end_ticks - start_ticks) / cv2.getTickFrequency() * 1000.0

    # --- 8. analytics, counted ONCE PER PERSON ------------------------
    person_count = len(people)

    def count_people_with(category, value):
        return sum(1 for p in people if p.get(category) == value)

    helmet_count = count_people_with("helmet", "helmet")
    no_helmet_count = count_people_with("helmet", "no helmet")

    mask_count = count_people_with("mask", "mask")
    no_mask_count = count_people_with("mask", "no mask")

    specs_count = count_people_with("specs", "specs")
    no_specs_count = count_people_with("specs", "no specs")

    vest_count = count_people_with("vest", "vest")
    no_vest_count = count_people_with("vest", "no vest")

    gloves_count = count_people_with("gloves", "gloves")
    no_gloves_count = count_people_with("gloves", "no gloves")

    def compliance(positive, negative):
        """positive / (positive + negative). Never divides by zero."""
        total = positive + negative
        if total <= 0:
            return 0.0
        return round((positive / total) * 100.0, 1)

    helmet_compliance = compliance(helmet_count, no_helmet_count)
    mask_compliance = compliance(mask_count, no_mask_count)
    specs_compliance = compliance(specs_count, no_specs_count)
    vest_compliance = compliance(vest_count, no_vest_count)
    gloves_compliance = compliance(gloves_count, no_gloves_count)

    color_counts = {}
    for item in people + other_objects:
        color = item.get("color", "unknown")
        color_counts[color] = color_counts.get(color, 0) + 1

    # --- 9. debug -----------------------------------------------------
    _debug([
        f"[OBJECT]  raw detections: {len(tracked_objects)}",
        f"[TRACKER] source: {tracker_note}",
        f"[PEOPLE]  people: {len(people)}   other objects: {len(other_objects)}"
        f"   not yet stable: {unstable}",
        f"[COLOR]   person colors: {[p.get('color') for p in people]}",
        f"[PPE]     mode: {PPE_MODE}   raw: {len(raw_ppe)}   "
        f"from crops: {crop_assigned}   too small: {crop_too_small + size_cleared}   "
        f"remembered: {remembered_count}   voted: {voted_count}   "
        f"inferred: {inferred_count}",
        f"[HELMET]  helmet: {helmet_count}, no helmet: {no_helmet_count}",
        f"[MASK]    mask: {mask_count}, no mask: {no_mask_count}",
        f"[SPECS]   specs: {specs_count}, no specs: {no_specs_count}",
    ])

    return {
        "people": people,
        "objects": other_objects,
        # Same list under the name the dashboard contract asks for.
        "other_objects": other_objects,
        "all_objects": people + other_objects,

        # Calculated AND returned. The old code worked specs and masks
        # out and then threw them away before the return.
        "helmets": helmets,
        "specs": specs,
        "masks": masks,
        "vests": vests,
        "gloves": gloves,
        "ppe": raw_ppe,

        "person_count": person_count,
        "object_count": len(other_objects),

        "helmet_count": helmet_count,
        "no_helmet_count": no_helmet_count,
        "helmet_compliance": helmet_compliance,

        "mask_count": mask_count,
        "no_mask_count": no_mask_count,
        "mask_compliance": mask_compliance,

        "specs_count": specs_count,
        "no_specs_count": no_specs_count,
        "specs_compliance": specs_compliance,

        "vest_count": vest_count,
        "no_vest_count": no_vest_count,
        "vest_compliance": vest_compliance,

        "gloves_count": gloves_count,
        "no_gloves_count": no_gloves_count,
        "gloves_compliance": gloves_compliance,

        "color_counts": color_counts,
        "colors": color_counts,

        "inference_latency_ms": round(latency_ms, 2),
        "ppe_enabled": bool(run_ppe and PPE_MODEL_AVAILABLE),
        "ppe_remembered": remembered_count,
        "ppe_inferred": inferred_count,
        "ppe_from_crops": crop_assigned,
        "ppe_too_small": crop_too_small + size_cleared,
        "ppe_mode": PPE_MODE,
        "tracker_source": tracker_note,
        "objects_unstable": unstable,
    }


# =====================================================================
# 11. SEARCH TERM NORMALISATION
# =====================================================================

SEARCH_ALIASES = {
    "person": "person", "people": "person", "persons": "person",
    "human": "person", "humans": "person", "man": "person",
    "woman": "person", "worker": "person", "workers": "person",

    "helmet": "helmet", "helmets": "helmet", "hardhat": "helmet",
    "hard hat": "helmet", "hard_hat": "helmet", "safety helmet": "helmet",
    "wearing helmet": "helmet", "with helmet": "helmet",

    "no helmet": "no helmet", "nohelmet": "no helmet",
    "no hardhat": "no helmet", "nohardhat": "no helmet",
    "no hard hat": "no helmet", "without helmet": "no helmet",
    "without hardhat": "no helmet", "not wearing helmet": "no helmet",
    "missing helmet": "no helmet",

    "mask": "mask", "masks": "mask", "face mask": "mask",
    "facemask": "mask", "face_mask": "mask", "wearing mask": "mask",
    "with mask": "mask",

    "no mask": "no mask", "nomask": "no mask", "no face mask": "no mask",
    "nofacemask": "no mask", "without mask": "no mask",
    "not wearing mask": "no mask", "missing mask": "no mask",
    "unmasked": "no mask",

    "specs": "specs", "spec": "specs", "goggles": "specs",
    "goggle": "specs", "glasses": "specs", "eyeglasses": "specs",
    "spectacles": "specs", "eyewear": "specs", "safety glasses": "specs",
    "wearing goggles": "specs", "with goggles": "specs",

    "no specs": "no specs", "nospecs": "no specs",
    "no goggles": "no specs", "nogoggles": "no specs",
    "no glasses": "no specs", "noglasses": "no specs",
    "without goggles": "no specs", "without glasses": "no specs",
    "without specs": "no specs", "not wearing goggles": "no specs",

    "vest": "vest", "safety vest": "vest", "hi vis": "vest",
    "high vis": "vest", "reflective vest": "vest",
    "no vest": "no vest", "novest": "no vest",
    "no safety vest": "no vest", "without vest": "no vest",

    "gloves": "gloves", "glove": "gloves",
    "no gloves": "no gloves", "nogloves": "no gloves",
    "without gloves": "no gloves",

    "phone": "cell phone", "mobile": "cell phone",
    "cellphone": "cell phone", "smartphone": "cell phone",
    "bag": "backpack", "monitor": "tv", "television": "tv",
    "table": "dining table", "plant": "potted plant",
    "computer": "laptop", "bike": "bicycle", "motorbike": "motorcycle",

    "grey": "gray", "navy": "blue", "maroon": "red", "crimson": "red",
    "lime": "green", "violet": "purple", "magenta": "purple",
    "beige": "brown", "tan": "brown", "silver": "gray", "gold": "yellow",
}

PPE_SEARCH_TERMS = {
    "helmet", "no helmet",
    "mask", "no mask",
    "specs", "no specs",
    "vest", "no vest",
    "gloves", "no gloves",
    "ppe",
}


def normalize_search(search):
    """
    Turn anything the user types into one clean term.

        'No Mask' / 'no-mask' / 'nomask'  -> 'no mask'
        'HARDHAT'                          -> 'helmet'
    """
    text = str(search or "").strip().lower()
    if not text:
        return ""

    text = text.replace("-", " ").replace("_", " ")
    text = " ".join(text.split())

    if text in SEARCH_ALIASES:
        return SEARCH_ALIASES[text]

    glued = text.replace(" ", "")
    if glued in SEARCH_ALIASES:
        return SEARCH_ALIASES[glued]

    if glued.startswith("no") and len(glued) > 2:
        spaced = "no " + glued[2:]
        if spaced in SEARCH_ALIASES:
            return SEARCH_ALIASES[spaced]

    return text


def search_tokens(search):
    """
    Break a phrase into the meaningful terms it contains.

        'blue person'         -> ['blue', 'person']
        'person without mask' -> ['person', 'no mask']
    """
    text = str(search or "").strip().lower()
    if not text:
        return []

    text = text.replace("-", " ").replace("_", " ")
    words = text.split()

    tokens = []
    index = 0

    while index < len(words):
        matched = False

        for length in (4, 3, 2, 1):
            if index + length > len(words):
                continue
            phrase = " ".join(words[index:index + length])
            resolved = normalize_search(phrase)
            if phrase in SEARCH_ALIASES or resolved != phrase:
                tokens.append(resolved)
                index += length
                matched = True
                break

        if not matched:
            tokens.append(words[index])
            index += 1

    filler = {"a", "an", "the", "with", "wearing", "who", "is",
              "are", "in", "on", "and", "of"}
    return [t for t in tokens if t and t not in filler]


def query_needs_ppe(search):
    """Does this search need the PPE model to run?"""
    if not search:
        return False
    for token in search_tokens(search):
        if token in PPE_SEARCH_TERMS:
            return True
    return False


def detection_matches(detection, search):
    """Does one detection match what the user typed?"""
    if not isinstance(detection, dict):
        return False

    tokens = search_tokens(search)
    if not tokens:
        return True

    name = str(detection.get("name", "")).lower()
    color = normalize_color_name(detection.get("color", ""))

    fields = {
        str(detection.get("helmet", "")).lower(),
        str(detection.get("mask", "")).lower(),
        str(detection.get("specs", "")).lower(),
        str(detection.get("vest", "")).lower(),
        str(detection.get("gloves", "")).lower(),
        str(detection.get("status", "")).lower(),
    }
    fields.discard("")
    fields.discard("none")
    fields.discard("unknown")

    for token in tokens:
        if token == name:
            continue
        if token in COLOR_NAMES and token == color:
            continue
        if token in fields:
            continue
        if token == str(detection.get("category", "")).lower():
            continue
        return False

    return True


# =====================================================================
# 12. MODEL STATUS
# =====================================================================

def get_model_status():
    """Honest report of every model. Nothing here guesses."""
    ppe_status = get_ppe_model_status()

    return {
        "device": DEVICE,

        "general_model_available": GENERAL_MODEL_AVAILABLE,
        "general_model_path": str(GENERAL_MODEL_PATH),
        "general_model_exists": Path(GENERAL_MODEL_PATH).is_file(),
        "general_model_error": GENERAL_LOAD_ERROR,
        "general_confidence": GENERAL_CONFIDENCE,

        "helmet_model_available": HELMET_MODEL_AVAILABLE,
        "helmet_model_path": str(HELMET_LOCAL_PATH),
        "helmet_model_exists": Path(HELMET_LOCAL_PATH).is_file(),
        "helmet_model_error": HELMET_LOAD_ERROR,
        "helmet_model_used_in_pipeline": False,

        "ppe_model_available": ppe_status["ppe_model_available"],
        "ppe_model_path": ppe_status["ppe_model_path"],
        "ppe_model_exists": ppe_status["ppe_model_exists"],
        "ppe_model_error": ppe_status["ppe_load_error"],
        "ppe_classes": ppe_status["ppe_classes"],
        "ppe_capabilities": ppe_status["ppe_capabilities"],
        "ppe_confidence": ppe_status["ppe_confidence"],
        "ppe_category_confidence": ppe_status.get("ppe_category_confidence", {}),

        "ppe_match_threshold": PPE_MATCH_THRESHOLD,
        "ppe_memory_seconds": PPE_MEMORY_SECONDS,
        "ppe_regions": PPE_REGIONS,

        "face_ppe": get_face_ppe_status(),

        "absence_inference": PPE_INFER_ABSENCE,
        "absence_mode": PPE_ABSENCE_MODE,
        "color_detector_available": True,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_model_status(), indent=2, default=str))