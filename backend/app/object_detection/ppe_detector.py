"""
PPE DETECTOR
============

Primary PPE model: YOLO26m PPE model.

EXPECTED YOLO26 CLASSES
-----------------------

The trained model must contain exactly these 7 semantic classes:

    goggles
    helmet
    mask
    no-goggles
    no-helmet
    no-mask
    person

The order of class IDs does NOT matter. The names are read from the
actual model and validated at startup.

INTERNAL COMPATIBILITY
----------------------

The rest of the existing application historically uses "specs" for
goggles. Therefore:

    goggles      -> category="specs", status="specs"
    no-goggles   -> category="specs", status="no specs"

This allows the existing detector / tracker / analytics code to keep
working while the frontend can display the user-facing name "Goggles".

The YOLO26 model does NOT provide:

    vest
    gloves

Those categories are therefore NOT claimed as capabilities of the
primary model.

IMPORTANT
---------

This file deliberately refuses to silently use an old construction-site
14-class best.pt as the primary model.

That prevents a very confusing situation where:

    ppe_yolo26_best.pt is missing
             ↓
    old best.pt is found
             ↓
    application starts
             ↓
    dashboard appears to work
             ↓
    goggles / mask / no-mask detections are missing

Instead, the model is validated before being accepted.
"""

from __future__ import annotations
from ._quiet import print  # noqa: A004 - diagnostics go to logging, not stdout

import os
import time
from pathlib import Path

from dotenv import load_dotenv

# Must run before any os.getenv() call below - see detector.py for why
# (this file can be imported before config.py has had a chance to load
# .env, which silently drops every PPE_* override).
load_dotenv()

from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None


# =====================================================================
# 1. PATHS
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent
# Model files live in the host app's MODEL_DIR (set by app/object_detection/service.py).
_MODELS_DIR = Path(os.environ.get("OBJECT_DETECTION_MODEL_DIR") or BASE_DIR / "models")


# The YOLO26 model MUST be preferred.
#
# The older generic best.pt is intentionally kept only as a compatibility
# candidate, but it will NOT be accepted if its class list does not match
# the expected YOLO26 PPE model.
PPE_MODEL_CANDIDATES = [
    _MODELS_DIR / "ppe_yolo26_best.pt",
    _MODELS_DIR / "ppe_best.pt",
    _MODELS_DIR / "best.pt",
    _MODELS_DIR / "ppe" / "best.pt",
    BASE_DIR.parent / "models" / "ppe" / "best.pt",
    BASE_DIR.parent / "models" / "best.pt",
    BASE_DIR / "best.pt",
]


# =====================================================================
# 2. EXPECTED YOLO26 MODEL
# =====================================================================

EXPECTED_PPE_CLASSES = {
    "goggles",
    "helmet",
    "mask",
    "no_goggles",
    "no_helmet",
    "no_mask",
    "person",
}


# User-facing / application-facing active PPE categories.
#
# "specs" is intentionally retained internally because the existing
# detector, tracker and analytics code use that field name.
ACTIVE_PPE_CATEGORIES = (
    "helmet",
    "mask",
    "specs",
)


# =====================================================================
# 3. ENVIRONMENT HELPERS
# =====================================================================

def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name, default=False):
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


# =====================================================================
# 4. DEVICE
# =====================================================================

if torch is not None and torch.cuda.is_available():
    PPE_DEVICE = "cuda"
else:
    PPE_DEVICE = "cpu"


# =====================================================================
# 5. MODEL SETTINGS
# =====================================================================

# Low inference floor allows weak detections to reach this Python layer.
# They are then filtered using the category threshold below.
PPE_INFERENCE_FLOOR = _env_float(
    "PPE_INFERENCE_FLOOR",
    0.08,
)


# ---------------------------------------------------------------------
# Per-category confidence thresholds
# ---------------------------------------------------------------------
#
# These values apply equally to positive and negative states:
#
# helmet      -> helmet / no helmet
# mask        -> mask / no mask
# specs       -> goggles / no goggles
#
# The old vest/gloves values remain only for backward compatibility
# with code that may ask for them. The YOLO26 model does not expose
# those categories.
PPE_CATEGORY_CONFIDENCE = {
    "helmet": _env_float("PPE_CONF_HELMET", 0.15),
    "mask": _env_float("PPE_CONF_MASK", 0.15),
    "specs": _env_float("PPE_CONF_SPECS", 0.15),

    # Legacy categories. Not supported by YOLO26 primary model.
    "vest": _env_float("PPE_CONF_VEST", 0.30),
    "gloves": _env_float("PPE_CONF_GLOVES", 0.30),

    # Person from the PPE model is normally ignored by detector.py
    # because the general YOLO model already detects people.
    "person": _env_float("PPE_CONF_PERSON", 0.60),

    "other": _env_float("PPE_CONF_OTHER", 0.50),
}


# Never allow a category threshold below the absolute noise floor.
PPE_ABSOLUTE_FLOOR = 0.10

for _category, _value in list(PPE_CATEGORY_CONFIDENCE.items()):
    if _value < PPE_ABSOLUTE_FLOOR:
        PPE_CATEGORY_CONFIDENCE[_category] = PPE_ABSOLUTE_FLOOR


# Backward compatibility.
PPE_CONFIDENCE = min(
    PPE_CATEGORY_CONFIDENCE[category]
    for category in ACTIVE_PPE_CATEGORIES
)


PPE_IOU = _env_float("PPE_IOU", 0.45)

PPE_IMAGE_SIZE = _env_int(
    "PPE_IMGSZ",
    640,
)

PPE_DEBUG = _env_bool(
    "PPE_DEBUG",
    True,
)

PPE_SHOW_REJECTED = _env_bool(
    "PPE_SHOW_REJECTED",
    True,
)


_DEBUG_INTERVAL_SECONDS = 1.0
_last_debug_time = 0.0


# =====================================================================
# 6. MODEL FILE SELECTION
# =====================================================================

def find_ppe_model_file():
    """
    Find the primary PPE model.

    PPE_MODEL environment variable has priority.

    Otherwise candidates are searched in order.

    NOTE:
    We do not assume that an arbitrary best.pt is the YOLO26 model.
    Class validation happens after loading.
    """

    from_env = os.getenv("PPE_MODEL")

    if from_env:
        candidate = Path(from_env)

        if candidate.is_file():
            return candidate

        print(
            f"[PPE] PPE_MODEL points to a missing file: {candidate}"
        )

        return None

    for candidate in PPE_MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate

    return None


# =====================================================================
# 7. CLASS NAME NORMALIZATION
# =====================================================================

def normalize_ppe_class(class_name):
    """
    Normalize a YOLO class name.

    Examples:

        Goggles       -> goggles
        NO-Goggles    -> no_goggles
        NO_Goggles    -> no_goggles
        No Goggles    -> no_goggles
        Hard Hat      -> hard_hat
        Safety Vest   -> safety_vest
    """

    name = str(class_name).strip().lower()

    name = name.replace("-", "_")
    name = name.replace(" ", "_")

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


# =====================================================================
# 8. SEMANTIC CLASSIFICATION
# =====================================================================

CATEGORY_KEYWORDS = {
    "helmet": (
        "hardhat",
        "hard_hat",
        "helmet",
    ),

    "mask": (
        "mask",
        "face_mask",
        "facemask",
        "medical_mask",
    ),

    "specs": (
        "goggles",
        "goggle",
        "specs",
        "spectacles",
        "glasses",
        "eyewear",
    ),

    "vest": (
        "safety_vest",
        "vest",
        "hi_vis",
        "reflective_vest",
    ),

    "gloves": (
        "gloves",
        "glove",
    ),

    "person": (
        "person",
        "people",
        "worker",
        "pedestrian",
    ),
}


NEGATIVE_PREFIXES = (
    "no_",
    "non_",
    "not_",
    "without_",
)


def _classify_class_name(normalized_name):
    """
    Return:

        category
        is_negative

    Examples:

        no_helmet  -> helmet, True
        helmet     -> helmet, False
        goggles    -> specs, False
        no_goggles -> specs, True
        mask       -> mask, False
        no_mask    -> mask, True
        person     -> person, False
    """

    is_negative = False
    base = normalized_name

    for prefix in NEGATIVE_PREFIXES:
        if base.startswith(prefix):
            is_negative = True
            base = base[len(prefix):]
            break

    # Most specific/important categories first.
    #
    # Goggles are internally called "specs".
    for category, keywords in CATEGORY_KEYWORDS.items():

        for keyword in keywords:

            if keyword in base:
                return category, is_negative

    return "other", is_negative


# =====================================================================
# 9. CLASS MAP
# =====================================================================

def build_class_map(names):
    """
    Build:

        {
            class_id: {
                raw,
                name,
                category,
                negative,
                status,
                threshold
            }
        }

    from the actual YOLO model names.
    """

    class_map = {}

    if not names:
        return class_map

    if isinstance(names, (list, tuple)):
        names = {
            index: value
            for index, value in enumerate(names)
        }

    for class_id, raw_name in names.items():

        try:
            class_id = int(class_id)
        except (TypeError, ValueError):
            continue

        normalized = normalize_ppe_class(raw_name)

        category, negative = _classify_class_name(
            normalized
        )

        if category == "other":

            status = normalized.replace(
                "_",
                " ",
            )

        elif negative:

            status = f"no {category}"

        else:

            status = category

        class_map[class_id] = {
            "raw": str(raw_name),
            "name": normalized,
            "category": category,
            "negative": negative,
            "status": status,
            "threshold": PPE_CATEGORY_CONFIDENCE.get(
                category,
                0.35,
            ),
        }

    return class_map


# =====================================================================
# 10. STRICT YOLO26 CLASS VALIDATION
# =====================================================================

def validate_yolo26_class_set(class_map):
    """
    Verify that the loaded model is the intended YOLO26 PPE model.

    Expected semantic set:

        goggles
        helmet
        mask
        no-goggles
        no-helmet
        no-mask
        person

    The numeric class IDs do not matter.

    Returns:

        True / False
    """

    actual = {
        info["name"]
        for info in class_map.values()
    }

    missing = EXPECTED_PPE_CLASSES - actual
    unexpected = actual - EXPECTED_PPE_CLASSES

    if not missing and not unexpected:
        return True

    print("=" * 70)
    print("[PPE] WRONG PPE MODEL CLASS SET")
    print("=" * 70)

    print("[PPE] Expected YOLO26 classes:")
    for name in sorted(EXPECTED_PPE_CLASSES):
        print(f"[PPE]   {name}")

    print()

    print("[PPE] Actual model classes:")
    for name in sorted(actual):
        print(f"[PPE]   {name}")

    if missing:
        print()
        print("[PPE] Missing expected classes:")
        for name in sorted(missing):
            print(f"[PPE]   {name}")

    if unexpected:
        print()
        print("[PPE] Unexpected classes:")
        for name in sorted(unexpected):
            print(f"[PPE]   {name}")

    print()
    print(
        "[PPE] This model will NOT be used as the primary YOLO26 "
        "PPE model."
    )

    print("=" * 70)

    return False


# =====================================================================
# 11. CAPABILITIES
# =====================================================================

def build_capabilities(class_map):
    """
    Report what the currently loaded model can actually detect.
    """

    capabilities = {}

    for category in (
        "helmet",
        "mask",
        "specs",
        "vest",
        "gloves",
        "person",
    ):
        capabilities[category] = {
            "positive": False,
            "negative": False,
        }

    for info in class_map.values():

        category = info["category"]

        if category not in capabilities:
            continue

        if info["negative"]:
            capabilities[category]["negative"] = True
        else:
            capabilities[category]["positive"] = True

    return capabilities


# =====================================================================
# 12. GLOBAL MODEL STATE
# =====================================================================

ppe_model = None

PPE_MODEL_AVAILABLE = False

PPE_MODEL_PATH = (
    find_ppe_model_file()
    or PPE_MODEL_CANDIDATES[0]
)

PPE_CLASS_MAP = {}

PPE_CAPABILITIES = build_capabilities({})

PPE_LOAD_ERROR = None


# =====================================================================
# 13. LOAD PRIMARY PPE MODEL
# =====================================================================

def load_ppe_model():
    """
    Load and validate the primary YOLO26 PPE model.

    A wrong model is rejected instead of silently becoming the active
    detector.
    """

    global ppe_model
    global PPE_MODEL_AVAILABLE
    global PPE_MODEL_PATH
    global PPE_CLASS_MAP
    global PPE_CAPABILITIES
    global PPE_LOAD_ERROR

    ppe_model = None
    PPE_MODEL_AVAILABLE = False
    PPE_CLASS_MAP = {}
    PPE_CAPABILITIES = build_capabilities({})
    PPE_LOAD_ERROR = None

    model_file = find_ppe_model_file()

    if model_file is None:

        PPE_LOAD_ERROR = (
            "No PPE model file found."
        )

        print("=" * 70)
        print("[PPE] YOLO26 PPE MODEL NOT FOUND")
        print("=" * 70)

        print("[PPE] Expected primary file:")
        print(
            f"[PPE]   {BASE_DIR / 'models' / 'ppe_yolo26_best.pt'}"
        )

        print()
        print("[PPE] Searched:")
        for candidate in PPE_MODEL_CANDIDATES:
            print(f"[PPE]   {candidate}")

        print()
        print("[PPE] PPE detection is DISABLED.")
        print("=" * 70)

        return False

    PPE_MODEL_PATH = model_file

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    try:

        model = YOLO(
            str(model_file)
        )

    except Exception as exc:

        PPE_LOAD_ERROR = (
            f"{type(exc).__name__}: {exc}"
        )

        print("=" * 70)
        print("[PPE] MODEL FAILED TO LOAD")
        print("=" * 70)
        print(f"[PPE] File   : {model_file}")
        print(f"[PPE] Reason : {PPE_LOAD_ERROR}")

        message = str(exc).lower()

        looks_like_version_problem = any(
            hint in message
            for hint in (
                "unsupported",
                "unknown model",
                "no module named",
                "attribute",
                "cannot import",
                "not supported",
                "unexpected key",
                "yolo26",
                "v26",
            )
        )

        if looks_like_version_problem:

            try:
                import ultralytics

                installed_version = getattr(
                    ultralytics,
                    "__version__",
                    "unknown",
                )

            except Exception:
                installed_version = "unknown"

            print()
            print(
                "[PPE] This may be an Ultralytics version "
                "compatibility problem."
            )

            print(
                f"[PPE] Installed ultralytics: "
                f"{installed_version}"
            )

            print()
            print(
                "[PPE] Upgrade Ultralytics in the same Python "
                "environment used by FastAPI."
            )

            print(
                "[PPE] Example:"
            )

            print(
                "[PPE]   python -m pip install -U ultralytics"
            )

        print()
        print("[PPE] PPE detection is DISABLED.")
        print("=" * 70)

        return False

    # ---------------------------------------------------------------
    # Read model names BEFORE accepting the model
    # ---------------------------------------------------------------

    model_names = getattr(
        model,
        "names",
        {},
    )

    class_map = build_class_map(
        model_names
    )

    # ---------------------------------------------------------------
    # STRICT validation
    # ---------------------------------------------------------------

    if not validate_yolo26_class_set(
        class_map
    ):

        PPE_LOAD_ERROR = (
            "Loaded model does not contain the expected "
            "YOLO26 7-class PPE set."
        )

        print(
            f"[PPE] Refusing to activate: {model_file}"
        )

        return False

    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    try:

        model.to(
            PPE_DEVICE
        )

    except Exception as exc:

        print(
            f"[PPE] Could not move model to "
            f"{PPE_DEVICE}: {exc}"
        )

        print(
            "[PPE] Continuing; Ultralytics will use "
            "its available device."
        )

    # ---------------------------------------------------------------
    # Accept model
    # ---------------------------------------------------------------

    ppe_model = model

    PPE_MODEL_AVAILABLE = True

    PPE_CLASS_MAP = class_map

    PPE_CAPABILITIES = build_capabilities(
        PPE_CLASS_MAP
    )

    PPE_LOAD_ERROR = None

    print_ppe_report()

    return True


# =====================================================================
# 14. STARTUP REPORT
# =====================================================================

def print_ppe_report():
    """
    Print the actual loaded model information.
    """

    print("=" * 70)
    print("[PPE] YOLO26 PPE MODEL LOADED")
    print("=" * 70)

    print(
        f"[PPE] File           : {PPE_MODEL_PATH}"
    )

    print(
        f"[PPE] Device         : {PPE_DEVICE}"
    )

    print(
        f"[PPE] Image size     : {PPE_IMAGE_SIZE}"
    )

    print(
        f"[PPE] Inference floor: "
        f"{PPE_INFERENCE_FLOOR:.2f}"
    )

    print(
        f"[PPE] Classes        : "
        f"{len(PPE_CLASS_MAP)}"
    )

    print()

    print(
        "[PPE] ID   MODEL CLASS          -> CATEGORY   STATUS"
    )

    print(
        "[PPE] ---  -------------------     ---------  ------------"
    )

    for class_id in sorted(PPE_CLASS_MAP):

        info = PPE_CLASS_MAP[class_id]

        print(
            f"[PPE] {class_id:>3}  "
            f"{info['raw']:<19}  -> "
            f"{info['category']:<9}  "
            f"{info['status']:<12}"
        )

    print()

    print(
        "[PPE] Active dashboard PPE:"
    )

    print(
        "[PPE]   Helmet"
    )

    print(
        "[PPE]   Mask"
    )

    print(
        "[PPE]   Goggles "
        "(internal field: specs)"
    )

    print()

    print(
        "[PPE] Model capabilities:"
    )

    for category, flags in PPE_CAPABILITIES.items():

        positive = (
            "YES"
            if flags["positive"]
            else "NO"
        )

        negative = (
            "YES"
            if flags["negative"]
            else "NO"
        )

        print(
            f"[PPE]   {category:<7} "
            f"positive={positive} "
            f"negative={negative}"
        )

    print()

    print(
        "[PPE] Confidence thresholds:"
    )

    for category in (
        "helmet",
        "mask",
        "specs",
    ):

        print(
            f"[PPE]   {category:<8} "
            f"{PPE_CATEGORY_CONFIDENCE[category]:.2f}"
        )

    print()

    print(
        "[PPE] Strict YOLO26 class validation: PASS"
    )

    print("=" * 70)


# =====================================================================
# 15. LOAD MODEL AT IMPORT
# =====================================================================

load_ppe_model()


# =====================================================================
# 16. BASIC HELPERS
# =====================================================================

def get_class_name(class_id):
    """
    Return normalized model class name.
    """

    try:
        class_id = int(class_id)
    except (TypeError, ValueError):
        return "unknown"

    info = PPE_CLASS_MAP.get(
        class_id
    )

    if info:
        return info["name"]

    return "unknown"


def get_category_threshold(category):
    """
    Return the confidence threshold for a semantic category.
    """

    return PPE_CATEGORY_CONFIDENCE.get(
        category,
        0.35,
    )


def clamp_bbox(
    frame,
    x1,
    y1,
    x2,
    y2,
):
    """
    Keep bounding box inside frame.
    """

    if frame is None:
        return 0, 0, 0, 0

    height, width = frame.shape[:2]

    x1 = max(
        0,
        min(
            int(x1),
            width - 1,
        ),
    )

    y1 = max(
        0,
        min(
            int(y1),
            height - 1,
        ),
    )

    x2 = max(
        0,
        min(
            int(x2),
            width - 1,
        ),
    )

    y2 = max(
        0,
        min(
            int(y2),
            height - 1,
        ),
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# =====================================================================
# 17. PRIMARY PPE DETECTION
# =====================================================================

def run_ppe_detection(
    frame,
    conf=None,
    imgsz=None,
    return_rejected=False,
):
    """
    Run the primary YOLO26 PPE model once.

    Parameters
    ----------
    frame:
        OpenCV BGR image.

    conf:
        None:
            use category-specific thresholds.

        number:
            use one flat threshold for all classes.

    imgsz:
        Optional YOLO inference image size.

    return_rejected:
        If True:

            return kept, rejected

        Otherwise:

            return kept

    Detection structure:

        {
            "class_id": 1,
            "class_name": "helmet",
            "original_class_name": "helmet",
            "category": "helmet",
            "status": "helmet",
            "is_negative": False,
            "confidence": 0.91,
            "threshold": 0.15,
            "x1": 100,
            "y1": 50,
            "x2": 300,
            "y2": 200,
            "bbox": [100, 50, 300, 200],
        }

    Negative examples:

        no-helmet
            category = "helmet"
            status = "no helmet"
            is_negative = True

        no-mask
            category = "mask"
            status = "no mask"
            is_negative = True

        no-goggles
            category = "specs"
            status = "no specs"
            is_negative = True
    """

    global _last_debug_time

    kept = []
    rejected = []

    if (
        frame is None
        or ppe_model is None
        or not PPE_MODEL_AVAILABLE
    ):

        return (
            (kept, rejected)
            if return_rejected
            else kept
        )

    # ---------------------------------------------------------------
    # Flat threshold compatibility
    # ---------------------------------------------------------------

    flat_threshold = None

    if conf is not None:

        try:
            flat_threshold = float(conf)
        except (
            TypeError,
            ValueError,
        ):

            flat_threshold = None

    # ---------------------------------------------------------------
    # YOLO inference threshold
    # ---------------------------------------------------------------

    if flat_threshold is not None:

        inference_conf = min(
            PPE_INFERENCE_FLOOR,
            flat_threshold,
        )

    else:

        inference_conf = PPE_INFERENCE_FLOOR

    image_size = (
        PPE_IMAGE_SIZE
        if imgsz is None
        else int(imgsz)
    )

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------

    try:

        results = ppe_model(
            frame,
            conf=inference_conf,
            iou=PPE_IOU,
            imgsz=image_size,
            device=PPE_DEVICE,
            verbose=False,
        )

    except Exception as exc:

        print(
            f"[PPE] ERROR during inference: "
            f"{type(exc).__name__}: {exc}"
        )

        return (
            (kept, rejected)
            if return_rejected
            else kept
        )

    # ---------------------------------------------------------------
    # Parse results
    # ---------------------------------------------------------------

    for result in results:

        boxes = getattr(
            result,
            "boxes",
            None,
        )

        if boxes is None:
            continue

        for box in boxes:

            try:

                confidence = float(
                    box.conf[0]
                )

                class_id = int(
                    box.cls[0]
                )

                coordinates = (
                    box.xyxy[0]
                    .tolist()
                )

                x1, y1, x2, y2 = (
                    int(value)
                    for value in coordinates
                )

                (
                    x1,
                    y1,
                    x2,
                    y2,
                ) = clamp_bbox(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                )

                if (
                    x2 <= x1
                    or y2 <= y1
                ):
                    continue

                info = PPE_CLASS_MAP.get(
                    class_id
                )

                # This should never happen with a validated model,
                # but keep the parser defensive.
                if info is None:

                    info = {
                        "raw": f"class_{class_id}",
                        "name": f"class_{class_id}",
                        "category": "other",
                        "negative": False,
                        "status": f"class_{class_id}",
                        "threshold": PPE_CATEGORY_CONFIDENCE[
                            "other"
                        ],
                    }

                # ---------------------------------------------------
                # Threshold
                # ---------------------------------------------------

                if flat_threshold is not None:

                    threshold = flat_threshold

                else:

                    threshold = get_category_threshold(
                        info["category"]
                    )

                # ---------------------------------------------------
                # Detection object
                # ---------------------------------------------------

                detection = {
                    "class_id": class_id,

                    # Normalized class name.
                    "class_name": info["name"],

                    # Exact class name stored in model.
                    "original_class_name": info["raw"],

                    # Application semantic category.
                    #
                    # goggles -> specs
                    # no-goggles -> specs
                    "category": info["category"],

                    # helmet / no helmet
                    # mask / no mask
                    # specs / no specs
                    "status": info["status"],

                    "is_negative": info["negative"],

                    "confidence": round(
                        confidence,
                        3,
                    ),

                    "threshold": round(
                        threshold,
                        3,
                    ),

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,

                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                }

                # ---------------------------------------------------
                # Keep / reject
                # ---------------------------------------------------

                if confidence >= threshold:

                    kept.append(
                        detection
                    )

                else:

                    rejected.append(
                        detection
                    )

            except Exception as exc:

                print(
                    "[PPE] Skipped malformed "
                    f"box: {type(exc).__name__}: {exc}"
                )

    # =================================================================
    # DEBUG
    # =================================================================

    if PPE_DEBUG:

        now = time.time()

        if (
            now - _last_debug_time
            >= _DEBUG_INTERVAL_SECONDS
        ):

            _last_debug_time = now

            summary = {}

            for detection in kept:

                status = detection[
                    "status"
                ]

                summary[status] = (
                    summary.get(
                        status,
                        0,
                    )
                    + 1
                )

            print(
                f"[PPE] kept={len(kept)} "
                f"{summary}"
            )

            if (
                PPE_SHOW_REJECTED
                and rejected
            ):

                near = ", ".join(
                    (
                        f"{item['original_class_name']} "
                        f"{item['confidence']:.2f}"
                        f"(needs "
                        f"{item['threshold']:.2f})"
                    )
                    for item in sorted(
                        rejected,
                        key=lambda item: (
                            -item["confidence"]
                        ),
                    )[:5]
                )

                print(
                    f"[PPE] rejected="
                    f"{len(rejected)}: "
                    f"{near}"
                )

    if return_rejected:

        return (
            kept,
            rejected,
        )

    return kept


# =====================================================================
# 18. MODEL STATUS
# =====================================================================

def get_ppe_model_status():
    """
    Return the actual primary model status.

    This is deliberately explicit so the frontend/backend diagnostics
    can tell whether the correct YOLO26 model is loaded.
    """

    classes = {
        class_id: info["raw"]
        for class_id, info
        in PPE_CLASS_MAP.items()
    }

    return {
        "ppe_model_available": (
            PPE_MODEL_AVAILABLE
        ),

        "ppe_model_path": str(
            PPE_MODEL_PATH
        ),

        "ppe_model_exists": Path(
            PPE_MODEL_PATH
        ).is_file(),

        "ppe_load_error": (
            PPE_LOAD_ERROR
        ),

        "ppe_expected_classes": sorted(
            EXPECTED_PPE_CLASSES
        ),

        "ppe_classes": classes,

        "ppe_class_map": (
            PPE_CLASS_MAP
        ),

        "ppe_capabilities": (
            PPE_CAPABILITIES
        ),

        "ppe_active_categories": list(
            ACTIVE_PPE_CATEGORIES
        ),

        "ppe_confidence": (
            PPE_CONFIDENCE
        ),

        "ppe_category_confidence": dict(
            PPE_CATEGORY_CONFIDENCE
        ),

        "ppe_inference_floor": (
            PPE_INFERENCE_FLOOR
        ),

        "ppe_device": (
            PPE_DEVICE
        ),

        "ppe_model_validated": (
            PPE_MODEL_AVAILABLE
        ),
    }


# =====================================================================
# 19. OPTIONAL FACE PPE MODEL
# =====================================================================
#
# This remains optional.
#
# The primary YOLO26 model already contains:
#
#     goggles
#     mask
#     no-goggles
#     no-mask
#
# Therefore this second model is NOT required.
#
# If a separate face model exists, it can still be used by detector.py
# if that existing code calls run_face_ppe_detection().
#
# It is restricted to:
#
#     mask
#     specs/goggles
#
# and does not replace the primary YOLO26 model.
# =====================================================================

FACE_PPE_MODEL_CANDIDATES = [
    _MODELS_DIR / "face_ppe.pt",
    _MODELS_DIR / "mask_best.pt",
    _MODELS_DIR / "face_mask.pt",
    _MODELS_DIR / "mask.pt",
]


FACE_PPE_CATEGORIES = (
    "mask",
    "specs",
)


FACE_PPE_CONFIDENCE = _env_float(
    "FACE_PPE_CONFIDENCE",
    0.25,
)


face_ppe_model = None

FACE_PPE_MODEL_AVAILABLE = False

FACE_PPE_MODEL_PATH = None

FACE_PPE_CLASS_MAP = {}

FACE_PPE_LOAD_ERROR = None


# =====================================================================
# 20. FACE MODEL PATH
# =====================================================================

def find_face_ppe_model_file():
    """
    Find optional second face PPE model.
    """

    from_env = os.getenv(
        "FACE_PPE_MODEL"
    )

    if from_env:

        candidate = Path(
            from_env
        )

        if candidate.is_file():
            return candidate

        print(
            "[FACE-PPE] FACE_PPE_MODEL "
            f"is missing: {candidate}"
        )

    for candidate in FACE_PPE_MODEL_CANDIDATES:

        if candidate.is_file():
            return candidate

    return None


# =====================================================================
# 21. LOAD OPTIONAL FACE MODEL
# =====================================================================

def load_face_ppe_model():
    """
    Load optional face PPE model.

    Absence of this model is NOT an error.
    """

    global face_ppe_model
    global FACE_PPE_MODEL_AVAILABLE
    global FACE_PPE_MODEL_PATH
    global FACE_PPE_CLASS_MAP
    global FACE_PPE_LOAD_ERROR

    face_ppe_model = None

    FACE_PPE_MODEL_AVAILABLE = False

    FACE_PPE_MODEL_PATH = None

    FACE_PPE_CLASS_MAP = {}

    FACE_PPE_LOAD_ERROR = None

    model_file = (
        find_face_ppe_model_file()
    )

    if model_file is None:

        print(
            "[FACE-PPE] Optional second "
            "face model not installed."
        )

        return False

    FACE_PPE_MODEL_PATH = (
        model_file
    )

    try:

        model = YOLO(
            str(model_file)
        )

    except Exception as exc:

        FACE_PPE_LOAD_ERROR = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "[FACE-PPE] Failed to load "
            f"{model_file}: "
            f"{FACE_PPE_LOAD_ERROR}"
        )

        return False

    try:

        model.to(
            PPE_DEVICE
        )

    except Exception as exc:
        print(f"[FACE-PPE] could not move model to {PPE_DEVICE}, staying on CPU: {exc}")

    face_ppe_model = model

    FACE_PPE_MODEL_AVAILABLE = True

    FACE_PPE_CLASS_MAP = build_class_map(
        getattr(
            model,
            "names",
            {},
        )
    )

    print("=" * 70)
    print("[FACE-PPE] OPTIONAL FACE MODEL LOADED")
    print("=" * 70)

    print(
        f"[FACE-PPE] File       : {model_file}"
    )

    print(
        f"[FACE-PPE] Confidence : "
        f"{FACE_PPE_CONFIDENCE:.2f}"
    )

    print(
        f"[FACE-PPE] Classes    : "
        f"{len(FACE_PPE_CLASS_MAP)}"
    )

    for class_id in sorted(
        FACE_PPE_CLASS_MAP
    ):

        info = FACE_PPE_CLASS_MAP[
            class_id
        ]

        used = (
            "USED"
            if info["category"]
            in FACE_PPE_CATEGORIES
            else "IGNORED"
        )

        print(
            f"[FACE-PPE] "
            f"{class_id:>3} "
            f"{info['raw']:<20} "
            f"-> "
            f"{info['status']:<12} "
            f"{used}"
        )

    print("=" * 70)

    return True


load_face_ppe_model()


# =====================================================================
# 22. OPTIONAL FACE MODEL INFERENCE
# =====================================================================

def run_face_ppe_detection(
    frame,
    conf=None,
    imgsz=None,
):
    """
    Run optional face PPE model.

    Returns an empty list if no second model is installed.

    Only mask/specs categories are accepted.
    """

    detections = []

    if (
        frame is None
        or face_ppe_model is None
        or not FACE_PPE_MODEL_AVAILABLE
    ):

        return detections

    threshold = (
        FACE_PPE_CONFIDENCE
        if conf is None
        else float(conf)
    )

    image_size = (
        PPE_IMAGE_SIZE
        if imgsz is None
        else int(imgsz)
    )

    try:

        results = face_ppe_model(
            frame,
            conf=threshold,
            iou=PPE_IOU,
            imgsz=image_size,
            device=PPE_DEVICE,
            verbose=False,
        )

    except Exception as exc:

        print(
            "[FACE-PPE] inference error: "
            f"{type(exc).__name__}: {exc}"
        )

        return detections

    for result in results:

        boxes = getattr(
            result,
            "boxes",
            None,
        )

        if boxes is None:
            continue

        for box in boxes:

            try:

                confidence = float(
                    box.conf[0]
                )

                if confidence < threshold:
                    continue

                class_id = int(
                    box.cls[0]
                )

                info = FACE_PPE_CLASS_MAP.get(
                    class_id
                )

                if info is None:
                    continue

                # Only use face PPE categories.
                if (
                    info["category"]
                    not in FACE_PPE_CATEGORIES
                ):
                    continue

                (
                    x1,
                    y1,
                    x2,
                    y2,
                ) = (
                    int(value)
                    for value in box.xyxy[
                        0
                    ].tolist()
                )

                (
                    x1,
                    y1,
                    x2,
                    y2,
                ) = clamp_bbox(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                )

                if (
                    x2 <= x1
                    or y2 <= y1
                ):
                    continue

                detections.append(
                    {
                        "class_id": class_id,

                        "class_name": info[
                            "name"
                        ],

                        "original_class_name": info[
                            "raw"
                        ],

                        "category": info[
                            "category"
                        ],

                        "status": info[
                            "status"
                        ],

                        "is_negative": info[
                            "negative"
                        ],

                        "confidence": round(
                            confidence,
                            3,
                        ),

                        "threshold": round(
                            threshold,
                            3,
                        ),

                        "from_face_model": True,

                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,

                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                    }
                )

            except Exception as exc:

                print(
                    "[FACE-PPE] bad box: "
                    f"{type(exc).__name__}: {exc}"
                )

    return detections


# =====================================================================
# 23. OPTIONAL FACE MODEL STATUS
# =====================================================================

def get_face_ppe_status():
    """
    Report optional face model state.
    """

    return {
        "face_model_available": (
            FACE_PPE_MODEL_AVAILABLE
        ),

        "face_model_path": str(
            FACE_PPE_MODEL_PATH
            or ""
        ),

        "face_model_error": (
            FACE_PPE_LOAD_ERROR
        ),

        "face_model_classes": {
            class_id: info["raw"]
            for class_id, info
            in FACE_PPE_CLASS_MAP.items()
        },

        "face_model_categories": list(
            FACE_PPE_CATEGORIES
        ),

        "face_model_confidence": (
            FACE_PPE_CONFIDENCE
        ),

        "searched_paths": [
            str(path)
            for path
            in FACE_PPE_MODEL_CANDIDATES
        ],
    }


# =====================================================================
# 24. SELF TEST
# =====================================================================
#
# Run:
#
#     python ppe_detector.py
#
# Or:
#
#     python ppe_detector.py photo.jpg
#
# =====================================================================

if __name__ == "__main__":

    import sys

    print()
    print("=" * 70)
    print("YOLO26 PPE DETECTOR SELF TEST")
    print("=" * 70)
    print()

    if not PPE_MODEL_AVAILABLE:

        print(
            "RESULT: FAILED"
        )

        print()
        print(
            "The primary YOLO26 PPE model was not "
            "loaded."
        )

        if PPE_LOAD_ERROR:
            print()
            print(
                f"Reason: {PPE_LOAD_ERROR}"
            )

        raise SystemExit(1)

    print(
        "RESULT: YOLO26 PPE MODEL LOADED"
    )

    print()

    print(
        f"Model: {PPE_MODEL_PATH}"
    )

    print(
        f"Device: {PPE_DEVICE}"
    )

    print()

    print(
        "Verified classes:"
    )

    for class_id in sorted(
        PPE_CLASS_MAP
    ):

        info = PPE_CLASS_MAP[
            class_id
        ]

        print(
            f"  {class_id}: "
            f"{info['raw']} "
            f"-> "
            f"{info['category']} "
            f"-> "
            f"{info['status']}"
        )

    print()

    print(
        "Active thresholds:"
    )

    for category in (
        "helmet",
        "mask",
        "specs",
    ):

        print(
            f"  {category:<8} "
            f"{PPE_CATEGORY_CONFIDENCE[category]:.2f}"
        )

    print()

    if len(sys.argv) <= 1:

        print(
            "No image supplied."
        )

        print()
        print(
            "To test an image:"
        )

        print(
            "    python ppe_detector.py "
            "path\\to\\photo.jpg"
        )

        print()

        raise SystemExit(0)

    # ---------------------------------------------------------------
    # Image test
    # ---------------------------------------------------------------

    import cv2

    image_path = sys.argv[1]

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"Could not open image: "
            f"{image_path}"
        )

        raise SystemExit(1)

    print()

    print(
        f"Running detection on: "
        f"{image_path}"
    )

    print()

    found, rejected = run_ppe_detection(
        image,
        return_rejected=True,
    )

    # ---------------------------------------------------------------
    # Kept
    # ---------------------------------------------------------------

    print(
        f"KEPT DETECTIONS "
        f"({len(found)}):"
    )

    if not found:

        print(
            "  none"
        )

    else:

        for item in found:

            print(
                f"  "
                f"{item['status']:<12} "
                f"conf={item['confidence']:.2f} "
                f"threshold={item['threshold']:.2f} "
                f"box={item['bbox']}"
            )

    # ---------------------------------------------------------------
    # Rejected
    # ---------------------------------------------------------------

    print()

    print(
        f"REJECTED BELOW THRESHOLD "
        f"({len(rejected)}):"
    )

    if not rejected:

        print(
            "  none"
        )

    else:

        for item in sorted(
            rejected,
            key=lambda item: (
                -item["confidence"]
            ),
        )[:20]:

            print(
                f"  "
                f"{item['original_class_name']:<18} "
                f"conf={item['confidence']:.2f} "
                f"needed={item['threshold']:.2f}"
            )

    print()

    print(
        "SELF TEST COMPLETE"
    )