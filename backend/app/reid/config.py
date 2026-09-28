"""Settings for the body-appearance Re-ID unique-footfall engine.

A trimmed copy of unique-footfall-export/app/config.py: only the REID_* and
PEOPLEID_* settings the vendored modules in this package actually read (see
UNIQUE_FOOTFALL.md for what each one means and why the defaults are what
they are). Kept as its own module rather than merged into app/config.py so
the vendored files' `from . import config` resolves unchanged.

The one deliberate default change vs the export: REID_MOT_INTERVAL_SECONDS
is 0.5, not 3.0. The export was tuned for a reception camera where people
linger; at an entry gate someone crosses the frame in a few seconds, and a
3s cadence would see them once at most — not enough observations to either
confirm a match (REID_CONFIRMED_MIN_VOTES) or enroll a new identity
(REID_AUTO_ENROLL_MIN_OBSERVATIONS). With a counting zone drawn over just the
doorway a walker is inside it for ~2 s, so even 1.0 left most of them with
only 2 looks — one short of being enrolled.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# --- Person detection + tracking (PersonBodyTracker) ---
PEOPLEID_MOT_MODEL_PATH = os.getenv("PEOPLEID_MOT_MODEL_PATH", str(_MODELS_DIR / "yolov8n.pt"))
PEOPLEID_MOT_IMGSZ = int(os.getenv("PEOPLEID_MOT_IMGSZ", "640"))
PEOPLEID_MOT_CONFIDENCE = float(os.getenv("PEOPLEID_MOT_CONFIDENCE", "0.4"))

# --- Enrollment curation (peopleid_enrollment.curate) ---
PEOPLEID_ENROLLMENT_TARGET_EMBEDDINGS = int(os.getenv("PEOPLEID_ENROLLMENT_TARGET_EMBEDDINGS", "15"))
PEOPLEID_DEDUP_SIMILARITY = float(os.getenv("PEOPLEID_DEDUP_SIMILARITY", "0.92"))

# --- Re-ID embedding model ---
# osnet_x0_25 trained on MSMT17 — fetch with `python -m scripts.fetch_reid_model`
# from backend/. Without it the engine falls back to the ImageNet backbone,
# which over-counts badly (see UNIQUE_FOOTFALL.md).
REID_MODEL_NAME = os.getenv("REID_MODEL_NAME", "osnet_x0_25")
_REID_DEFAULT_MODEL = _MODELS_DIR / "osnet_x0_25_msmt17.pth"
REID_MODEL_PATH = os.getenv("REID_MODEL_PATH", str(_REID_DEFAULT_MODEL) if _REID_DEFAULT_MODEL.exists() else "")
REID_DEVICE = os.getenv("REID_DEVICE", "auto")  # "auto" | "cpu" | "cuda"

# --- Matching ---
REID_SIMILARITY_THRESHOLD = float(os.getenv("REID_SIMILARITY_THRESHOLD", "0.75"))
REID_MIN_MARGIN = float(os.getenv("REID_MIN_MARGIN", "0.05"))

# --- Crop quality gate ---
REID_QUALITY_MIN_SCORE = float(os.getenv("REID_QUALITY_MIN_SCORE", "0.55"))
REID_MIN_BODY_SIZE_PX = int(os.getenv("REID_MIN_BODY_SIZE_PX", "80"))
REID_BLUR_VARIANCE_FLOOR = float(os.getenv("REID_BLUR_VARIANCE_FLOOR", "80.0"))
REID_BLUR_VARIANCE_HARD_MIN = float(os.getenv("REID_BLUR_VARIANCE_HARD_MIN", "20.0"))
# Body boxes shorter than this height:width are skipped entirely (seated,
# head-only, merged boxes) — see reid_quality.assess.
REID_MIN_ASPECT_RATIO = float(os.getenv("REID_MIN_ASPECT_RATIO", "1.5"))

# --- Temporal fusion ---
REID_FUSION_WINDOW = int(os.getenv("REID_FUSION_WINDOW", "5"))
REID_CANDIDATE_MIN_VOTES = int(os.getenv("REID_CANDIDATE_MIN_VOTES", "2"))
REID_CONFIRMED_MIN_VOTES = int(os.getenv("REID_CONFIRMED_MIN_VOTES", "3"))
REID_CONFIRMED_CONTRADICTION_LIMIT = int(os.getenv("REID_CONFIRMED_CONTRADICTION_LIMIT", "2"))
REID_TRACK_TIMEOUT_SECONDS = float(os.getenv("REID_TRACK_TIMEOUT_SECONDS", "20.0"))

# --- Auto-enrollment of new identities ---
REID_AUTO_ENROLL_MIN_OBSERVATIONS = int(os.getenv("REID_AUTO_ENROLL_MIN_OBSERVATIONS", "3"))
REID_ENROLLMENT_TARGET_EMBEDDINGS = int(os.getenv("REID_ENROLLMENT_TARGET_EMBEDDINGS", "12"))

# --- Learning new views of known people (footfall._record) ---
# A confirmed match that scored below REID_LEARN_BELOW_SIMILARITY is a view
# the gallery doesn't cover well yet (new angle, other gate), so its
# embedding is added to that person — up to REID_MAX_EMBEDDINGS_PER_PERSON.
# Measured live: one person's views of themselves fell below 0.55 one time
# in ten, so a person enrolled from one angle often failed to match from
# another and got a second identity.
REID_LEARN_BELOW_SIMILARITY = float(os.getenv("REID_LEARN_BELOW_SIMILARITY", "0.88"))
REID_MAX_EMBEDDINGS_PER_PERSON = int(os.getenv("REID_MAX_EMBEDDINGS_PER_PERSON", "30"))

# --- Cadence (see module docstring for why 0.5 here) ---
REID_MOT_INTERVAL_SECONDS = float(os.getenv("REID_MOT_INTERVAL_SECONDS", "0.5"))
