"""
face_pipeline.py

Per-camera worker: detect -> track -> pick best frame per track -> align ->
embed (ArcFace) -> match against gallery -> auto-tag or push to review queue.

Design choices, and why:
  - YOLOv8-face for detection: fast, GPU-friendly, good on RTSP streams.
  - ByteTrack for tracking: gives each face a stable track_id across frames
    so we only run recognition ONCE per track (on the best frame) instead of
    on every frame. This is the main latency fix vs. a naive per-frame loop.
    Uses the `supervision` package's ByteTrack implementation rather than
    the original `yolox` package — `yolox` ships a native extension that
    requires cmake/a C++ toolchain to build and failed to install on this
    box; `supervision` is a pure-Python/numpy port of the same algorithm
    with no native build step, so it's the safer dependency here.
  - InsightFace (buffalo_l, ArcFace) for embeddings: we do NOT train/retrain
    this model. It's a fixed embedding extractor. "Training" = appending
    vectors to a person's bucket in the gallery (see face_db.py). This also
    means alignment matters a lot — buffalo_l's detector gives 5-point
    landmarks used for alignment before embedding, which was almost
    certainly missing from the old broken pipeline (raw bbox crops fed
    straight into ArcFace produce noisy embeddings -> everything reads as
    "unknown").

Install:
    pip install ultralytics insightface onnxruntime supervision opencv-python
    # swap onnxruntime for onnxruntime-gpu if this machine actually has CUDA

Model files:
    - YOLO face weights: any yolov8n-face / yolov11n-face .pt (fine-tuned for
      faces, not the generic COCO yolov8n.pt — that only detects "person").
      NOT bundled here and not auto-downloaded — see FACE_RECOGNITION_WIRING.md
      for why this is a manual step.
    - InsightFace buffalo_l downloads automatically on first run to
      ~/.insightface/models/buffalo_l.
"""

import logging
import os
import time
import uuid
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO
import insightface

from app import face_db
from app import employee_directory

log = logging.getLogger("face_pipeline")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Resolved relative to this file (matches camera_db.py / face_db.py's
# DB_PATH convention), NOT the process's cwd — uvicorn for this project is
# always launched from inside backend/ (see BACKEND_HANDOFF.md), so a plain
# "backend/data/..." relative default silently doubled up into a
# backend/backend/data/... tree. Env var overrides still work as absolute
# paths if you need to point somewhere else entirely.
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Stable, version-controllable location for model checkpoints — separate
# from data/ (runtime state: captures, the DB, the trained classifier).
# Same cwd-independence reasoning as _DATA_DIR above. The .pt file itself is
# NOT committed to git or fetched automatically (see FACE_RECOGNITION_WIRING.md)
# — it's a deliberate, one-time manual step.
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

YOLO_FACE_WEIGHTS = os.environ.get("YOLO_FACE_WEIGHTS", str(_MODELS_DIR / "yolov8n-face.pt"))
# Standard Ultralytics COCO-pretrained checkpoint (class 0 = "person") — the
# generic detector the face-tuned one above is explicitly NOT used as (see
# module docstring). Auto-downloaded by ultralytics on first load if not
# already sitting in models/, same as any other stock YOLOv8 weight name.
YOLO_PERSON_WEIGHTS = os.environ.get("YOLO_PERSON_WEIGHTS", str(_MODELS_DIR / "yolov8n.pt"))
SAMPLE_FPS = float(os.environ.get("FACE_PIPELINE_FPS", "3"))  # frames/sec to run detection on
MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.45"))  # cosine sim, tune per data
TRACK_MAX_AGE = 30  # frames a track can go unseen before it's dropped

# Fraction of the YOLO box's width/height added on each side before cropping
# for InsightFace. Confirmed via scripts/verify_face_detector.py: YOLO found
# a real face at 0.860 confidence, but FaceAnalysis.get() on that exact
# bbox-tight crop found none — InsightFace's own detector (SCRFD, run again
# inside _embed() to get real 5-point landmarks for alignment) needs some
# surrounding context and reliably fails when the face fills the entire
# frame with zero margin. This does not change what YOLO detected or how
# tracking/quality-scoring works — only how much surrounding image ships to
# the alignment step that was already there.
FACE_CROP_MARGIN = float(os.environ.get("FACE_CROP_MARGIN", "0.4"))
CAPTURE_DIR = os.environ.get("FACE_CAPTURE_DIR", str(_DATA_DIR / "face_captures"))
REVIEW_DEDUPE_SECONDS = 60  # don't re-queue the same track more than once per minute

# Bulk dataset collection for manual labeling + classifier training — every
# finished track lands here regardless of match confidence, unlike
# face_pending above (low-confidence only). Labeled captures get moved into
# <employee_id>/ subfolders by face_training_routes.py; this is just the
# unlabeled staging area.
TRAINING_CAPTURE_DIR = os.environ.get("FACE_TRAINING_DIR", str(_DATA_DIR / "face_training"))
TRAINING_UNLABELED_DIR = os.path.join(TRAINING_CAPTURE_DIR, "_unlabeled")

# --- Multi-day collection controls (see FACE_TRAINING.md "7-day collection
# sessions") — unchanged from the shorter-run design except where noted. ---

# Hard ceiling across ALL cameras combined, ALL statuses combined — enforced
# atomically in face_db.py (capture_limit_lock + count_training_captures())
# so concurrent camera threads can never overshoot it. Deliberately left at
# 15000 for a 7-day run too, per explicit instruction not to just raise this
# — a week-long dataset is meant to stay useful/diverse within this budget
# (see the dedup and per-camera controls below), not grow unbounded with run
# length.
MAX_TRAINING_CAPTURES = int(os.environ.get("MAX_TRAINING_CAPTURES", "15000"))

# Per-camera share of that same budget — stops one busy camera (e.g. a main
# entrance) from consuming the entire 15000-capture allowance before quieter
# cameras contribute anything. Default: 1/3 of the global limit, so no
# single camera can dominate even if only 2-3 cameras are actually active,
# while still leaving room for one camera to run alone if that's all that's
# configured. Enforced the same atomic way as MAX_TRAINING_CAPTURES.
MAX_CAPTURES_PER_CAMERA = int(os.environ.get("FACE_MAX_CAPTURES_PER_CAMERA", str(MAX_TRAINING_CAPTURES // 3)))

# Cross-track duplicate protection: if a new capture's embedding is at least
# DEDUP_SIMILARITY_THRESHOLD cosine-similar to one captured on the SAME
# camera within the last DEDUP_COOLDOWN_SECONDS, it's treated as almost
# certainly the same physical appearance re-surfacing (ByteTrack losing and
# re-acquiring the same person, brief occlusion, a quick RTSP reconnect) and
# is dropped before ever touching disk/DB. Deliberately per-camera — this is
# NOT identity recognition (no gallery/employee_id involved) and must never
# suppress the same real person showing up again later or on a different
# camera. See _is_recent_duplicate().
#
# Default raised from 120s (the original single-day design) to 900s (15
# min) for multi-day runs: at 120s, someone with even mildly flickery
# tracking sitting in one spot all day could still generate a new capture
# every ~2 minutes — up to ~240/day, times 7 days, of what's essentially one
# person. 15 minutes caps that to at most ~32/day per person per camera
# while still capturing them several times across a day (different times,
# lighting, pose) rather than just once — the "diverse, not just numerous"
# balance the 7-day plan calls for.
DEDUP_COOLDOWN_SECONDS = float(os.environ.get("FACE_DEDUP_COOLDOWN_SECONDS", "900"))
DEDUP_SIMILARITY_THRESHOLD = float(os.environ.get("FACE_DEDUP_SIMILARITY_THRESHOLD", "0.7"))

# Default session length for POST /api/faces/training/collection/start when
# no `days` is given. Per explicit instruction, collection must NOT stop on
# its own after a week — the only stop conditions are an explicit
# /collection/stop call, or someone passing an explicit `days` value. This
# stays a finite number (not a null/"forever" sentinel) so the persisted
# session row + expiry watcher in face_collection.py need no special-casing
# for a missing end time; 100 years is indefinite for any practical purpose.
DEFAULT_COLLECTION_DAYS = float(os.environ.get("FACE_COLLECTION_DAYS", "36500"))

# How often the background expiry watcher (face_collection.py) checks
# whether a running session's planned end time has passed.
SESSION_CHECK_INTERVAL_SECONDS = float(os.environ.get("FACE_SESSION_CHECK_INTERVAL", "60"))

# Quality gate applied ONLY to captures that already have an embedding —
# values below are conservative floors/ceiling, deliberately set well
# outside the range actually observed in real test captures (see
# FACE_TRAINING.md for the measured numbers this was based on), so this
# rejects only genuinely degenerate frames, not normal variation.
MIN_BLUR_SCORE = float(os.environ.get("FACE_MIN_BLUR_SCORE", "50"))
MIN_BRIGHTNESS = float(os.environ.get("FACE_MIN_BRIGHTNESS", "20"))
MAX_BRIGHTNESS = float(os.environ.get("FACE_MAX_BRIGHTNESS", "235"))
# Approximate tight-YOLO-bbox area in pixels^2 (not the padded crop area).
MIN_FACE_AREA = float(os.environ.get("FACE_MIN_AREA", "900"))

# Trained classifier (see face_training.py) — frozen ArcFace embeddings stay
# the feature extractor; this is the supervised layer trained on labeled
# camera captures. Absent until you run POST /api/faces/training/train, at
# which point _match() below starts using it automatically.
CLASSIFIER_PATH = os.path.join(TRAINING_CAPTURE_DIR, "classifier.joblib")
CLASSIFIER_MIN_PROBA = float(os.environ.get("FACE_CLASSIFIER_MIN_PROBA", "0.6"))

# How often (in *processed*, i.e. already-throttled-to-SAMPLE_FPS frames) a
# still-active person track gets its live-overlay identity re-classified —
# see _update_person_identity(). At SAMPLE_FPS=3 this is roughly once per
# second per visible face; deliberately not every frame, since embedding+predict_proba
# is the expensive step and doing it per-frame-per-track would multiply CPU
# cost for no visible UX benefit (see FACE_TRAINING.md's CPU guidance). Only
# runs at all once a classifier has actually been trained — before that,
# zero extra cost is added by this feature.
LIVE_CLASSIFY_INTERVAL_FRAMES = int(os.environ.get("FACE_LIVE_CLASSIFY_INTERVAL_FRAMES", "3"))

# --- Person-first live overlay (separate from the face-track pipeline
# above, which keeps building the training/review dataset exactly as
# before — none of this touches that). ---

# Lowered from 0.5 after a real-camera test (2 seated people, backs to the
# camera, upper body partially occluded by monitors/chairs): at the default
# inference resolution, YOLO found the harder of the two at only 0.25-0.35
# confidence — comfortably real, but below the old 0.5 floor, so that
# person got NO box at all despite being clearly, visibly present. Verified
# on an empty-scene frame from the other camera that 0.3 produces zero
# false positives.
PERSON_DETECT_CONF = float(os.environ.get("PERSON_DETECT_CONF", "0.3"))

# YOLO letterboxes/resizes the frame to this size before detection. These
# cameras shoot 1080p; at the library default (640) a seated/occluded
# person's box shrinks to very few pixels and the model's confidence for
# them drops sharply — confirmed directly: the same real person scored 0.25
# at imgsz=640 vs 0.39 at imgsz=960 in a same-frame comparison. 960 was
# chosen over testing an even higher 1280 (which scored slightly better
# still) because it costs ~1.7x a 640 call instead of ~2.8x, for
# comparable real-world recall on the actual failing case — see
# PERSON_DETECT_INTERVAL_FRAMES below for how that extra cost is kept off
# the live pipeline's CPU budget.
PERSON_DETECT_IMGSZ = int(os.environ.get("PERSON_DETECT_IMGSZ", "960"))

# Independent of SAMPLE_FPS (which still throttles the face-training path
# unchanged): only every Nth processed frame actually runs the (now
# heavier, due to PERSON_DETECT_IMGSZ) person detector. Frames in between
# reuse the last known per-track bbox via self.person_tracks/
# self._live_detections (already how a track survives a momentary miss —
# see PERSON_TRACK_MAX_AGE), so boxes don't disappear or freeze, they just
# update at ~1.5Hz instead of ~3Hz. At 2 gives roughly the SAME total
# person-detection CPU budget as the old 640/every-frame config despite the
# heavier per-call cost — measured, not assumed (see the CPU report this
# change shipped with).
PERSON_DETECT_INTERVAL_FRAMES = int(os.environ.get("PERSON_DETECT_INTERVAL_FRAMES", "2"))

# Independent second safety net against tiny spurious boxes, deliberately
# NOT implemented by raising the confidence threshold back up (that would
# undo the fix above) — any detection below this raw pixel area is dropped
# regardless of confidence. Set well below any real (even distant/seated)
# person's box in these cameras' framing, so it only ever catches genuine
# noise-level detections.
MIN_PERSON_BOX_AREA = float(os.environ.get("PERSON_MIN_BOX_AREA", "3000"))

# --- Rescue detector: a second, independent pass with a stronger (and
# ~3.3x costlier per call, measured) model, run far less often than the
# fast yolov8n pass above. Real-camera testing found frames where yolov8n
# still missed people even at PERSON_DETECT_IMGSZ/lowered confidence —
# specifically people very close to the camera, cropped by the frame edge,
# or at a steep angle (yolov8s found 3/3 real people on a frame where
# yolov8n found only 1/3, confirmed visually, no false positive). Running
# yolov8s at the fast pass's cadence would cost too much CPU on this box;
# instead it only ever ADDS a track the fast pass is missing — it never
# touches or replaces a track the fast pass already has, so normal
# walking/tracking responsiveness is entirely driven by yolov8n, unchanged.
PERSON_RESCUE_WEIGHTS = os.environ.get("PERSON_RESCUE_WEIGHTS", str(_MODELS_DIR / "yolov8s.pt"))

# In raw frame_idx units (same units as PERSON_DETECT_INTERVAL_FRAMES), not
# calls-to-_update_person_overlay units. Must be a multiple of
# PERSON_DETECT_INTERVAL_FRAMES. 6 at SAMPLE_FPS=3 -> a rescue sweep every
# ~2s — measured at ~1.2s/call on this box, so this adds roughly
# 1.2s/2s ≈ 0.6 CPU-core-seconds/sec per camera on top of the fast pass's
# existing ~0.44 — a real, bounded increase, not the ~4x a full switch to
# yolov8s at the fast cadence would cost.
PERSON_RESCUE_INTERVAL_FRAMES = int(os.environ.get("PERSON_RESCUE_INTERVAL_FRAMES", "6"))

# A rescue-pass box counts as "already found" (and is dropped, not added
# as a duplicate track) if it overlaps an already-tracked person's current
# bbox by at least this IoU — deliberately lower than the person tracker's
# own PERSON_TRACKER_MATCH_THRESHOLD since this is just a coarse
# same-person-or-not check, not the tracker's actual association step.
PERSON_RESCUE_DEDUP_IOU = float(os.environ.get("PERSON_RESCUE_DEDUP_IOU", "0.3"))

# Longer than the face tracker's TRACK_MAX_AGE (30): a person's BODY box is
# meant to survive a longer occlusion/turn-away/face-unavailable stretch
# than a face track would — that's the whole point of being person-first
# rather than face-first. ~15s at SAMPLE_FPS=3.
PERSON_TRACK_MAX_AGE = int(os.environ.get("PERSON_TRACK_MAX_AGE", "45"))

# ByteTrack's own IoU threshold for matching a detection to an existing
# track frame-to-frame (separate from YOLO's detection-time NMS, which was
# checked and confirmed NOT dropping boxes — see PERSON_DEBUG logging
# below). Lowered from ByteTrack's 0.8 default: a live-diagnostic capture
# of two people naturally standing close together showed
# raw_yolo_boxes=2 collapse to after_bytetrack_tracks=1 for one frame
# before self-correcting — a newly-appearing second person close to an
# existing track needed an extra frame to get promoted to its own track at
# 0.8. 0.6 gives the tracker more room to keep two close, distinct boxes
# as two tracks immediately instead of briefly folding one into the other.
PERSON_TRACKER_MATCH_THRESHOLD = float(os.environ.get("PERSON_TRACKER_MATCH_THRESHOLD", "0.6"))

# How long a committed identity stays displayed after face recognition last
# confirmed it, before the label falls back to the plain "Person" state —
# the "short temporal grace period" that stops the name flickering the
# instant a face turns away or gets briefly occluded.
IDENTITY_GRACE_SECONDS = float(os.environ.get("IDENTITY_GRACE_SECONDS", "8"))

# Consecutive confident classifier reads of the SAME candidate identity
# required before a person track's displayed identity is committed or
# switched — one lucky (or unlucky) frame is not enough to change what's
# shown, which is what caused the old face-track overlay to flicker.
RECOGNITION_STABILITY_FRAMES = int(os.environ.get("RECOGNITION_STABILITY_FRAMES", "2"))

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(TRAINING_UNLABELED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Track state (per camera)
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """Face track — feeds the training-capture/review pipeline ONLY
    (_recognize_and_route / _save_training_capture). The live-view overlay
    no longer reads anything from this; see PersonTrackState below."""
    track_id: int
    best_score: float = -1.0   # proxy for "quality" — we use bbox area * detection conf
    best_conf: float = -1.0    # raw detection confidence at the best_score frame
    best_area: float = 0.0     # raw (tight, unpadded) YOLO bbox area at the best_score frame
    best_crop: np.ndarray | None = None
    last_seen_frame: int = 0
    pushed_to_review: bool = False


@dataclass
class PersonTrackState:
    """Person track — drives the live-view overlay. A person's existence
    here depends only on the person detector/tracker, never on whether a
    face is currently visible (see _update_person_overlay). Identity is a
    separate, temporally-stabilized layer on top: current_identity only
    changes after RECOGNITION_STABILITY_FRAMES consecutive confident reads
    of the same candidate, and stays displayed for IDENTITY_GRACE_SECONDS
    after last confirmed even if the face becomes unavailable again."""
    track_id: int
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    last_seen_frame: int = 0
    current_identity: str | None = None   # committed employee_id, or None -> "Person"
    identity_confidence: float = 0.0
    last_recognized_time: float = 0.0
    pending_identity: str | None = None   # stability-vote candidate, not yet committed
    pending_count: int = 0
    last_classify_frame: int = -999_999


class CameraFacePipeline:
    """One instance per camera. Call feed_frame(frame) from the same loop
    that already reads RTSP frames in camera_stream.py — no separate
    connection to the camera is needed."""

    _yolo = None
    _yolo_person = None
    _yolo_person_rescue = None
    _arcface = None
    _model_lock = threading.Lock()
    _classifier = None        # sklearn classifier, lazy-loaded from CLASSIFIER_PATH
    _classifier_mtime = None  # reload automatically if the file changes (retrained)

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.tracks: dict[int, TrackState] = {}
        self.person_tracks: dict[int, PersonTrackState] = {}
        self.frame_idx = 0
        self._last_sample_time = 0.0
        # Negative sentinel so the very first has_viewer frame always runs
        # detection immediately rather than waiting a full
        # PERSON_DETECT_INTERVAL_FRAMES after this pipeline is created.
        self._last_person_detect_frame = -PERSON_DETECT_INTERVAL_FRAMES
        self._last_rescue_frame = -PERSON_RESCUE_INTERVAL_FRAMES
        self._gallery_cache: list[dict] = []
        self._gallery_loaded_at = 0.0
        # Recent (timestamp, embedding) pairs for THIS camera only, used by
        # _is_recent_duplicate() — only ever read/written by this camera's
        # own background thread (feed_frame is called from exactly one
        # thread per camera, see camera_stream.py), so no lock needed here.
        self._recent_captures: list[tuple[float, np.ndarray]] = []
        # Rebuilt in full at the end of every feed_frame() call (see
        # get_live_detections) — read from the websocket-serving asyncio
        # loop in main.py, written from this camera's own single background
        # thread. A plain list-reference swap is what's read; no lock, same
        # reasoning as _recent_captures above (single writer thread) plus
        # CPython list/attribute assignment being atomic under the GIL.
        self._live_detections: list[dict] = []
        self._ensure_models_loaded()

        # ByteTrack: lightweight, no GPU needed, just IoU + Kalman motion.
        # `supervision`'s implementation (see module docstring for why, over
        # the original `yolox` package). Two independent instances/track-id
        # spaces — face tracks (training-capture pipeline, unchanged) and
        # person tracks (live overlay) are tracked separately since they're
        # different detections with different lifetimes.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=TRACK_MAX_AGE,
            minimum_matching_threshold=0.8,
        )
        self.person_tracker = sv.ByteTrack(
            track_activation_threshold=0.4,
            lost_track_buffer=PERSON_TRACK_MAX_AGE,
            minimum_matching_threshold=PERSON_TRACKER_MATCH_THRESHOLD,
        )

    @classmethod
    def _ensure_arcface_loaded(cls):
        """Only the embedding model — enough for enrolling a photo directly
        (face_routes.enroll), which doesn't need detection/tracking at all
        since the human already framed the face."""
        with cls._model_lock:
            if cls._arcface is None:
                # This box has no CUDA (torch reports cuda.is_available() ==
                # False, and only plain `onnxruntime` is installed, not
                # `onnxruntime-gpu`) — CPUExecutionProvider only. Swap back
                # to the CUDA+CPU list if this ever runs on a GPU box.
                cls._arcface = insightface.app.FaceAnalysis(
                    name="buffalo_l", providers=["CPUExecutionProvider"]
                )
                cls._arcface.prepare(ctx_id=-1, det_size=(640, 640))

    @classmethod
    def _ensure_yolo_loaded(cls):
        with cls._model_lock:
            if cls._yolo is None:
                cls._yolo = YOLO(YOLO_FACE_WEIGHTS)

    @classmethod
    def _ensure_yolo_person_loaded(cls):
        """Generic COCO detector for the person-first live overlay — a
        separate model instance from the face-tuned one above (see module
        docstring for why the face model is never reused for this)."""
        with cls._model_lock:
            if cls._yolo_person is None:
                cls._yolo_person = YOLO(YOLO_PERSON_WEIGHTS)

    @classmethod
    def _ensure_yolo_person_rescue_loaded(cls):
        """Stronger, slower model used only for the occasional rescue sweep
        — see PERSON_RESCUE_WEIGHTS above. Loaded eagerly alongside the
        other models (same pattern as _ensure_yolo_person_loaded) rather
        than on first use, so a rescue cycle never pays a cold-load stall
        the first time it fires."""
        with cls._model_lock:
            if cls._yolo_person_rescue is None:
                cls._yolo_person_rescue = YOLO(PERSON_RESCUE_WEIGHTS)

    @classmethod
    def _ensure_models_loaded(cls):
        """All four models — needed for the live per-camera pipeline
        (face detect + track + embed, plus person detect + rescue for the
        overlay), unlike enrollment which only needs ArcFace."""
        cls._ensure_yolo_loaded()
        cls._ensure_yolo_person_loaded()
        cls._ensure_yolo_person_rescue_loaded()
        cls._ensure_arcface_loaded()

    def _refresh_gallery(self):
        # Reload the enrolled embeddings every 30s rather than per-frame.
        now = time.time()
        if now - self._gallery_loaded_at > 30 or not self._gallery_cache:
            self._gallery_cache = face_db.get_all_embeddings()
            self._gallery_loaded_at = now

    @classmethod
    def _get_classifier(cls):
        """Lazy-load the trained classifier and pick up a fresh one
        automatically if POST /api/faces/training/train produced a new file
        while this process is running — checked via mtime, not re-imported
        from disk on every single match (that stat call is cheap; loading
        the pickle is not)."""
        if not os.path.exists(CLASSIFIER_PATH):
            return None
        mtime = os.path.getmtime(CLASSIFIER_PATH)
        if cls._classifier is None or mtime != cls._classifier_mtime:
            with cls._model_lock:
                import joblib
                cls._classifier = joblib.load(CLASSIFIER_PATH)
                cls._classifier_mtime = mtime
        return cls._classifier

    @staticmethod
    def _pad_bbox(x1: int, y1: int, x2: int, y2: int, frame_shape: tuple, margin: float = FACE_CROP_MARGIN):
        """Expand a detection box by `margin` fraction of its size on each
        side, clamped to the frame. See FACE_CROP_MARGIN above for why this
        exists — feeds InsightFace's re-detection/alignment step (in
        _embed(), unchanged) enough surrounding context to actually find
        the face, instead of a bbox-tight crop it reliably fails on."""
        h, w = frame_shape[:2]
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * margin), int(bh * margin)
        return max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)

    @staticmethod
    def _quality_metrics(crop: np.ndarray) -> tuple[float, float]:
        """Cheap, first-version-only quality signals: blur (variance of
        Laplacian — higher is sharper) and brightness (mean grayscale
        intensity). Stored alongside every capture so a future pass can
        filter out unusable samples before training without having to
        re-derive this from the image again."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        return blur, brightness

    def feed_frame(self, frame: np.ndarray, has_viewer: bool = True):
        """Call this on every frame from the existing RTSP read loop.
        Internally throttles to SAMPLE_FPS so we don't run YOLO on every
        single frame — 2-5 fps is plenty for catching faces walking past.

        `has_viewer`: whether anyone is actually watching this camera's
        live feed right now (see camera_stream.CameraStream.has_real_viewer).
        The background face-collection dataset pipeline below ALWAYS runs
        regardless — that's real training data being built 24/7 and must
        not depend on a browser tab being open. Only the person-detection
        live-overlay (a second full YOLO pass, purely cosmetic for a
        connected viewer) is skipped while has_viewer is False, so it isn't
        burning CPU around the clock for a display nobody is looking at."""
        now = time.time()
        if now - self._last_sample_time < (1.0 / SAMPLE_FPS):
            return
        self._last_sample_time = now
        self.frame_idx += 1

        # --- Face detection + face tracking: UNCHANGED from before the
        # person-first overlay existed. Feeds ONLY the training-capture /
        # review pipeline below (_recognize_and_route / _save_training_capture)
        # — the live-view overlay no longer reads from this at all (see
        # _update_person_overlay). ---
        detections = self._detect_faces(frame)  # -> np.array [[x1,y1,x2,y2,conf], ...]
        if len(detections):
            sv_detections = sv.Detections(xyxy=detections[:, :4], confidence=detections[:, 4])
            tracked = self.tracker.update_with_detections(sv_detections)
        else:
            tracked = sv.Detections.empty()

        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            track_id = int(tracked.tracker_id[i])
            score = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0

            x1, y1, x2, y2 = map(lambda v: max(0, int(v)), (x1, y1, x2, y2))
            px1, py1, px2, py2 = self._pad_bbox(x1, y1, x2, y2, frame.shape)
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            # Quality/tracking scoring stays keyed on the tight YOLO box —
            # only the stored crop itself is padded.
            area = (x2 - x1) * (y2 - y1)
            quality = area * score

            state = self.tracks.setdefault(track_id, TrackState(track_id=track_id))
            state.last_seen_frame = self.frame_idx
            if quality > state.best_score:
                state.best_score = quality
                state.best_conf = score
                state.best_area = area
                state.best_crop = crop.copy()

        # Drop stale tracks and flush any that finished (left frame / went stale)
        # to recognition — this is where "best frame per track" pays off:
        # one embedding computed per person-appearance, not per frame.
        stale = [
            tid for tid, s in self.tracks.items()
            if self.frame_idx - s.last_seen_frame > TRACK_MAX_AGE
        ]
        for tid in stale:
            state = self.tracks.pop(tid)
            if state.best_crop is not None and not state.pushed_to_review:
                self._recognize_and_route(state)

        # --- Person-first live overlay: entirely separate track space and
        # display logic. Reuses this same frame's face detections (above)
        # for identity, but a person's box/existence never depends on one
        # being found. Skipped while nobody is watching — see has_viewer
        # docstring above. person_tracks is left as-is (not cleared) while
        # skipped; frame_idx keeps advancing underneath it, so on the next
        # real frame every existing track reads as long-stale and is
        # dropped cleanly rather than reappearing with a stale box. ---
        if has_viewer and self.frame_idx - self._last_person_detect_frame >= PERSON_DETECT_INTERVAL_FRAMES:
            self._last_person_detect_frame = self.frame_idx
            self._update_person_overlay(frame, detections)

    def _detect_faces(self, frame: np.ndarray) -> np.ndarray:
        results = self._yolo.predict(frame, verbose=False, conf=0.5)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        return np.hstack([boxes, confs[:, None]]).astype(np.float32)

    @staticmethod
    def _filter_person_boxes(results) -> np.ndarray:
        """Shared by the fast and rescue person detectors: same contract as
        _detect_faces's return shape, plus MIN_PERSON_BOX_AREA — a second,
        confidence-independent floor against noise-level boxes."""
        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = areas >= MIN_PERSON_BOX_AREA
        return np.hstack([boxes[keep], confs[keep, None]]).astype(np.float32)

    def _detect_persons(self, frame: np.ndarray) -> np.ndarray:
        """The fast, frequent pass (every PERSON_DETECT_INTERVAL_FRAMES) —
        generic COCO model restricted to class 0 ("person") — classes=[0]
        so a chair or laptop never becomes a detection in the first place,
        rather than being filtered out after the fact. imgsz=
        PERSON_DETECT_IMGSZ (see its comment) is what recovers a
        seated/back-facing person that the library's 640 default missed
        entirely. Drives all normal tracking/walking responsiveness."""
        results = self._yolo_person.predict(
            frame, verbose=False, conf=PERSON_DETECT_CONF, classes=[0], imgsz=PERSON_DETECT_IMGSZ,
        )[0]
        return self._filter_person_boxes(results)

    def _detect_persons_rescue(self, frame: np.ndarray) -> np.ndarray:
        """The slow, occasional pass (every PERSON_RESCUE_INTERVAL_FRAMES)
        — a stronger, costlier model, same conf/imgsz/area filtering, used
        only to catch people the fast pass above is missing (see
        PERSON_RESCUE_WEIGHTS). Never used to update a track the fast pass
        already found — see _merge_rescue_detections."""
        results = self._yolo_person_rescue.predict(
            frame, verbose=False, conf=PERSON_DETECT_CONF, classes=[0], imgsz=PERSON_DETECT_IMGSZ,
        )[0]
        return self._filter_person_boxes(results)

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = box_a[:4]
        bx1, by1, bx2, by2 = box_b[:4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _merge_rescue_detections(self, fast_dets: np.ndarray, rescue_dets: np.ndarray) -> np.ndarray:
        """Appends only the rescue-pass boxes that don't already overlap a
        fast-pass box (PERSON_RESCUE_DEDUP_IOU) — this is what makes the
        rescue pass strictly additive: it can only ever introduce a NEW
        track for someone the fast pass missed this cycle, never move,
        replace, or duplicate a box the fast pass already produced."""
        if len(rescue_dets) == 0:
            return fast_dets
        if len(fast_dets) == 0:
            return rescue_dets
        new_boxes = [
            rd for rd in rescue_dets
            if max(self._iou(rd, fd) for fd in fast_dets) < PERSON_RESCUE_DEDUP_IOU
        ]
        if not new_boxes:
            return fast_dets
        return np.vstack([fast_dets, np.array(new_boxes, dtype=np.float32)])

    @staticmethod
    def _faces_by_person(person_boxes: list[tuple[int, int, int, int]], face_detections: np.ndarray) -> dict[int, np.ndarray]:
        """Maps person-box index -> the best face detection whose CENTER
        point falls inside that box (clearest faces claimed first, each
        face used at most once) — this is the "person exists -> look for a
        face inside it" association, never the other way around. A person
        with no face inside simply gets no entry here, which is expected
        and handled by the caller (identity just doesn't update this cycle,
        the box is shown regardless)."""
        assignments: dict[int, np.ndarray] = {}
        if len(face_detections) == 0 or not person_boxes:
            return assignments
        order = sorted(range(len(face_detections)), key=lambda i: -face_detections[i][4])
        for fi in order:
            fx1, fy1, fx2, fy2, _fconf = face_detections[fi]
            cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            for pi, (px1, py1, px2, py2) in enumerate(person_boxes):
                if pi in assignments:
                    continue
                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    assignments[pi] = face_detections[fi]
                    break
        return assignments

    def _update_person_overlay(self, frame: np.ndarray, face_detections: np.ndarray) -> None:
        """Person-first: detect + track bodies unconditionally, then for
        each currently-visible person track, try to find a face inside its
        box and (throttled) refresh its temporally-stabilized identity. A
        person's box is shown regardless of whether a face is found — see
        module-level PersonTrackState docstring for the stability/grace
        rules that decide what label goes with it."""
        person_dets = self._detect_persons(frame)

        if self.frame_idx - self._last_rescue_frame >= PERSON_RESCUE_INTERVAL_FRAMES:
            self._last_rescue_frame = self.frame_idx
            rescue_dets = self._detect_persons_rescue(frame)
            person_dets = self._merge_rescue_detections(person_dets, rescue_dets)

        # Always call update_with_detections, even with zero detections —
        # ByteTrack still needs every frame to age its internal state
        # correctly. Its RETURN value only ever contains tracks matched to
        # an actual detection THIS frame though (it doesn't Kalman-predict
        # a box forward through a miss), so a single frame where the
        # detector found nobody would otherwise wipe every person off the
        # overlay — directly against "box must not disappear just because
        # detection momentarily missed". Fixed below by building the
        # overlay from self.person_tracks (each track's LAST KNOWN bbox,
        # kept until it's truly gone for PERSON_TRACK_MAX_AGE frames), not
        # from this frame's raw tracker output alone.
        if len(person_dets):
            sv_persons = sv.Detections(xyxy=person_dets[:, :4], confidence=person_dets[:, 4])
            tracked_persons = self.person_tracker.update_with_detections(sv_persons)
        else:
            tracked_persons = self.person_tracker.update_with_detections(sv.Detections.empty())

        n = len(tracked_persons)
        seen_this_frame: dict[int, tuple[int, int, int, int]] = {}
        for i in range(n):
            track_id = int(tracked_persons.tracker_id[i])
            x1, y1, x2, y2 = map(lambda v: max(0, int(v)), tracked_persons.xyxy[i])
            seen_this_frame[track_id] = (x1, y1, x2, y2)

        # Face association only makes sense against boxes actually detected
        # this frame — a coasting track's stale bbox has no fresh face
        # detection to look inside.
        track_ids_this_frame = list(seen_this_frame.keys())
        boxes_this_frame = [seen_this_frame[tid] for tid in track_ids_this_frame]
        face_assignments = self._faces_by_person(boxes_this_frame, face_detections)

        for idx, track_id in enumerate(track_ids_this_frame):
            pstate = self.person_tracks.setdefault(track_id, PersonTrackState(track_id=track_id))
            pstate.bbox = boxes_this_frame[idx]
            pstate.last_seen_frame = self.frame_idx
            face_row = face_assignments.get(idx)
            if face_row is not None:
                self._update_person_identity(pstate, frame, face_row)

        now = time.time()
        live_this_frame = []
        stale_persons = []
        for track_id, pstate in self.person_tracks.items():
            if self.frame_idx - pstate.last_seen_frame > PERSON_TRACK_MAX_AGE:
                stale_persons.append(track_id)
                continue

            # Grace period evaluated fresh every frame at display time, not
            # baked into current_identity itself — so it decays smoothly
            # rather than needing its own timer/callback.
            display_identity = None
            if pstate.current_identity and (now - pstate.last_recognized_time) <= IDENTITY_GRACE_SECONDS:
                display_identity = pstate.current_identity

            # name/color resolved from employee_directory.py, keyed by
            # employee_id only — never changes tracking/identity logic
            # above, just what's attached to the already-decided ID for
            # display. None name -> frontend shows "Person"/neutral color,
            # never a guessed company.
            name, color = employee_directory.get_display(display_identity)

            live_this_frame.append({
                "track_id": track_id,
                "bbox": list(pstate.bbox),
                "employee_id": display_identity,  # None -> frontend shows "Person", never "Unknown"
                "name": name,
                "color": color,
                "confidence": pstate.identity_confidence if display_identity else 0.0,
            })

        # Replaced wholesale each frame (not merged) — a person who's
        # genuinely aged out (PERSON_TRACK_MAX_AGE) drops off the overlay on
        # the very next update; everyone still within that window keeps
        # showing at their last known position regardless of this frame's
        # raw detection result (see comment above).
        self._live_detections = live_this_frame

        for tid in stale_persons:
            self.person_tracks.pop(tid)

    def _update_person_identity(self, pstate: PersonTrackState, frame: np.ndarray, face_row: np.ndarray) -> None:
        """Throttled (LIVE_CLASSIFY_INTERVAL_FRAMES) classifier read on the
        face found inside this person's box, with a stability vote before
        committing/switching pstate.current_identity — see PersonTrackState
        docstring. No-op entirely if no classifier has been trained yet
        (zero extra CPU cost before that point, same as before)."""
        clf = self._get_classifier()
        if clf is None:
            return
        if self.frame_idx - pstate.last_classify_frame < LIVE_CLASSIFY_INTERVAL_FRAMES:
            return
        pstate.last_classify_frame = self.frame_idx

        fx1, fy1, fx2, fy2, _fconf = face_row
        fx1, fy1, fx2, fy2 = map(lambda v: max(0, int(v)), (fx1, fy1, fx2, fy2))
        px1, py1, px2, py2 = self._pad_bbox(fx1, fy1, fx2, fy2, frame.shape)
        crop = frame[py1:py2, px1:px2]
        if crop.size == 0:
            return
        embedding = self._embed(crop)
        if embedding is None:
            return

        proba = clf.predict_proba(embedding.reshape(1, -1))[0]
        best_idx = int(np.argmax(proba))
        score = float(proba[best_idx])
        if score < CLASSIFIER_MIN_PROBA:
            # Not confident this cycle. Does NOT clear an already-committed
            # identity (the grace period, checked at display time, handles
            # that decay) — just resets the stability vote so a run of weak
            # reads can't slowly accumulate into a wrong switch later.
            pstate.pending_identity = None
            pstate.pending_count = 0
            return

        predicted = str(clf.classes_[best_idx])

        if predicted == pstate.current_identity:
            # Already showing this person — just refresh the grace timer
            # and confidence, no need to re-run the stability vote.
            pstate.last_recognized_time = time.time()
            pstate.identity_confidence = score
            return

        if predicted == pstate.pending_identity:
            pstate.pending_count += 1
        else:
            pstate.pending_identity = predicted
            pstate.pending_count = 1

        if pstate.pending_count >= RECOGNITION_STABILITY_FRAMES:
            pstate.current_identity = predicted
            pstate.identity_confidence = score
            pstate.last_recognized_time = time.time()

    def get_live_detections(self) -> list[dict]:
        """Snapshot for the /ws/detections/{camera_id} route — see the
        _live_detections attribute comment for the threading contract.
        Now person-track-based (see _update_person_overlay), not face-track."""
        return self._live_detections

    def _embed(self, crop: np.ndarray) -> np.ndarray | None:
        """Run InsightFace's own detector on the crop to get landmarks and
        alignment right, then extract the ArcFace embedding. This
        re-detection-on-the-crop step is deliberate: YOLO gave us a fast
        rough box for tracking, but ArcFace's accuracy depends heavily on
        proper 5-point alignment, which InsightFace's FaceAnalysis handles
        internally. Skipping this (feeding YOLO's raw crop straight into an
        embedding model) is the most common cause of "everyone is unknown."
        """
        faces = self._arcface.get(crop)
        if not faces:
            return None
        # Largest face in the crop (should be the only one)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding  # already L2-normalized, 512-dim

    def _match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Two modes, tried in this order:
        1. Trained classifier (POST /api/faces/training/train has been run
           at least once) — a supervised decision boundary learned from
           real, human-labeled camera captures. score is predict_proba for
           the winning class, compared against FACE_CLASSIFIER_MIN_PROBA.
        2. Fallback: nearest-neighbor cosine similarity against whatever's
           in face_embeddings (enrollment photos + assigned review
           captures), compared against MATCH_THRESHOLD. This is the only
           mode available before any training run.
        Both return (person_id | None, score) so callers don't need to know
        which mode produced it — MATCH_THRESHOLD vs. CLASSIFIER_MIN_PROBA is
        handled here, not by the caller.
        """
        clf = self._get_classifier()
        if clf is not None:
            proba = clf.predict_proba(embedding.reshape(1, -1))[0]
            best_idx = int(np.argmax(proba))
            return str(clf.classes_[best_idx]), float(proba[best_idx])

        self._refresh_gallery()
        if not self._gallery_cache:
            return None, 0.0
        best_person, best_score = None, -1.0
        for entry in self._gallery_cache:
            gallery_vec = np.array(entry["embedding"], dtype=np.float32)
            score = float(np.dot(embedding, gallery_vec))  # cosine sim (both normalized)
            if score > best_score:
                best_score = score
                best_person = entry["person_id"]
        return best_person, best_score

    def _match_threshold(self) -> float:
        return CLASSIFIER_MIN_PROBA if self._get_classifier() is not None else MATCH_THRESHOLD

    def _is_recent_duplicate(self, embedding: np.ndarray) -> bool:
        """True if `embedding` is highly similar to one captured on THIS
        camera within the last DEDUP_COOLDOWN_SECONDS — almost certainly the
        same physical appearance re-surfacing under a new track_id (tracking
        flicker, brief occlusion, a quick RTSP reconnect), not a genuinely
        new sighting. Deliberately narrow in scope:
          - per-camera only (each pipeline instance has its own list)
          - cooldown window (default 900s / 15 min for multi-day runs — see
            DEDUP_COOLDOWN_SECONDS) — does NOT suppress the same person
            appearing again later in the day, nor on a different camera
          - a single cosine-similarity check against recent embeddings,
            reusing the exact same math _match() already uses — no new
            model, no employee_id involved, nothing is auto-labeled.
        """
        now = time.time()
        self._recent_captures = [
            (t, e) for t, e in self._recent_captures if now - t <= DEDUP_COOLDOWN_SECONDS
        ]
        for _, recent_embedding in self._recent_captures:
            if float(np.dot(embedding, recent_embedding)) >= DEDUP_SIMILARITY_THRESHOLD:
                return True
        return False

    def _save_training_capture(self, state: TrackState, embedding: np.ndarray | None):
        """Every finished track lands here, unconditionally — confident
        match, low-confidence, or no embedding at all. This is the bulk
        dataset for manual labeling; face_pending (below) stays a separate,
        narrower live-correction queue. Never lets a failure here stop
        recognition — worst case, this capture is silently missing from the
        training set.

        Gates, in order:
          1. Cross-track duplicate check (embedding-based, see
             _is_recent_duplicate) — skips entirely, no file, no DB row.
             Counted (face_db.increment_running_session_duplicates_rejected)
             purely for monitoring visibility into how much this is
             happening; the count is session-scoped, not enforcement.
          2. MAX_TRAINING_CAPTURES — a global, all-cameras, all-statuses
             ceiling — and MAX_CAPTURES_PER_CAMERA — this camera's share of
             it — both checked+enforced atomically via
             face_db.capture_limit_lock so concurrent camera threads can't
             overshoot either.
          3. label_status is decided from embedding presence + quality
             metrics: only a capture with a real embedding AND acceptable
             blur/brightness/face-size becomes 'unlabeled' (the only status
             /face-training's queue ever shows). Everything else is still
             saved (nothing is deleted) as 'no_embedding' or 'rejected' for
             diagnostics, but is invisible to the labeling queue and
             excluded from classifier training.
        """
        try:
            if embedding is not None and self._is_recent_duplicate(embedding):
                try:
                    face_db.increment_running_session_duplicates_rejected()
                except Exception:
                    pass
                return

            with face_db.capture_limit_lock:
                if face_db.count_training_captures() >= MAX_TRAINING_CAPTURES:
                    return
                if face_db.count_training_captures_for_camera(self.camera_id) >= MAX_CAPTURES_PER_CAMERA:
                    return

                blur, brightness = self._quality_metrics(state.best_crop)

                if embedding is None:
                    label_status = "no_embedding"
                elif (
                    blur < MIN_BLUR_SCORE
                    or not (MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS)
                    or state.best_area < MIN_FACE_AREA
                ):
                    label_status = "rejected"
                else:
                    label_status = "unlabeled"

                image_path = os.path.join(
                    TRAINING_UNLABELED_DIR,
                    f"cam{self.camera_id}_track{state.track_id}_{uuid.uuid4().hex[:8]}.jpg",
                )
                write_ok = cv2.imwrite(image_path, state.best_crop)
                # Never create a DB row for an image that isn't actually
                # sitting on disk — cv2.imwrite() can return True yet still
                # not leave a readable file behind on some setups (e.g. a
                # synced/cloud-backed folder interfering with a burst of
                # small file writes), which previously produced orphaned
                # rows whose /face-training image would silently 404. Both
                # checks: the call's own return value, and a real stat.
                if not write_ok or not os.path.exists(image_path):
                    log.error(
                        "camera %s: wrote training capture image but it's not on disk afterward (%s) — dropping this capture",
                        self.camera_id, image_path,
                    )
                    return
                face_db.add_training_capture(
                    camera_id=self.camera_id,
                    track_id=state.track_id,
                    image_path=image_path,
                    embedding=embedding.tolist() if embedding is not None else None,
                    detection_confidence=state.best_conf,
                    blur_score=blur,
                    brightness=brightness,
                    label_status=label_status,
                )

            try:
                face_db.touch_running_collection_session()
            except Exception:
                pass

            if embedding is not None:
                self._recent_captures.append((time.time(), embedding))
        except Exception:
            pass

    def _recognize_and_route(self, state: TrackState):
        embedding = self._embed(state.best_crop)

        # Bulk dataset collection — unconditional, regardless of whether a
        # clean embedding or a confident match came out of this track.
        self._save_training_capture(state, embedding)

        if embedding is None:
            return  # couldn't get a clean aligned face out of the crop; drop it

        person_id, score = self._match(embedding)

        if person_id is not None and score >= self._match_threshold():
            # Confident match: no human needed. Optionally still log the
            # sighting somewhere (attendance/footfall tables) — not shown
            # here since that's a separate concern from recognition itself.
            return

        # Below threshold (or empty gallery) -> queue for human review,
        # but don't spam the same track twice within REVIEW_DEDUPE_SECONDS.
        since = time.time() - REVIEW_DEDUPE_SECONDS
        if face_db.has_pending_for_track(self.camera_id, state.track_id, since):
            return

        image_path = os.path.join(
            CAPTURE_DIR, f"cam{self.camera_id}_track{state.track_id}_{uuid.uuid4().hex[:8]}.jpg"
        )
        cv2.imwrite(image_path, state.best_crop)

        face_db.add_pending(
            camera_id=self.camera_id,
            track_id=state.track_id,
            image_path=image_path,
            embedding=embedding.tolist(),
            best_match_person_id=person_id,
            best_match_score=score if person_id else None,
        )
        state.pushed_to_review = True


# ---------------------------------------------------------------------------
# Registry: one pipeline instance per camera, reused across frames
# ---------------------------------------------------------------------------

_pipelines: dict[int, CameraFacePipeline] = {}
_pipelines_lock = threading.Lock()


def get_pipeline(camera_id: int) -> CameraFacePipeline:
    with _pipelines_lock:
        if camera_id not in _pipelines:
            _pipelines[camera_id] = CameraFacePipeline(camera_id)
        return _pipelines[camera_id]


def get_existing_pipeline(camera_id: int) -> "CameraFacePipeline | None":
    """Non-creating lookup for /ws/detections — that route must never spin
    up a full pipeline (loads YOLO+ArcFace) on its own just because a
    browser tab opened the overlay socket; a pipeline only ever gets created
    by camera_stream.py once /ws/live's RTSP read loop for that camera is
    actually running."""
    with _pipelines_lock:
        return _pipelines.get(camera_id)
