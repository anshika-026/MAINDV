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
# no `days` is given.
DEFAULT_COLLECTION_DAYS = float(os.environ.get("FACE_COLLECTION_DAYS", "7"))

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

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(TRAINING_UNLABELED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Track state (per camera)
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    track_id: int
    best_score: float = -1.0   # proxy for "quality" — we use bbox area * detection conf
    best_conf: float = -1.0    # raw detection confidence at the best_score frame
    best_area: float = 0.0     # raw (tight, unpadded) YOLO bbox area at the best_score frame
    best_crop: np.ndarray | None = None
    last_seen_frame: int = 0
    recognized_person_id: str | None = None
    pushed_to_review: bool = False


class CameraFacePipeline:
    """One instance per camera. Call feed_frame(frame) from the same loop
    that already reads RTSP frames in camera_stream.py — no separate
    connection to the camera is needed."""

    _yolo = None
    _arcface = None
    _model_lock = threading.Lock()
    _classifier = None        # sklearn classifier, lazy-loaded from CLASSIFIER_PATH
    _classifier_mtime = None  # reload automatically if the file changes (retrained)

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.tracks: dict[int, TrackState] = {}
        self.frame_idx = 0
        self._last_sample_time = 0.0
        self._gallery_cache: list[dict] = []
        self._gallery_loaded_at = 0.0
        # Recent (timestamp, embedding) pairs for THIS camera only, used by
        # _is_recent_duplicate() — only ever read/written by this camera's
        # own background thread (feed_frame is called from exactly one
        # thread per camera, see camera_stream.py), so no lock needed here.
        self._recent_captures: list[tuple[float, np.ndarray]] = []
        self._ensure_models_loaded()

        # ByteTrack: lightweight, no GPU needed, just IoU + Kalman motion.
        # `supervision`'s implementation (see module docstring for why, over
        # the original `yolox` package).
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=TRACK_MAX_AGE,
            minimum_matching_threshold=0.8,
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
    def _ensure_models_loaded(cls):
        """Both models — needed for the live per-camera pipeline (detect +
        track + embed), unlike enrollment which only needs ArcFace."""
        cls._ensure_yolo_loaded()
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

    def feed_frame(self, frame: np.ndarray):
        """Call this on every frame from the existing RTSP read loop.
        Internally throttles to SAMPLE_FPS so we don't run YOLO on every
        single frame — 2-5 fps is plenty for catching faces walking past."""
        now = time.time()
        if now - self._last_sample_time < (1.0 / SAMPLE_FPS):
            return
        self._last_sample_time = now
        self.frame_idx += 1

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

    def _detect_faces(self, frame: np.ndarray) -> np.ndarray:
        results = self._yolo.predict(frame, verbose=False, conf=0.5)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        return np.hstack([boxes, confs[:, None]]).astype(np.float32)

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
            state.recognized_person_id = person_id
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
