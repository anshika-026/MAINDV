"""Per-frame integration hook for the body-appearance Re-ID unique-footfall
engine — called from inside detection_worker.py's existing per-camera
worker-process loop, mirroring peopleid_worker.py's role/structure exactly
(same process, same reasoning: reused across every tracked person in every
frame, wrapped in try/except at the call site in detection_worker.py so a
failure here can never take down face-rec/footfall/desk/gate processing
sharing that process).

The one thing genuinely different from peopleid_worker.py: a track with no
confident gallery match isn't just flagged for a human to review later —
after REID_AUTO_ENROLL_MIN_OBSERVATIONS quality-passing observations with no
match, this module curates the track's own accumulated crops (spec sections
2/7/31: identity creation must be fully automatic, no human involved) and
hands the curated set back in its result dict as "pending_new_person". All
SQLite writes still happen in the main process (pipeline.py's
_dispatch_reid_result) — this module and the worker process it runs in never
touch reid_db directly, same boundary peopleid_worker.py already respects.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import config, peopleid_enrollment, peopleid_fusion, peopleid_gallery, reid_db, reid_embedding, reid_quality
from .peopleid_tracker import PersonBodyTracker


class ReidWorkerState:
    """camera_config: this camera's reid_db.get_camera_config() row,
    resolved ONCE by the main process at worker-spawn time — same
    load-time-constant convention as PeopleIdWorkerState.camera_config (see
    that class' docstring)."""

    def __init__(self, camera_config: dict | None = None):
        self.tracker: PersonBodyTracker | None = None  # lazy — only built if this camera actually runs Re-ID
        self.embedder: reid_embedding.BodyReIdEmbedder | None = None  # lazy — real model load, do it once, off the import path
        self.fusion = peopleid_fusion.IdentityFusion(
            window=config.REID_FUSION_WINDOW,
            candidate_min_votes=config.REID_CANDIDATE_MIN_VOTES,
            confirmed_min_votes=config.REID_CONFIRMED_MIN_VOTES,
            contradiction_limit=config.REID_CONFIRMED_CONTRADICTION_LIMIT,
            track_timeout_seconds=config.REID_TRACK_TIMEOUT_SECONDS,
        )
        self.gallery = peopleid_gallery.VectorGallery(
            embedding_loader=reid_db.load_all_embeddings,
            similarity_threshold=config.REID_SIMILARITY_THRESHOLD,
            min_margin=config.REID_MIN_MARGIN,
            # Identities here are auto-created, so the gallery can hold
            # several fragments of one person; a near-tie between two of
            # them that BOTH clear the threshold means "same human", not
            # "unsure who". See VectorGallery.__init__.
            duplicate_identities_expected=True,
        )
        self.last_mot_at: float = 0.0
        # track_id -> accumulated quality-passing, unmatched samples, same
        # shape/role as scripts/demo_reid.py's DemoRunner._pending_samples
        # (that script's own copy exists only because it predates this
        # module and never imports the live pipeline's process boundary —
        # see this module's docstring for why the REAL version can't just
        # write to reid_db directly the way the demo script conveniently does).
        self.pending_samples: dict[int, list[dict]] = {}
        # Tracks that have already produced an identity this session.
        #
        # Auto-enrolment fires when a track accumulates
        # REID_AUTO_ENROLL_MIN_OBSERVATIONS unmatched samples, and then
        # clears them. Without this set, a track that STILL doesn't match
        # simply starts accumulating again and mints a second identity for
        # the same person a few seconds later — observed live: track 81
        # created both PERSON_017 and PERSON_018 twelve seconds apart. It
        # happens because the person this track just created isn't
        # matchable yet (the main process has to write the rows and signal
        # every worker to rebuild its gallery) and because a body seen from
        # a new angle may not clear the threshold against its own first
        # embeddings.
        #
        # One continuous track is one person by definition, so it may only
        # ever create ONE identity. Further observations still match
        # normally — they just can't enrol again.
        self.enrolled_tracks: set[int] = set()
        # track_id -> when it was last processed, so the two structures above
        # can be dropped once a track is long gone. The tracker hands out a
        # fresh id per appearance and never reuses one, so without this both
        # would grow for as long as the process lives.
        self.track_last_seen: dict[int, float] = {}
        self.camera_config = camera_config or {}

    def prune_expired_tracks(self, now: float, timeout_seconds: float) -> None:
        stale = [tid for tid, seen in self.track_last_seen.items() if now - seen > timeout_seconds]
        for track_id in stale:
            del self.track_last_seen[track_id]
            self.pending_samples.pop(track_id, None)
            self.enrolled_tracks.discard(track_id)


def _in_roi(bbox: list[float], roi: list[list[float]] | None, frame_shape: tuple[int, int]) -> bool:
    if not roi:
        return True
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h  # fractional, matches reid_db.set_camera_config's roi convention
    contour = np.array(roi, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, (cx, cy), False) >= 0


def _encode_small_jpeg(crop: np.ndarray) -> bytes | None:
    if crop.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else None


def _crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


def process_frame(camera_id: int, frame: np.ndarray, state: ReidWorkerState) -> dict | None:
    """Returns {"tracks": [...]}, attached to detection_worker's result
    dict as result["reid"] when this cycle's MOT cadence fires; None when
    the cadence gate hasn't elapsed yet — same optional-key convention
    peopleid_worker.process_frame already uses."""
    now = time.time()
    mot_interval = state.camera_config.get("mot_interval_seconds") or config.REID_MOT_INTERVAL_SECONDS
    if now - state.last_mot_at < mot_interval:
        return None
    state.last_mot_at = now

    if state.tracker is None:
        state.tracker = PersonBodyTracker(config.PEOPLEID_MOT_MODEL_PATH, config.PEOPLEID_MOT_IMGSZ, config.PEOPLEID_MOT_CONFIDENCE)
    if state.embedder is None:
        state.embedder = reid_embedding.BodyReIdEmbedder(config.REID_MODEL_NAME, config.REID_MODEL_PATH, config.REID_DEVICE)

    people = state.tracker.update(frame)
    roi = state.camera_config.get("roi")
    track_results = []
    for person in people:
        if not _in_roi(person["bbox"], roi, frame.shape):
            continue
        try:
            track_result = _process_one_track(camera_id, person["track_id"], person["bbox"], person["confidence"], frame, state, now)
        except Exception:
            # One bad crop/track must never take the whole camera's Re-ID
            # pass down — skip it, keep processing the rest of this frame
            # (same fail-soft contract as peopleid_worker.process_frame).
            track_result = {"track_id": person["track_id"], "bbox": person["bbox"], "state": "unknown", "person_id": None, "confidence": 0.0}
        track_results.append(track_result)

    state.fusion.prune_expired(now)
    state.prune_expired_tracks(now, config.REID_TRACK_TIMEOUT_SECONDS)
    return {"tracks": track_results}


def _process_one_track(camera_id: int, track_id: int, bbox: list[float], confidence: float, frame: np.ndarray, state: ReidWorkerState, now: float) -> dict:
    state.track_last_seen[track_id] = now
    quality = reid_quality.assess(frame, bbox, confidence, quality_min_score=state.camera_config.get("quality_min_score"))
    result = {"track_id": track_id, "bbox": [float(v) for v in bbox], "state": "unknown", "person_id": None, "confidence": 0.0}
    if not quality.passed:
        # Not treated as contradicting evidence for an already-CONFIRMED
        # track (mirrors peopleid_fusion's own "a single bad frame must not
        # undo a confirmed identity" rule) — just report its last-known
        # state back rather than forcing a decision from a bad crop.
        existing = state.fusion.get(camera_id, track_id)
        if existing is not None:
            result.update(state=existing.state, person_id=existing.person_id, confidence=round(existing.confidence, 3))
        return result

    embedding = state.embedder.embed(frame, bbox)
    if embedding is None:
        return result

    person_id, score, _margin = state.gallery.best_match(embedding, similarity_threshold=state.camera_config.get("similarity_threshold"))
    track_state = state.fusion.update_with_observation(camera_id, track_id, person_id, score, now=now)
    result.update(state=track_state.state, person_id=track_state.person_id, confidence=round(track_state.confidence, 3))

    if person_id is not None:
        # Matched — this track is no longer accumulating toward a NEW
        # identity (spec: don't auto-create a duplicate for someone who
        # already matched, even if the match only arrived after a few
        # earlier unmatched-looking cycles).
        state.pending_samples.pop(track_id, None)
        return result

    if track_id in state.enrolled_tracks:
        # This track already created an identity — see enrolled_tracks.
        # Keep reporting its fused state; never enrol it a second time.
        return result

    samples = state.pending_samples.setdefault(track_id, [])
    samples.append({
        "embedding": embedding, "quality_score": quality.composite_score,
        "quality_passed": True, "pose_label": "body", "crop_jpeg": _encode_small_jpeg(_crop(frame, bbox)),
    })
    if len(samples) >= config.REID_AUTO_ENROLL_MIN_OBSERVATIONS:
        curated = peopleid_enrollment.curate(samples, target_count=config.REID_ENROLLMENT_TARGET_EMBEDDINGS)
        del state.pending_samples[track_id]
        state.enrolled_tracks.add(track_id)
        # JSON/queue-safe payload for the main process to actually create
        # the person from (reid_db writes never happen in this process —
        # see this module's docstring) — same tolist()/bytes convention
        # peopleid_worker.py uses for its own unassigned_embedding field.
        result["pending_new_person"] = [
            {"embedding": s["embedding"].astype(np.float32).tolist(), "quality_score": s["quality_score"], "crop_jpeg": s["crop_jpeg"]}
            for s in curated
        ]
    return result
