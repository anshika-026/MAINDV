"""Local demo mode for the body-appearance Re-ID unique-footfall engine —
spec section 30: prove the core matching logic (detect -> track -> quality
gate -> embed -> gallery search -> temporal fusion -> auto-enrollment ->
unique footfall) against a local video file, no RTSP camera, no live
pipeline/API/frontend needed.

Deliberately mirrors production's own cadence-gated calling pattern (the
body tracker + Re-ID embedding pass runs every REID_MOT_INTERVAL_SECONDS,
not every frame — see reid_worker.py / config.py's own comment on why) by
converting that interval into a frame-skip count from the video's own FPS,
rather than processing every single frame, so timing behavior here is
representative of what a live camera would actually see.

Usage:
    python -m scripts.demo_reid --video test.mp4
    python -m scripts.demo_reid --video test.mp4 --show
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app import config, peopleid_enrollment, peopleid_fusion, peopleid_gallery, reid_db, reid_embedding, reid_quality  # noqa: E402
from app.peopleid_tracker import PersonBodyTracker  # noqa: E402

DEMO_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "reid_demo.db"
DEMO_CAMERA_ID = 0


def _encode_jpeg(crop: np.ndarray) -> bytes | None:
    if crop.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


def _crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


class DemoRunner:
    def __init__(self):
        self.tracker = PersonBodyTracker(config.PEOPLEID_MOT_MODEL_PATH, config.PEOPLEID_MOT_IMGSZ, config.PEOPLEID_MOT_CONFIDENCE)
        print(f"Loading Re-ID model ({config.REID_MODEL_NAME})...")
        self.embedder = reid_embedding.BodyReIdEmbedder(config.REID_MODEL_NAME, config.REID_MODEL_PATH, config.REID_DEVICE)
        self.gallery = peopleid_gallery.VectorGallery(
            embedding_loader=reid_db.load_all_embeddings,
            similarity_threshold=config.REID_SIMILARITY_THRESHOLD,
            min_margin=config.REID_MIN_MARGIN,
        )
        self.fusion = peopleid_fusion.IdentityFusion(
            window=config.REID_FUSION_WINDOW,
            candidate_min_votes=config.REID_CANDIDATE_MIN_VOTES,
            confirmed_min_votes=config.REID_CONFIRMED_MIN_VOTES,
            contradiction_limit=config.REID_CONFIRMED_CONTRADICTION_LIMIT,
            track_timeout_seconds=config.REID_TRACK_TIMEOUT_SECONDS,
        )
        # track_id -> accumulated quality-passing samples with no confident
        # gallery match yet — spec sections 2/7/31: after enough of these,
        # auto-create a new identity from the track's OWN crops, no human
        # ever assigns it (unlike peopleid_worker.py's unknown-cluster
        # review flow, which waits for a human).
        self._pending_samples: dict[int, list[dict]] = {}
        self._events_logged_this_track: set[int] = set()

    def process_track(self, camera_id: int, track_id: int, bbox: list[float], confidence: float, frame: np.ndarray, now: float) -> str:
        quality = reid_quality.assess(frame, bbox, confidence)
        if not quality.passed:
            return f"track {track_id}: low quality ({quality.reject_reason})"

        embedding = self.embedder.embed(frame, bbox)
        if embedding is None:
            return f"track {track_id}: embedding failed"

        person_id, score, margin = self.gallery.best_match(embedding)
        track_state = self.fusion.update_with_observation(camera_id, track_id, person_id, score, now=now)

        if person_id is None:
            samples = self._pending_samples.setdefault(track_id, [])
            samples.append({
                "embedding": embedding, "quality_score": quality.composite_score,
                "quality_passed": True, "pose_label": "body", "crop_jpeg": _encode_jpeg(_crop(frame, bbox)),
            })
            if len(samples) >= config.REID_AUTO_ENROLL_MIN_OBSERVATIONS:
                new_person_id = self._auto_enroll(camera_id, track_id, samples, now)
                del self._pending_samples[track_id]
                return f"track {track_id}: NEW identity created -> PERSON_{new_person_id:03d}"
            return f"track {track_id}: unmatched ({len(samples)}/{config.REID_AUTO_ENROLL_MIN_OBSERVATIONS} observations before auto-enroll)"

        if track_state.state == peopleid_fusion.STATE_CONFIRMED:
            reid_db.touch_person(person_id, now=now)
            key = (track_id, person_id)
            if key not in self._events_logged_this_track:
                self._events_logged_this_track.add(key)
                reid_db.log_event(camera_id, track_id, reid_db.EVENT_SIGHTING, person_id=person_id, confidence=score, now=now)
            self._pending_samples.pop(track_id, None)
            person = reid_db.get_person(person_id)
            label = person["label"] if person else f"PERSON_{person_id:03d}"
            return f"track {track_id}: CONFIRMED {label} (score={score:.3f}, margin={margin:.3f})"

        return f"track {track_id}: candidate match person_id={person_id} (score={score:.3f}, state={track_state.state})"

    def _auto_enroll(self, camera_id: int, track_id: int, samples: list[dict], now: float) -> int:
        curated = peopleid_enrollment.curate(samples, target_count=config.REID_ENROLLMENT_TARGET_EMBEDDINGS)
        person_id = reid_db.create_person(now=now)
        reid_db.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(curated):
            reid_db.add_embedding(person_id, sample["embedding"], quality_score=sample["quality_score"], source_camera=camera_id)
            if sample.get("crop_jpeg"):
                filename = f"person_{person_id}_{i}_{int(now)}.jpg"
                (reid_db.SNAPSHOTS_DIR / filename).write_bytes(sample["crop_jpeg"])
                reid_db.add_snapshot(person_id, camera_id, filename, quality_score=sample["quality_score"])
        reid_db.log_event(camera_id, track_id, reid_db.EVENT_NEW_PERSON, person_id=person_id, now=now)
        self.gallery.reload()
        return person_id


def main():
    parser = argparse.ArgumentParser(description="Demo the body Re-ID unique-footfall engine against a local video file")
    parser.add_argument("--video", required=True, help="Path to a video file (or 0 for the default webcam)")
    parser.add_argument("--show", action="store_true", help="Show a live preview window with track boxes")
    parser.add_argument("--reset", action="store_true", help="Clear the demo database before running (default: keep accumulating across runs, same as production)")
    parser.add_argument("--max-seconds", type=float, default=None, help="Stop after this many wall-clock seconds — needed for an RTSP URL, which has no natural end")
    args = parser.parse_args()

    if args.reset and DEMO_DB_PATH.exists():
        DEMO_DB_PATH.unlink()
        print(f"Cleared {DEMO_DB_PATH}")

    reid_db.DB_PATH = DEMO_DB_PATH
    reid_db.init_db()

    source = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {args.video}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip = max(1, round(fps * config.REID_MOT_INTERVAL_SECONDS))
    print(f"Video FPS={fps:.1f}, sampling every {frame_skip} frame(s) (~{config.REID_MOT_INTERVAL_SECONDS}s cadence)")

    runner = DemoRunner()
    frame_idx = 0
    wall_start = time.time()
    try:
        while True:
            if args.max_seconds is not None and time.time() - wall_start > args.max_seconds:
                print(f"Reached --max-seconds={args.max_seconds}, stopping.")
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            now = time.time()
            people = runner.tracker.update(frame)
            for person in people:
                msg = runner.process_track(DEMO_CAMERA_ID, person["track_id"], person["bbox"], person["confidence"], frame, now)
                print(f"[t={now - wall_start:6.1f}s] {msg}")
                if args.show:
                    x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

            if args.show:
                cv2.imshow("demo_reid", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    print("\n--- Summary ---")
    persons = reid_db.list_persons(status=None)
    print(f"Total identities created: {len(persons)}")
    for p in persons:
        print(f"  {p['label']}: first_seen={p['first_seen']:.0f}, last_seen={p['last_seen']:.0f}, "
              f"embeddings={p['embedding_count']}, snapshots={p['snapshot_count']}")
    print(f"Unique footfall today: {reid_db.count_unique_today()}")
    print(f"Unique footfall lifetime: {reid_db.count_unique_lifetime()}")


if __name__ == "__main__":
    main()
