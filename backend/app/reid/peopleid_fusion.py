"""Temporal identity fusion for the People Identification module — the
piece that stops a single noisy frame from deciding (or flipping) a
track's identity. Spec requirement: "Anshika, Anshika, Unknown, Anshika,
Anshika" across five observations must resolve to a stable, HIGH-CONFIDENCE
Anshika, never flicker.

State machine per (camera_id, track_id): UNKNOWN -> CANDIDATE -> CONFIRMED.
- CANDIDATE requires PEOPLEID_CANDIDATE_MIN_VOTES agreeing, quality-passing
  face observations within the rolling PEOPLEID_FUSION_WINDOW.
- CONFIRMED requires PEOPLEID_CONFIRMED_MIN_VOTES.
- Dropping a CONFIRMED identity requires PEOPLEID_CONFIRMED_CONTRADICTION_LIMIT
  CONSECUTIVE quality-passing observations naming a DIFFERENT person — a
  single "Unknown" (no confident gallery match) observation never demotes a
  CONFIRMED identity by itself; the whole point of this module is that one
  bad frame must not undo an established identity.
- Re-ID (peopleid_reid.py) can ONLY bridge an already-CONFIRMED track
  through a face-absent stretch (its own reid_bridge_seconds) — it can
  never independently create or promote an identity, enforced here by the
  bridge method simply refusing to touch anything but an already-CONFIRMED
  track with a matching person_id already set.

Parameterized over its window/vote/timeout thresholds (constructor args
below) rather than reading config.PEOPLEID_* directly: reid_worker.py (the
body-appearance Re-ID engine) reuses this exact state machine with its own
REID_* thresholds — the confirmation/contradiction-protection LOGIC is
identical between a face-driven and a body-driven identity signal, only the
tuning differs per caller. bridge_with_reid's own args (reid_enabled etc.)
are similarly parameterized; reid_worker.py simply never calls that method
(its primary embedding already IS the appearance signal a short-term bridge
would otherwise approximate), so those defaults never take effect there.
"""

from __future__ import annotations

import time
from collections import deque

from . import peopleid_db, peopleid_reid

STATE_UNKNOWN = peopleid_db.TRACK_STATE_UNKNOWN
STATE_CANDIDATE = peopleid_db.TRACK_STATE_CANDIDATE
STATE_CONFIRMED = peopleid_db.TRACK_STATE_CONFIRMED


class _Observation:
    __slots__ = ("ts", "person_id", "score")

    def __init__(self, ts: float, person_id: int | None, score: float):
        self.ts = ts
        self.person_id = person_id
        self.score = score


class TrackState:
    __slots__ = (
        "camera_id", "track_id", "state", "person_id", "confidence",
        "observations", "contradiction_streak", "reid_signature",
        "last_face_seen_at", "last_update_at",
    )

    def __init__(self, camera_id: int, track_id: int, now: float, window: int):
        self.camera_id = camera_id
        self.track_id = track_id
        self.state = STATE_UNKNOWN
        self.person_id: int | None = None
        self.confidence = 0.0
        self.observations: deque[_Observation] = deque(maxlen=window)
        self.contradiction_streak = 0
        self.reid_signature = None
        self.last_face_seen_at = now
        self.last_update_at = now

    def _votes_for(self, person_id: int) -> int:
        return sum(1 for o in self.observations if o.person_id == person_id)

    def _top_candidate(self) -> tuple[int | None, int]:
        counts: dict[int, int] = {}
        for o in self.observations:
            if o.person_id is not None:
                counts[o.person_id] = counts.get(o.person_id, 0) + 1
        if not counts:
            return None, 0
        top = max(counts.items(), key=lambda kv: kv[1])
        return top


class IdentityFusion:
    def __init__(
        self,
        window: int,
        candidate_min_votes: int,
        confirmed_min_votes: int,
        contradiction_limit: int,
        track_timeout_seconds: float,
        reid_enabled: bool = False,
        reid_bridge_seconds: float = 0.0,
        reid_similarity_threshold: float = 1.0,
    ):
        self._tracks: dict[tuple[int, int], TrackState] = {}
        self._window = window
        self._candidate_min_votes = candidate_min_votes
        self._confirmed_min_votes = confirmed_min_votes
        self._contradiction_limit = contradiction_limit
        self._track_timeout_seconds = track_timeout_seconds
        # bridge_with_reid-only settings — irrelevant to callers (like
        # reid_worker.py) that never call that method.
        self._reid_enabled = reid_enabled
        self._reid_bridge_seconds = reid_bridge_seconds
        self._reid_similarity_threshold = reid_similarity_threshold

    def _get_or_create(self, camera_id: int, track_id: int, now: float) -> TrackState:
        key = (camera_id, track_id)
        track = self._tracks.get(key)
        if track is None:
            track = TrackState(camera_id, track_id, now, self._window)
            self._tracks[key] = track
        return track

    def update_with_observation(
        self, camera_id: int, track_id: int, person_id: int | None, score: float,
        reid_signature=None, now: float | None = None,
    ) -> TrackState:
        """Called once per quality-passing identity observation — a face
        match for peopleid_worker.py, a body-appearance match for
        reid_worker.py (the caller must not call this for a quality
        -rejected observation; this class trusts every observation it
        receives already cleared the quality gate). `reid_signature` is
        peopleid_worker.py's own short-term HSV-bridge concept (see
        bridge_with_reid) — reid_worker.py simply never passes one."""
        now = now if now is not None else time.time()
        track = self._get_or_create(camera_id, track_id, now)
        track.last_face_seen_at = now
        track.last_update_at = now
        track.observations.append(_Observation(now, person_id, score))
        if reid_signature is not None:
            track.reid_signature = reid_signature

        top_person, top_votes = track._top_candidate()

        if track.state == STATE_UNKNOWN:
            if top_person is not None and top_votes >= self._candidate_min_votes:
                track.state = STATE_CANDIDATE
                track.person_id = top_person
                track.confidence = score
        elif track.state == STATE_CANDIDATE:
            if top_person is not None and top_votes >= self._confirmed_min_votes:
                track.state = STATE_CONFIRMED
                track.person_id = top_person
                track.confidence = score
                track.contradiction_streak = 0
            elif top_person is not None and top_votes >= self._candidate_min_votes:
                # Pre-confirmation, the leading candidate can still shift —
                # only CONFIRMED gets the stronger contradiction-streak guard.
                track.person_id = top_person
                track.confidence = score
            elif top_person is None:
                track.state = STATE_UNKNOWN
                track.person_id = None
        elif track.state == STATE_CONFIRMED:
            if person_id is None:
                # A quality-passing face that simply didn't match anyone
                # confidently — NOT treated as contradicting evidence (spec:
                # a single bad frame must not undo a confirmed identity).
                pass
            elif person_id == track.person_id:
                track.contradiction_streak = 0
                track.confidence = max(track.confidence, score)
            else:
                track.contradiction_streak += 1
                if track.contradiction_streak >= self._contradiction_limit:
                    track.state = STATE_CANDIDATE
                    track.person_id = person_id
                    track.confidence = score
                    track.contradiction_streak = 0
                    track.observations.clear()
                    track.observations.append(_Observation(now, person_id, score))

        return track

    def bridge_with_reid(self, camera_id: int, track_id: int, current_signature, now: float | None = None) -> TrackState | None:
        """Face is unavailable/quality-rejected this cycle. Only ever keeps
        an ALREADY-CONFIRMED track's existing identity alive a bit longer —
        never creates, promotes, or changes an identity. Returns the track
        if the bridge held, or None if there's nothing to bridge (no track,
        not confirmed, no stored signature, similarity too low, or the
        bridge window has expired — caller should then treat identity as
        lost for this cycle, not force a decision)."""
        if not self._reid_enabled:
            return None
        now = now if now is not None else time.time()
        track = self._tracks.get((camera_id, track_id))
        if track is None or track.state != STATE_CONFIRMED or track.reid_signature is None:
            return None
        if now - track.last_face_seen_at > self._reid_bridge_seconds:
            return None
        if current_signature is None:
            return None
        sim = peopleid_reid.similarity(track.reid_signature, current_signature)
        if sim < self._reid_similarity_threshold:
            return None
        track.last_update_at = now
        return track

    def get(self, camera_id: int, track_id: int) -> TrackState | None:
        return self._tracks.get((camera_id, track_id))

    def prune_expired(self, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        expired = [
            key for key, t in self._tracks.items()
            if now - t.last_update_at > self._track_timeout_seconds
        ]
        for key in expired:
            del self._tracks[key]

    def drop(self, camera_id: int, track_id: int) -> None:
        self._tracks.pop((camera_id, track_id), None)
