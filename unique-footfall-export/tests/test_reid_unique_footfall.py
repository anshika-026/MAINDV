"""Unique-footfall identity tests — the spec's PRIMARY SUCCESS CRITERION.

    Person A enters 10 times  -> PERSON_001, unique footfall = 1
    Person B enters           -> PERSON_002, unique footfall = 2
    Person A returns          -> PERSON_001, unique footfall STAYS 2

These exercise the real components the live engine uses — peopleid_gallery's
vector search, peopleid_fusion's temporal confirmation, and reid_db's
counting against a real throwaway SQLite file (same no-mocking-the-DB
convention as conftest.temp_db). Only the embeddings themselves are
synthetic, so the tests are deterministic and need no model download or
camera: reid_embedding's job (turning a crop into a discriminative vector)
is measured separately by scripts/eval_reid_weights.py against real crops,
which is a benchmark, not a unit test.

What these lock down is the logic AROUND the embedding — the part where
"same person seen 10 times" must not become 10 counts. That logic is what
regressed in practice (a track that fails the decisiveness margin spawns a
fresh identity, so footfall counted appearances instead of people).
"""

from __future__ import annotations

import numpy as np
import pytest

from app import config, peopleid_db, reid_db
from app.peopleid_fusion import IdentityFusion
from app.peopleid_gallery import VectorGallery

DIM = 512

# Measured on this deployment's own camera with the MSMT17 checkpoint
# (scripts/eval_reid_weights.py + visually-verified crop pairs): two views
# of the SAME person score ~0.94, two DIFFERENT people ~0.45. Tests below
# synthesize embeddings at exactly these similarities so they encode the
# real operating point rather than arbitrary numbers.
SAME_PERSON_SIMILARITY = 0.94
DIFFERENT_PERSON_SIMILARITY = 0.45


@pytest.fixture
def reid_temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_reid.db"
    monkeypatch.setattr(reid_db, "DB_PATH", db_path)
    monkeypatch.setattr(reid_db, "SNAPSHOTS_DIR", tmp_path / "snaps")
    reid_db.init_db()
    return db_path


def person_embedding(seed: int) -> np.ndarray:
    """A stable unit vector standing in for one person's appearance."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def observed(base: np.ndarray, similarity: float, seed: int) -> np.ndarray:
    """Another view of `base` at an EXACT cosine similarity.

    Built by mixing in a component orthogonal to `base` rather than adding
    isotropic noise: in 512 dimensions a unit vector's components are only
    ~0.044, so even "small" isotropic noise swamps the signal (sigma=0.05
    drops similarity to 0.69, i.e. not the same person at all). Constructing
    the angle directly makes each test state the similarity it means.
    """
    rng = np.random.RandomState(seed)
    noise = rng.randn(DIM).astype(np.float32)
    noise -= float(noise @ base) * base  # keep only the orthogonal part
    noise /= np.linalg.norm(noise)
    vec = similarity * base + np.sqrt(1.0 - similarity**2) * noise
    return vec / np.linalg.norm(vec)


def make_gallery(threshold=0.75, min_margin=0.05, duplicate_identities_expected=True) -> VectorGallery:
    """Defaults match how reid_worker builds it (auto-enrolled gallery)."""
    return VectorGallery(
        embedding_loader=reid_db.load_all_embeddings,
        similarity_threshold=threshold,
        min_margin=min_margin,
        dim=DIM,
        duplicate_identities_expected=duplicate_identities_expected,
    )


def make_fusion() -> IdentityFusion:
    return IdentityFusion(
        window=5, candidate_min_votes=2, confirmed_min_votes=3,
        contradiction_limit=2, track_timeout_seconds=20.0,
    )


def enroll(embeddings: list[np.ndarray], camera_id: int = 1) -> int:
    """Mirrors pipeline._auto_create_reid_person: new identity + its
    embeddings, which is the ONLY thing that may increment unique footfall."""
    person_id = reid_db.create_person()
    for emb in embeddings:
        reid_db.add_embedding(person_id, emb, quality_score=0.9, source_camera=camera_id)
    return person_id


def confirm_identity(fusion, gallery, camera_id, track_id, embedding, frames=3, t0=1000.0):
    """Drives one track through the real fusion state machine until it
    confirms, returning the settled (state, person_id)."""
    state = None
    for i in range(frames):
        pid, score, _margin = gallery.best_match(embedding)
        state = fusion.update_with_observation(camera_id, track_id, pid, score, now=t0 + i)
    return state


# --- Spec section 29: identity ------------------------------------------

def test_same_embedding_matches_same_person(reid_temp_db):
    a = person_embedding(1)
    pid = enroll([a])
    gallery = make_gallery()
    matched, score, _ = gallery.best_match(observed(a, SAME_PERSON_SIMILARITY, seed=99))
    assert matched == pid
    assert score >= 0.75


def test_different_embedding_does_not_match(reid_temp_db):
    enroll([person_embedding(1)])
    gallery = make_gallery()
    matched, _score, _ = gallery.best_match(person_embedding(2))
    assert matched is None


# --- Spec section 21: identity-merging protection ------------------------

def test_ambiguous_match_below_threshold_resolves_to_unknown(reid_temp_db):
    """The 0.61/0.60/0.59 case the margin rule exists for: several
    candidates are close AND none is convincing, so the answer is Unknown
    rather than the nominally-highest one. Holds in both gallery modes."""
    base = person_embedding(7)
    enroll([observed(base, 0.60, seed=11)])
    enroll([observed(base, 0.59, seed=12)])
    for duplicates_expected in (False, True):
        gallery = make_gallery(duplicate_identities_expected=duplicates_expected)
        matched, score, _margin = gallery.best_match(base)
        assert matched is None, f"unconvincing near-tie must be Unknown (duplicates={duplicates_expected})"
        assert score < 0.75


def test_near_tie_between_duplicate_identities_matches_instead_of_fragmenting(reid_temp_db):
    """The bug that made unique footfall over-count.

    Auto-enrolment can split one person across several identities. A later
    crop of that person then matches BOTH fragments almost equally, and the
    margin rule rejected it as "ambiguous" — minting a THIRD fragment, which
    made the next near-tie even more likely. Measured on this deployment's
    own crops, that loop rejected 94 of 186 crops and produced 131
    identities for a handful of real people; matching the best candidate
    instead brought the same data to 37.

    The face gallery must keep the strict behaviour: there, two identities
    really are two different people, so a near-tie is genuine ambiguity.
    """
    base = person_embedding(7)
    enroll([observed(base, 0.97, seed=11)])  # same person, fragment A
    enroll([observed(base, 0.96, seed=12)])  # same person, fragment B

    reid_style = make_gallery(duplicate_identities_expected=True)
    matched, score, margin = reid_style.best_match(base)
    assert matched is not None, "a confident near-tie must not spawn another fragment"
    assert score >= 0.75
    assert margin < 0.05, "this is precisely the near-tie case"

    face_style = make_gallery(duplicate_identities_expected=False)
    assert face_style.best_match(base)[0] is None, "face gallery must stay strict"


def test_single_frame_does_not_confirm_an_identity(reid_temp_db):
    """Spec section 9/31: never trust one frame."""
    a = person_embedding(1)
    pid = enroll([a])
    gallery, fusion = make_gallery(), make_fusion()
    match, score, _ = gallery.best_match(a)
    state = fusion.update_with_observation(1, 17, match, score, now=1000.0)
    assert state.state != peopleid_db.TRACK_STATE_CONFIRMED
    assert pid is not None


def test_inconsistent_votes_do_not_confirm(reid_temp_db):
    """Flapping matches (A, B, A, C) must not settle into an identity."""
    pid_a, pid_b, pid_c = enroll([person_embedding(1)]), enroll([person_embedding(2)]), enroll([person_embedding(3)])
    fusion = make_fusion()
    state = None
    for i, pid in enumerate([pid_a, pid_b, pid_a, pid_c]):
        state = fusion.update_with_observation(1, 17, pid, 0.80, now=1000.0 + i)
    assert state.state != peopleid_db.TRACK_STATE_CONFIRMED


# --- One track may only ever create ONE identity -------------------------

def test_a_single_track_cannot_enrol_twice(reid_temp_db, monkeypatch):
    """Observed live: track 81 created PERSON_017 and PERSON_018 twelve
    seconds apart. Auto-enrolment fires once a track has N unmatched
    samples and then clears them, so a track that still doesn't match
    starts over and mints a second identity for the same person. One
    continuous track is one person, so it must enrol at most once.
    """
    from app import reid_worker

    state = reid_worker.ReidWorkerState.__new__(reid_worker.ReidWorkerState)
    state.pending_samples = {}
    state.enrolled_tracks = set()
    state.track_last_seen = {}

    track_id = 81
    # First enrolment: accumulate to the threshold, then emit.
    for _ in range(config.REID_AUTO_ENROLL_MIN_OBSERVATIONS):
        state.pending_samples.setdefault(track_id, []).append({"embedding": person_embedding(1)})
    assert len(state.pending_samples[track_id]) >= config.REID_AUTO_ENROLL_MIN_OBSERVATIONS
    del state.pending_samples[track_id]
    state.enrolled_tracks.add(track_id)

    # The same track keeps failing to match — it must NOT enrol again.
    assert track_id in state.enrolled_tracks, "track must be remembered as already-enrolled"
    assert track_id not in state.pending_samples

    # Once the track has been gone longer than the timeout, the bookkeeping
    # is released so the id can't leak for the life of the process.
    state.track_last_seen[track_id] = 1000.0
    state.prune_expired_tracks(1000.0 + config.REID_TRACK_TIMEOUT_SECONDS + 1, config.REID_TRACK_TIMEOUT_SECONDS)
    assert track_id not in state.enrolled_tracks
    assert track_id not in state.track_last_seen


# --- Spec section 29: tracker ID change ----------------------------------

def test_tracker_id_change_keeps_same_person_and_footfall(reid_temp_db):
    """Track 17 -> PERSON_001, track disappears, track 82 returns as the
    same human -> still PERSON_001, unique footfall stays 1."""
    a = person_embedding(1)
    pid = enroll([a, observed(a, SAME_PERSON_SIMILARITY, seed=21)])
    gallery, fusion = make_gallery(), make_fusion()

    first = confirm_identity(fusion, gallery, 1, 17, observed(a, SAME_PERSON_SIMILARITY, seed=31), t0=1000.0)
    assert first.person_id == pid
    assert first.state == peopleid_db.TRACK_STATE_CONFIRMED

    # Same human returns under a brand-new tracker id.
    second = confirm_identity(fusion, gallery, 1, 82, observed(a, SAME_PERSON_SIMILARITY, seed=32), t0=2000.0)
    assert second.person_id == pid, "a new track id must not create a new identity"
    assert reid_db.count_unique_lifetime() == 1


# --- PRIMARY SUCCESS CRITERION -------------------------------------------

def test_primary_success_criterion(reid_temp_db):
    """Person A x10 -> 1. Person B -> 2. Person A returns -> still 2."""
    a, b = person_embedding(1), person_embedding(2)
    pid_a = enroll([a, observed(a, SAME_PERSON_SIMILARITY, seed=41)])
    gallery, fusion = make_gallery(), make_fusion()

    # Person A enters ten times, each under a different tracker id.
    for visit in range(10):
        state = confirm_identity(
            fusion, gallery, 1, track_id=100 + visit,
            embedding=observed(a, SAME_PERSON_SIMILARITY, seed=200 + visit), t0=1000.0 + visit * 60,
        )
        assert state.person_id == pid_a, f"visit {visit} should still be PERSON_001"
    assert reid_db.count_unique_lifetime() == 1, "ten visits by one person is one unique person"

    # Person B is genuinely new: no match, so a new identity is created.
    matched, _score, _ = gallery.best_match(b)
    assert matched is None, "a new person must not match an existing identity"
    pid_b = enroll([b])
    gallery.reload()
    assert reid_db.count_unique_lifetime() == 2

    state_b = confirm_identity(fusion, gallery, 1, 300, observed(b, SAME_PERSON_SIMILARITY, seed=77), t0=5000.0)
    assert state_b.person_id == pid_b

    # Person A returns after B — must resolve to the EXISTING identity.
    state_a = confirm_identity(fusion, gallery, 1, 400, observed(a, SAME_PERSON_SIMILARITY, seed=88), t0=6000.0)
    assert state_a.person_id == pid_a
    assert reid_db.count_unique_lifetime() == 2, "a returning person must not increment footfall"


def test_daily_unique_footfall_counts_people_not_appearances(reid_temp_db):
    """Spec section 12/32: only a NEW confirmed person_id increments.

    "Unique footfall today" counts people SEEN today, so it's driven by
    sighting events, not by how many identities exist — an identity
    enrolled but not seen today correctly doesn't count toward today.
    """
    pid_a = enroll([person_embedding(1)])
    for track_id in range(10):  # ten separate appearances, ten tracker ids
        reid_db.log_event(1, track_id=track_id, event_type=reid_db.EVENT_SIGHTING, person_id=pid_a, confidence=0.9)
    assert reid_db.count_unique_today() == 1, "ten sightings of one person is still one"

    pid_b = enroll([person_embedding(2)])
    reid_db.log_event(1, track_id=99, event_type=reid_db.EVENT_SIGHTING, person_id=pid_b, confidence=0.9)
    assert reid_db.count_unique_today() == 2


def test_identity_enrolled_but_not_seen_today_is_not_todays_footfall(reid_temp_db):
    """Yesterday's visitor who didn't come back must not inflate today."""
    pid_old = enroll([person_embedding(5)])
    yesterday = reid_db._day_start() - 3600  # one hour before today began
    reid_db.log_event(1, 1, reid_db.EVENT_SIGHTING, person_id=pid_old, confidence=0.9, now=yesterday)
    assert reid_db.count_unique_today() == 0
    assert reid_db.count_unique_lifetime() == 1
