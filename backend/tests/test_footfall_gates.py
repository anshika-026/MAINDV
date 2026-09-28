"""Cross-gate unique footfall: every gate shares one identity gallery, so a
person counts once however many gates they're seen at.

Drives the real FootfallService / reid_worker / gallery / fusion / reid_db
code against a throwaway SQLite file. Only the person detector and the
Re-ID embedder are faked (no camera or model download needed), using the
same synthetic-embedding approach as unique-footfall-export's own tests:
two views of one person at cosine 0.94, different people ~0 (well under
the measured real-world 0.45).

Run from backend/:  python -m pytest tests -v
"""

import numpy as np
import pytest

from app import camera_db, footfall
from app.reid import config as reid_config
from app.reid import reid_db

DIM = 512
SAME_PERSON_SIMILARITY = 0.94
GATE_1, GATE_2, GATE_3 = 101, 102, 103


def person_embedding(seed: int) -> np.ndarray:
    v = np.random.RandomState(seed).randn(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def another_view(base: np.ndarray, seed: int) -> np.ndarray:
    noise = np.random.RandomState(seed).randn(DIM).astype(np.float32)
    noise -= float(noise @ base) * base
    noise /= np.linalg.norm(noise)
    v = SAME_PERSON_SIMILARITY * base + np.sqrt(1 - SAME_PERSON_SIMILARITY**2) * noise
    return v / np.linalg.norm(v)


class FakeScene:
    """What each gate currently "sees": track_id -> person seed."""

    def __init__(self):
        self.visible: dict[int, int] = {}
        self._view_seed = 10_000

    @staticmethod
    def bbox_for(track_id: int) -> list[float]:
        x = 20 + 150 * (track_id % 3)
        return [x, 50, x + 110, 330]  # passes the size/aspect quality gate

    def next_view_seed(self) -> int:
        self._view_seed += 1
        return self._view_seed


class FakeTracker:
    def __init__(self, scene: FakeScene):
        self.scene = scene

    def update(self, frame):
        return [{"track_id": tid, "bbox": FakeScene.bbox_for(tid), "confidence": 0.9} for tid in self.scene.visible]


class FakeEmbedder:
    def __init__(self, scene: FakeScene):
        self.scene = scene

    def embed(self, frame, bbox):
        for tid, person in self.scene.visible.items():
            if FakeScene.bbox_for(tid) == bbox:
                return another_view(person_embedding(person), self.scene.next_view_seed())
        return None


# Textured mid-grey noise: sharp and well-lit enough to pass reid_quality.
FRAME = np.random.RandomState(0).randint(60, 200, (400, 520, 3)).astype(np.uint8)


@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.setattr(reid_db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(reid_db, "SNAPSHOTS_DIR", tmp_path / "snaps")
    monkeypatch.setattr(reid_config, "REID_MOT_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(camera_db, "list_cameras", lambda: [
        {"id": GATE_1, "name": "Gate 1", "purpose": "Entry/Exit", "host": "x"},
        {"id": GATE_2, "name": "Gate 2", "purpose": "Entry/Exit", "host": "x"},
        {"id": GATE_3, "name": "Gate 3", "purpose": "Entry/Exit", "host": "x"},
        {"id": 7, "name": "Lobby", "purpose": "GENERAL", "host": "x"},
    ])
    reid_db.init_db()
    service = footfall.FootfallService()
    service._gallery = footfall._SharedGallery()
    service.scenes = {}
    for cid in (GATE_1, GATE_2, GATE_3):
        runner = footfall._GateRunner(cid, service._gallery)
        scene = FakeScene()
        runner.state.tracker = FakeTracker(scene)
        runner.state.embedder = FakeEmbedder(scene)
        service._runners[cid] = runner
        service.scenes[cid] = scene
    return service


def walk_past(service, gate: int, person: int, track_id: int, frames: int = 6):
    scene = service.scenes[gate]
    scene.visible = {track_id: person}
    for _ in range(frames):
        service._process(service._runners[gate], FRAME)
    scene.visible = {}


def per_gate(service) -> dict:
    return {g["name"]: g["unique_today"] for g in service.summary()["gates"]}


def test_in_at_gate_1_out_at_gate_3_is_one_person(svc):
    walk_past(svc, GATE_1, person=1, track_id=1)
    walk_past(svc, GATE_3, person=1, track_id=1)
    s = svc.summary()
    assert s["unique_today"] == 1
    assert per_gate(svc) == {"Gate 1": 1, "Gate 2": 0, "Gate 3": 1}
    assert s["visitors"][0]["gates"] and set(s["visitors"][0]["gates"]) == {"Gate 1", "Gate 3"}


def test_coming_back_through_another_gate_is_still_one(svc):
    walk_past(svc, GATE_1, person=1, track_id=1)
    walk_past(svc, GATE_3, person=1, track_id=1)
    walk_past(svc, GATE_2, person=1, track_id=5)
    walk_past(svc, GATE_1, person=1, track_id=9)
    assert svc.summary()["unique_today"] == 1
    assert reid_db.count_unique_lifetime() == 1


def test_different_people_at_different_gates_are_counted_separately(svc):
    walk_past(svc, GATE_1, person=1, track_id=1)
    walk_past(svc, GATE_2, person=2, track_id=1)
    walk_past(svc, GATE_3, person=3, track_id=1)
    walk_past(svc, GATE_3, person=1, track_id=2)  # person 1 leaving
    s = svc.summary()
    assert s["unique_today"] == 3
    assert per_gate(svc) == {"Gate 1": 1, "Gate 2": 1, "Gate 3": 2}
    assert s["avg_per_gate"] == pytest.approx(4 / 3, abs=0.05)


def test_same_person_at_two_gates_at_once_is_one(svc):
    """Both gates see a brand-new person in the same seconds, interleaved —
    the second gate must match the identity the first one just enrolled
    instead of minting its own."""
    svc.scenes[GATE_1].visible = {1: 1}
    svc.scenes[GATE_2].visible = {1: 1}
    for _ in range(6):
        svc._process(svc._runners[GATE_1], FRAME)
        svc._process(svc._runners[GATE_2], FRAME)
    assert svc.summary()["unique_today"] == 1
    assert per_gate(svc) == {"Gate 1": 1, "Gate 2": 1, "Gate 3": 0}


def test_enroll_recheck_matches_identity_created_by_another_gate(svc):
    """Directly exercises _enroll's last-moment recheck: samples collected
    before another gate enrolled the same person must not create a second."""
    walk_past(svc, GATE_1, person=1, track_id=1)
    samples = [
        {"embedding": another_view(person_embedding(1), 500 + i).tolist(), "quality_score": 0.8, "crop_jpeg": None}
        for i in range(3)
    ]
    pid = svc._enroll(GATE_2, {"track_id": 42, "confidence": 0.9}, samples, now=__import__("time").time())
    assert reid_db.count_unique_lifetime() == 1
    assert pid == reid_db.list_persons()[0]["id"]


def test_non_gate_cameras_are_not_in_the_breakdown(svc):
    assert [g["name"] for g in svc.summary()["gates"]] == ["Gate 1", "Gate 2", "Gate 3"]


def test_reset_restarts_the_count_and_numbering(svc):
    walk_past(svc, GATE_1, person=1, track_id=1)
    walk_past(svc, GATE_2, person=2, track_id=1)
    assert svc.summary()["unique_today"] == 2

    assert svc.reset()["people_removed"] == 2
    s = svc.summary()
    assert s["unique_today"] == 0 and s["visitors"] == [] and svc.people() == []

    # Runners were dropped by reset; rebuild them like feed() would.
    for cid, scene in svc.scenes.items():
        runner = footfall._GateRunner(cid, svc._gallery, svc._generation)
        runner.state.tracker, runner.state.embedder = FakeTracker(scene), FakeEmbedder(scene)
        svc._runners[cid] = runner
    walk_past(svc, GATE_3, person=1, track_id=7)
    people = svc.people()
    assert svc.summary()["unique_today"] == 1
    assert people[0]["label"] == "PERSON_001" and people[0]["gates"] == ["Gate 3"]


def test_result_computed_before_reset_is_discarded(svc):
    old_generation = svc._generation
    svc.reset()
    stale = {"tracks": [{"track_id": 1, "bbox": [0, 0, 1, 1], "state": "unknown", "person_id": None,
                         "confidence": 0.9, "pending_new_person": [
                             {"embedding": person_embedding(9).tolist(), "quality_score": 0.8, "crop_jpeg": None}]}]}
    svc._record(GATE_1, stale, old_generation)
    assert svc.summary()["unique_today"] == 0


def rebuild_runner(svc, cid):
    runner = footfall._GateRunner(cid, svc._gallery, svc._generation)
    scene = svc.scenes[cid]
    runner.state.tracker, runner.state.embedder = FakeTracker(scene), FakeEmbedder(scene)
    svc._runners[cid] = runner


def test_poorly_covered_view_of_a_confirmed_person_is_learned(svc):
    walk_past(svc, GATE_1, person=1, track_id=1)
    pid = reid_db.list_persons()[0]["id"]
    before = reid_db.embedding_count_for_person(pid)
    track = {"track_id": 3, "state": "confirmed", "person_id": pid, "confidence": 0.8,
             "match": {"person_id": pid, "score": 0.78, "quality_score": 0.7,
                       "embedding": another_view(person_embedding(1), 777).tolist()}}
    svc._record(GATE_3, {"tracks": [track]})
    assert reid_db.embedding_count_for_person(pid) == before + 1
    # A view the gallery already covers well is not stored again.
    track["match"]["score"] = 0.95
    track["track_id"] = 4
    svc._record(GATE_3, {"tracks": [track]})
    assert reid_db.embedding_count_for_person(pid) == before + 1


def test_people_outside_the_counting_zone_are_ignored(svc):
    # FakeScene puts track 1 at x=170..280 of a 520px-wide frame (centre ~0.43).
    svc.set_zone(GATE_1, [[0.0, 0.0], [0.3, 0.0], [0.3, 1.0], [0.0, 1.0]])  # left 30% only
    rebuild_runner(svc, GATE_1)
    walk_past(svc, GATE_1, person=1, track_id=1)
    assert svc.summary()["unique_today"] == 0
    svc.set_zone(GATE_1, None)  # whole frame again
    rebuild_runner(svc, GATE_1)
    walk_past(svc, GATE_1, person=1, track_id=4)
    assert svc.summary()["unique_today"] == 1
