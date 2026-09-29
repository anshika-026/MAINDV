"""Identity hand-off by body appearance (app/appearance.py + the overlay's use
of it in face_pipeline.py). Embeddings are synthetic: two views of one person
at cosine ~0.9, different people ~0."""

import datetime
import time

import numpy as np
import pytest

from app import analytics_settings, appearance
from app.appearance import AppearanceGallery


def person(seed):
    v = np.random.RandomState(seed).randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def view(seed, sim=0.9, noise_seed=99):
    base = person(seed)
    n = np.random.RandomState(noise_seed).randn(512).astype(np.float32)
    n -= float(n @ base) * base
    n /= np.linalg.norm(n)
    return sim * base + np.sqrt(1 - sim ** 2) * n


def test_matches_only_people_recognised_today_and_only_clearly():
    g = AppearanceGallery()
    now = time.time()
    g.add_embedding("014", person(14), now)
    g.add_embedding("038", person(38), now)
    assert g.match_embedding(view(14), now=now) == ("014", pytest.approx(0.9, abs=0.01))
    assert g.match_embedding(view(14, sim=0.6), now=now)[0] is None      # not similar enough
    assert g.match_embedding(person(99), now=now)[0] is None              # nobody we know
    assert g.match_embedding(view(14), exclude={"014"}, now=now)[0] is None


def test_look_alikes_are_not_guessed():
    g = AppearanceGallery()
    now = time.time()
    a = person(1)
    b = 0.97 * a + 0.03 * person(2)            # someone dressed almost identically
    g.add_embedding("001", a, now)
    g.add_embedding("002", b / np.linalg.norm(b), now)
    assert g.match_embedding(view(1), now=now)[0] is None


def test_forgets_everyone_at_the_start_of_a_new_day():
    g = AppearanceGallery()
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).timestamp()
    g.add_embedding("014", person(14), yesterday)
    assert g.match_embedding(view(14))[0] is None


# ---- overlay behaviour ----------------------------------------------------------

class FakeGallery:
    def __init__(self, answer):
        self.answer = answer

    def known_count(self):
        return 1

    def match(self, frame, bbox, exclude=None):
        emp, score = self.answer
        return (None, score) if emp in (exclude or set()) else (emp, score)


@pytest.fixture
def pipeline(monkeypatch):
    from app.face_pipeline import CameraFacePipeline, PersonTrackState

    monkeypatch.setattr(analytics_settings, "_state", {k: True for k in analytics_settings.FEATURES})
    p = CameraFacePipeline.__new__(CameraFacePipeline)  # no models needed for this logic
    p.camera_id = 11
    p.person_tracks = {3: PersonTrackState(track_id=3), 4: PersonTrackState(track_id=4)}
    return p


def test_needs_two_agreeing_matches_before_naming(pipeline, monkeypatch):
    monkeypatch.setattr(appearance, "gallery", FakeGallery(("014", 0.85)))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    pipeline._update_appearance_identities(frame, [3])
    assert pipeline.person_tracks[3].appearance_identity is None     # one match: still "Person"
    pipeline.person_tracks[3].appearance_last_try = 0                 # skip the 2 s wait
    pipeline._update_appearance_identities(frame, [3])
    assert pipeline.person_tracks[3].appearance_identity == "014"


def test_face_identity_wins_and_an_employee_isnt_shown_twice(pipeline, monkeypatch):
    monkeypatch.setattr(appearance, "gallery", FakeGallery(("014", 0.85)))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    t4 = pipeline.person_tracks[4]
    t4.current_identity, t4.last_recognized_time = "014", time.time()   # 014 recognised by face on track 4
    for _ in range(3):
        pipeline.person_tracks[3].appearance_last_try = 0
        pipeline._update_appearance_identities(frame, [3, 4])
    assert pipeline.person_tracks[3].appearance_identity is None     # 014 is already on screen by face
    assert t4.appearance_identity is None and t4.appearance_last_try == 0  # face-identified: never tried
