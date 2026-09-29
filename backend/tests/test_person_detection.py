"""Shared per-camera person detector (app/person_detection.py): one detection
run feeds every consumer, each at its own rate, and nothing runs when no
consumer wants the camera."""

import numpy as np

from app.person_detection import SharedPersonDetection, _Cam


class FakeTensor:
    def __init__(self, a):
        self.a = np.array(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeModel:
    """Stands in for YOLO.track: one tracked person, counts its calls."""

    def __init__(self):
        self.calls = 0

    def track(self, *a, **k):
        self.calls += 1
        boxes = type("B", (), {"xyxy": FakeTensor([[10, 20, 60, 200]]), "conf": FakeTensor([0.9]),
                               "id": FakeTensor([7]), "__len__": lambda s: 1})()
        return [type("R", (), {"boxes": boxes})()]


FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


def run_frames(svc, cam, seconds, fps_in=25):
    """Push frames through _run directly (synchronously) at the detector's own pace."""
    wanted = svc._wanted(cam.camera_id)
    step = 1.0 / max(wanted.values())
    t = 1000.0
    while t < 1000.0 + seconds:
        svc._run(cam, FRAME, t, wanted)
        t += step


def test_one_detection_feeds_every_consumer_at_its_own_rate():
    svc = SharedPersonDetection()
    got = {"staff": [], "footfall": [], "intrusion": []}
    for name, fps in (("staff", 5), ("footfall", 2), ("intrusion", 1)):
        svc.register(name, lambda cid, fps=fps: fps, lambda cid, f, people, ts, name=name: got[name].append((ts, people)))
    cam = _Cam(8)
    cam.model = FakeModel()
    run_frames(svc, cam, seconds=10)
    assert cam.model.calls == 50                    # detection ran once per frame, at 5 fps
    assert len(got["staff"]) == 50
    assert 19 <= len(got["footfall"]) <= 21          # ~2 fps
    assert 9 <= len(got["intrusion"]) <= 11          # ~1 fps
    assert got["staff"][0][1] == [{"track_id": 7, "bbox": [10.0, 20.0, 60.0, 200.0], "confidence": 0.9}]


def test_nothing_runs_when_no_consumer_wants_the_camera():
    svc = SharedPersonDetection()
    svc.register("footfall", lambda cid: 0, lambda *a: None)
    svc.feed(8, FRAME)
    assert svc._cams == {}


def test_a_failing_consumer_doesnt_stop_the_others():
    svc = SharedPersonDetection()
    good = []
    svc.register("broken", lambda cid: 5, lambda *a: 1 / 0)
    svc.register("good", lambda cid: 5, lambda cid, f, people, ts: good.append(ts))
    cam = _Cam(8)
    cam.model = FakeModel()
    run_frames(svc, cam, seconds=2)
    assert len(good) == 10
