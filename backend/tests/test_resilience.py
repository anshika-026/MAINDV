"""Analytics components recover from failures instead of switching off for good."""

import numpy as np
import pytest

from app import resilience
from app.person_detection import SharedPersonDetection, _Cam


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(**kw):
    clock = Clock()
    kw.setdefault("base", 5)
    kw.setdefault("maximum", 60)
    kw.setdefault("rebuild_after", 3)
    return resilience.FeatureHealth("test", 1, clock=clock, **kw), clock


def test_failure_backs_off_exponentially_and_is_capped():
    h, clock = make()
    delays = []
    for _ in range(6):
        h.failure(RuntimeError("boom"))
        start = clock.t
        assert not h.allow()
        while not h.allow():
            clock.t += 1
        delays.append(clock.t - start)
    assert delays == [5, 10, 20, 40, 60, 60]


def test_states_and_recovery():
    h, clock = make()
    assert h.state == resilience.RUNNING and h.allow()
    h.failure(ValueError("x"))
    assert h.state == resilience.DEGRADED and h.failed
    clock.t += 5
    assert h.allow() and h.state == resilience.RECOVERING
    h.success()
    assert h.state == resilience.RUNNING and not h.failed and h.allow()


def test_rebuild_is_requested_every_n_failures_and_state_becomes_failed():
    h, clock = make(rebuild_after=3)
    flags = [h.failure(RuntimeError()) for _ in range(6)]
    assert flags == [False, False, True, False, False, True]
    assert h.state == resilience.FAILED


def test_stop_blocks_until_reset():
    h, _ = make()
    h.stop()
    assert not h.allow() and h.state == resilience.STOPPED
    h.reset()
    assert h.allow()


def test_snapshot_is_safe_and_bounded():
    h, _ = make()
    h.failure(RuntimeError("x" * 5000))
    snap = h.snapshot()
    assert len(snap["last_error"]) <= 300 and snap["consecutive_failures"] == 1


def test_backoff_delay_with_jitter_stays_in_bounds():
    for attempt in range(1, 12):
        for r in (0.0, 0.5, 1.0):
            d = resilience.backoff_delay(attempt, 2, 60, jitter=0.2, rand=lambda: r)
            nominal = min(60, 2 * 2 ** (attempt - 1))
            assert nominal * 0.8 - 1e-9 <= d <= nominal * 1.2 + 1e-9


# --- person detection: one failure no longer stops footfall/staff/intrusion ---

class FlakyModel:
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def track(self, *a, **k):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("inference failed")
        boxes = type("B", (), {"xyxy": _T([[1, 2, 3, 4]]), "conf": _T([0.9]), "id": _T([5]), "__len__": lambda s: 1})()
        return [type("R", (), {"boxes": boxes})()]


class _T:
    def __init__(self, a):
        self.a = np.array(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


FRAME = np.zeros((50, 50, 3), np.uint8)


def test_person_detection_recovers_after_a_model_exception():
    svc = SharedPersonDetection()
    got = []
    svc.register("footfall", lambda cid: 2, lambda cid, f, people, ts: got.append(people))
    cam = _Cam(7)
    cam.health = resilience.FeatureHealth("person_detection", 7, base=0, maximum=0, rebuild_after=5)
    cam.model = FlakyModel(fail_times=1)
    wanted = svc._wanted(7)
    svc._run(cam, FRAME, 1.0, wanted)
    assert cam.failed and got == []
    assert cam.health.allow()            # retried, not disabled
    svc._run(cam, FRAME, 2.0, wanted)
    assert not cam.failed and cam.health.state == resilience.RUNNING
    assert got and got[-1][0]["track_id"] == 5


def test_person_detection_reloads_the_model_after_repeated_failures():
    svc = SharedPersonDetection()
    svc.register("staff", lambda cid: 5, lambda *a: None)
    cam = _Cam(8)
    cam.health = resilience.FeatureHealth("person_detection", 8, base=0, maximum=0, rebuild_after=2)
    cam.model = FlakyModel(fail_times=99)
    wanted = svc._wanted(8)
    svc._run(cam, FRAME, 1.0, wanted)
    assert cam.model is not None
    svc._run(cam, FRAME, 2.0, wanted)
    assert cam.model is None             # dropped -> reloaded on next run


def test_person_detection_skips_frames_while_backing_off():
    svc = SharedPersonDetection()
    svc.register("intrusion", lambda cid: 1, lambda *a: None)
    cam = svc._cams.setdefault(9, _Cam(9))
    cam.health = resilience.FeatureHealth("person_detection", 9, base=1000, maximum=1000)
    cam.health.failure(RuntimeError())
    svc.feed(9, FRAME)
    assert cam.future is None            # nothing submitted during backoff


# --- face recognition in the camera stream ------------------------------------

def test_face_pipeline_failure_pauses_then_rebuilds_then_recovers(monkeypatch):
    from app import camera_stream

    calls = {"n": 0, "discarded": 0}

    class FlakyPipeline:
        def feed_frame(self, frame, has_viewer=False):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("classifier exploded")

    monkeypatch.setattr(camera_stream, "get_pipeline", lambda cid: FlakyPipeline())
    monkeypatch.setattr(camera_stream, "discard_pipeline", lambda cid: calls.__setitem__("discarded", calls["discarded"] + 1))
    s = camera_stream.CameraStream(42)
    s.face_health = resilience.FeatureHealth("face_recognition", 42, base=0, maximum=0, rebuild_after=2)
    s._feed_pipeline(FRAME, False)
    assert s.face_health.state == resilience.DEGRADED and s._face_pipeline is not None
    s._feed_pipeline(FRAME, False)
    assert s.face_health.state == resilience.FAILED and calls["discarded"] == 1 and s._face_pipeline is None
    s._feed_pipeline(FRAME, False)
    assert s.face_health.state == resilience.RUNNING and calls["n"] == 3


def test_model_initialization_failure_is_retried(monkeypatch):
    from app import camera_stream

    attempts = {"n": 0}

    def get_pipeline(cid):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise FileNotFoundError("yolov8n-face.pt missing")
        return type("P", (), {"feed_frame": lambda self, f, has_viewer=False: None})()

    monkeypatch.setattr(camera_stream, "get_pipeline", get_pipeline)
    s = camera_stream.CameraStream(43)
    s.face_health = resilience.FeatureHealth("face_recognition", 43, base=0, maximum=0)
    s._feed_pipeline(FRAME, False)
    assert s.face_health.failed
    s._feed_pipeline(FRAME, False)
    assert not s.face_health.failed and attempts["n"] == 2


# --- intrusion: executor exceptions are no longer swallowed ----------------------

def test_intrusion_detection_failure_is_recorded_and_recovers(monkeypatch):
    from app import intrusion

    svc = intrusion.IntrusionService() if hasattr(intrusion, "IntrusionService") else type(intrusion.service)()
    svc._health[3] = resilience.FeatureHealth("intrusion", 3, base=0, maximum=0)
    boom = {"on": True}

    def inner(*a):
        if boom["on"]:
            raise RuntimeError("database is locked")

    monkeypatch.setattr(svc, "_detect_inner", inner)
    svc._detect(3, FRAME, [], 1.0, [[0, 0, 1, 1]])
    assert svc.health(3).failed
    boom["on"] = False
    svc._detect(3, FRAME, [], 2.0, [[0, 0, 1, 1]])
    assert not svc.health(3).failed


def test_footfall_gate_failure_is_retried_not_disabled(monkeypatch):
    from app import footfall
    from app.reid import reid_worker

    svc = type(footfall.service)()
    svc._health[5] = resilience.FeatureHealth("footfall", 5, base=0, maximum=0, rebuild_after=2)
    runner = type("Runner", (), {"camera_id": 5, "state": None, "generation": 0,
                                 "executor": type("E", (), {"shutdown": lambda self, wait=False: None})()})()
    svc._runners[5] = runner
    monkeypatch.setattr(reid_worker, "process_frame", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reid")))
    svc._process(runner, FRAME, [], 1.0)
    assert svc.health(5).failed and svc._runners.get(5) is runner
    svc._process(runner, FRAME, [], 2.0)
    assert 5 not in svc._runners          # rebuilt next time
    monkeypatch.setattr(reid_worker, "process_frame", lambda *a, **k: None)
    svc._process(runner, FRAME, [], 3.0)
    assert not svc.health(5).failed


@pytest.fixture(autouse=True)
def _reset_lifecycle():
    from app import lifecycle

    lifecycle.reset()
    yield
    lifecycle.reset()


def test_lifecycle_wait_returns_on_shutdown():
    import threading
    import time

    from app import lifecycle

    t0 = time.monotonic()
    threading.Timer(0.1, lifecycle.begin_shutdown).start()
    assert lifecycle.wait(10) is True
    assert time.monotonic() - t0 < 5
