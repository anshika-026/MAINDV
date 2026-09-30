"""Object detection service (app/object_detection/service.py) and its API,
with a fake detector: rate limiting, class filtering, recovery, tenant scope."""

import types

import numpy as np
import pytest

from app import analytics_settings, camera_db, license_db, resilience
from app.object_detection import service as svc_mod
from tests.conftest import bearer

FRAME = np.zeros((100, 200, 3), np.uint8)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeDetector:
    def __init__(self, objects=None, fail_times=0):
        self.objects = objects if objects is not None else [
            {"name": "laptop", "confidence": 0.93, "bbox": [10, 10, 50, 40], "track_id": 1, "color": "black"},
            {"name": "backpack", "confidence": 0.8, "bbox": [60, 10, 90, 80], "track_id": 2},
            {"name": "potted plant", "confidence": 0.9, "bbox": [0, 0, 5, 5]},       # not a target class
            {"name": "bottle", "confidence": 0.5, "bbox": [150, 20, 400, 999]},      # clamped to the frame
            {"name": "handbag", "confidence": 0.5, "bbox": [30, 30, 30, 60]},        # zero width: dropped
            {"name": "laptop", "confidence": 0.4, "bbox": ["x", 1, 2, 3]},           # malformed: dropped
        ]
        self.fail_times = fail_times
        self.calls = []
        self.trackers = 0

    def run(self, frame, object_tracker=None, run_ppe=True):
        self.calls.append((object_tracker, run_ppe))
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("inference failed")
        return {"objects": [dict(o) for o in self.objects], "all_objects": []}

    def new_tracker(self):
        self.trackers += 1
        return object()


@pytest.fixture
def switch_on(monkeypatch):
    monkeypatch.setitem(analytics_settings._state, "object_detection", True)


def make(fake, **kw):
    clock = Clock()
    loads = {"n": 0}

    def loader():
        loads["n"] += 1
        return types.SimpleNamespace(run=fake.run, new_tracker=fake.new_tracker)

    s = svc_mod.ObjectDetectionService(loader=loader, clock=clock)
    s.health = resilience.FeatureHealth("object_detection", base=0, maximum=0, rebuild_after=kw.get("rebuild_after", 3))
    return s, clock, loads


def test_off_by_default_and_ignores_frames_when_off():
    assert analytics_settings.DEFAULT_OFF >= {"object_detection"}
    s, _, loads = make(FakeDetector())
    s.feed(3, FRAME)
    assert s.run_pending() == 0 and loads["n"] == 0


def test_only_target_classes_are_reported_and_boxes_are_sane(switch_on):
    fake = FakeDetector()
    s, _, _ = make(fake)
    s.feed(3, FRAME)
    assert s.run_pending() == 1
    [r] = s.latest()
    assert [o["class"] for o in r["objects"]] == ["laptop", "backpack", "bottle"]
    assert r["objects"][2]["bbox"] == [150, 20, 200, 100]
    assert r["counts"] == {"backpack": 1, "handbag": 0, "bottle": 1, "laptop": 1}
    assert (r["frame_w"], r["frame_h"]) == (200, 100)
    assert fake.calls[0][1] is False                      # PPE never run


def test_each_camera_is_rate_limited_and_keeps_its_own_tracker(switch_on):
    fake = FakeDetector()
    s, clock, _ = make(fake)
    for cid in (3, 4):
        s.feed(cid, FRAME)
    s.run_pending()
    s.feed(3, FRAME)                                       # too soon
    assert s.run_pending() == 0
    clock.t += 1 / svc_mod.FPS + 0.01
    s.feed(3, FRAME)
    assert s.run_pending() == 1
    assert fake.trackers == 2 and len(fake.calls) == 3
    assert fake.calls[0][0] is fake.calls[2][0]           # camera 3 reused its tracker


def test_least_recently_served_camera_goes_first(switch_on):
    fake = FakeDetector()
    s, clock, _ = make(fake)
    order = []
    s._detect = lambda cid, frame: order.append(cid) or {"camera_id": cid, "ts": clock(), "objects": [], "counts": {}}
    s._last_run.update({3: 10.0, 4: 5.0, 7: 1.0})
    for cid in (3, 4, 7):
        s._pending[cid] = FRAME
    s.run_pending()
    assert order == [7, 4, 3]


def test_a_failure_backs_off_then_recovers_and_reloads_after_repeated_failures(switch_on):
    fake = FakeDetector(fail_times=2)
    s, clock, loads = make(fake, rebuild_after=2)
    for i in range(3):
        clock.t += 100
        s.feed(3, FRAME)
        s.run_pending()
        if i == 0:
            assert s.health.state == resilience.DEGRADED and s._detector is not None
        if i == 1:
            assert s.health.state == resilience.FAILED and s._detector is None   # dropped for reload
    assert s.health.state == resilience.RUNNING and loads["n"] == 2
    assert s.latest()[0]["counts"]["laptop"] == 1


def test_missing_model_is_a_clear_retryable_error(switch_on, monkeypatch, tmp_path):
    from app import config, models

    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_OFFLINE_MODE", True)
    with pytest.raises(models.ModelMissingError, match="yolo26s.pt"):
        svc_mod._load_detector()


def test_stale_results_are_flagged(switch_on):
    s, clock, _ = make(FakeDetector())
    s.feed(3, FRAME)
    s.run_pending()
    assert s.latest()[0]["stale"] is False
    clock.t += (svc_mod.STALE_AFTER_INTERVALS + 1) / svc_mod.FPS
    assert s.latest()[0]["stale"] is True


def test_vendored_modules_do_not_print_to_stdout(capsys):
    from app.object_detection._quiet import print as quiet_print

    quiet_print("[OBJECT] raw detections: 3")
    assert capsys.readouterr().out == ""


# --- API -------------------------------------------------------------------------

@pytest.fixture
def od_api(api, admin_token, monkeypatch, switch_on):
    s, _, _ = make(FakeDetector())
    monkeypatch.setattr(svc_mod, "service", s)
    from app.object_detection import routes

    monkeypatch.setattr(routes, "service", s)
    cam_a = camera_db.add_camera("Hall A", "Site", host="10.0.0.1")
    cam_b = camera_db.add_camera("Hall B", "Site", host="10.0.0.2")
    for cid in (cam_a, cam_b):
        s.feed(cid, FRAME)
    s.run_pending()
    company = license_db.create_company("Acme")
    lic = license_db.create_license(company["id"], 2, "a", username="acme", password="acme-pass-123")
    license_db.assign_cameras(lic["id"], [cam_a])
    token = api.post("/api/licenses/client-login", json={"username": "acme", "password": "acme-pass-123"}).json()["session_token"]
    return api, admin_token, token, lic, cam_a, cam_b


def test_admin_sees_every_camera(od_api):
    api, admin, _, _, cam_a, cam_b = od_api
    r = api.get("/api/objects/latest", headers=bearer(admin))
    assert r.status_code == 200 and {c["camera_id"] for c in r.json()["cameras"]} == {cam_a, cam_b}


def test_client_needs_the_licensed_feature(od_api):
    api, _, client, _, _, _ = od_api
    assert api.get("/api/objects/latest", headers=bearer(client)).status_code == 403


def test_client_with_the_feature_sees_only_their_cameras(od_api):
    api, _, client, lic, cam_a, _ = od_api
    license_db.set_license_features(lic["id"], ["object_detection"])
    r = api.get("/api/objects/latest", headers=bearer(client))
    assert r.status_code == 200 and [c["camera_id"] for c in r.json()["cameras"]] == [cam_a]


def test_status_is_admin_only(od_api):
    api, admin, client, lic, _, _ = od_api
    license_db.set_license_features(lic["id"], ["object_detection"])
    assert api.get("/api/objects/status").status_code == 401
    assert api.get("/api/objects/status", headers=bearer(client)).status_code == 403
    assert api.get("/api/objects/status", headers=bearer(admin)).json()["classes"] == list(svc_mod.TARGET_CLASSES)
