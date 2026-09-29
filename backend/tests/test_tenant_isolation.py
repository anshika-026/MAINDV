"""Client A must never see client B's data, whatever ID it puts in a URL."""

import time

import pytest
from starlette.websockets import WebSocketDisconnect

from app import camera_db, license_db
from app.staff import occupancy
from app.staff.service import service as staff_service
from tests.conftest import bearer


@pytest.fixture
def tenants(api, admin_token):
    cam_a = camera_db.add_camera("Gate A", "Site A", host="10.0.0.1", password="secretA", purpose="Entry/Exit")
    cam_b = camera_db.add_camera("Gate B", "Site B", host="10.0.0.2", password="secretB", purpose="Entry/Exit")
    camera_db.add_site("Site A", "")
    camera_db.add_site("Site B", "")
    tokens = {}
    for name, cam in (("a", cam_a), ("b", cam_b)):
        company = license_db.create_company(f"Company {name.upper()}")
        lic = license_db.create_license(company["id"], 2, name, username=f"client_{name}", password=f"pass-{name}-123456")
        license_db.set_license_features(lic["id"], ["attendance", "footfall_analytics"])
        license_db.assign_cameras(lic["id"], [cam])
        r = api.post("/api/licenses/client-login", json={"username": f"client_{name}", "password": f"pass-{name}-123456"})
        assert r.status_code == 200, r.text
        tokens[name] = r.json()["session_token"]
        tokens[f"lic_{name}"] = lic["id"]

    # One staff event and one present visit on each camera.
    mgr = occupancy.StaffOccupancyManager()
    staff_service.manager = mgr
    now = time.time()
    for cam, emp in ((cam_a, "EMP-A"), (cam_b, "EMP-B")):
        mgr._event(now, "ENTRY", camera_id=cam, employee_id=emp)
        mgr._present[f"v-{emp}"] = {"visit_id": f"v-{emp}", "employee_id": emp, "entry_time": now,
                                     "last_seen": now, "camera_id": cam, "confidence": 0.9}
    yield {"a": tokens["a"], "b": tokens["b"], "cam_a": cam_a, "cam_b": cam_b,
           "lic_a": tokens["lic_a"], "lic_b": tokens["lic_b"], "admin": admin_token}
    staff_service.manager = None


def test_client_sees_only_their_cameras(api, tenants):
    r = api.get("/api/cameras", headers=bearer(tenants["a"]))
    assert [c["id"] for c in r.json()] == [tenants["cam_a"]]
    assert "password" not in r.text and "secretB" not in r.text and "secretA" not in r.text


def test_sites_never_leak_other_tenants_camera_names(api, tenants):
    r = api.get("/api/sites", headers=bearer(tenants["a"]))
    body = r.text
    assert "Gate A" in body and "Gate B" not in body and "Site B" not in body


def test_admin_still_sees_everything(api, tenants):
    h = bearer(tenants["admin"])
    assert {c["id"] for c in api.get("/api/cameras", headers=h).json()} == {tenants["cam_a"], tenants["cam_b"]}
    assert {e["employee_id"] for e in api.get("/api/staff/events", headers=h).json()} == {"EMP-A", "EMP-B"}


@pytest.mark.parametrize("path", ["/api/staff/events", "/api/staff/entries", "/api/staff/present"])
def test_staff_data_is_scoped_to_the_clients_cameras(api, tenants, path):
    rows = api.get(path, headers=bearer(tenants["a"])).json()
    assert rows and {r["employee_id"] for r in rows} == {"EMP-A"}


def test_staff_count_is_scoped(api, tenants):
    a = api.get("/api/staff/count", headers=bearer(tenants["a"])).json()
    admin = api.get("/api/staff/count", headers=bearer(tenants["admin"])).json()
    assert a["total_persons"] == 1 and a["total_entries_today"] == 1
    assert admin["total_persons"] == 2 and admin["total_entries_today"] == 2


def test_staff_requires_the_licensed_feature(api, tenants):
    license_db.set_license_features(tenants["lic_a"], ["footfall_analytics"])
    assert api.get("/api/staff/count", headers=bearer(tenants["a"])).status_code == 403


@pytest.mark.parametrize("method,path", [
    ("get", "/api/licenses/{lic_b}"),
    ("get", "/api/licenses/{lic_b}/cameras"),
    ("get", "/api/licenses/{lic_b}/qr"),
    ("put", "/api/cameras/{cam_b}"),
    ("delete", "/api/cameras/{cam_b}"),
    ("get", "/api/cameras/{cam_b}/frame"),
    ("get", "/api/footfall/cameras/{cam_b}/frame"),
    ("get", "/api/footfall/snapshots/1"),
    ("get", "/api/alerts/1/snapshot"),
    ("get", "/api/faces/people/photo/1"),
    ("get", "/api/faces/training/image/1"),
    ("get", "/api/faces/gallery/EMP-B/count"),
    ("get", "/api/attendance/EMP-B/history"),
    ("get", "/api/faces/identity/roster"),
    ("get", "/api/staff/status"),
    ("get", "/api/audit"),
    ("get", "/api/settings"),
])
def test_client_a_cannot_reach_other_tenant_or_admin_objects(api, tenants, method, path):
    url = path.format(**tenants)
    kwargs = {"json": {}} if method == "put" else {}
    r = getattr(api, method)(url, headers=bearer(tenants["a"]), **kwargs)
    assert r.status_code == 403, (url, r.status_code, r.text[:200])


def test_live_websocket_rejects_other_tenants_camera(api, tenants):
    with api.websocket_connect(f"/ws/live/{tenants['cam_b']}?token={tenants['a']}") as ws:
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_bytes()
    assert e.value.code == 4401


def test_detections_websocket_rejects_other_tenants_camera(api, tenants):
    with api.websocket_connect(f"/ws/detections/{tenants['cam_b']}?token={tenants['a']}") as ws:
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    assert e.value.code == 4401


def test_staff_websocket_is_scoped(api, tenants):
    with api.websocket_connect(f"/ws/staff?token={tenants['a']}") as ws:
        assert ws.receive_json()["total_persons"] == 1


def test_staff_debug_websocket_is_admin_only(api, tenants):
    with api.websocket_connect(f"/ws/staff/debug/{tenants['cam_a']}?token={tenants['a']}") as ws:
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    assert e.value.code == 4403


def test_suspending_a_license_ends_its_sessions(api, tenants):
    r = api.post(f"/api/licenses/{tenants['lic_a']}/status", headers=bearer(tenants["admin"]), json={"status": "suspended"})
    assert r.status_code == 200
    assert api.get("/api/cameras", headers=bearer(tenants["a"])).status_code == 401


def test_resetting_credentials_ends_its_sessions(api, tenants):
    r = api.put(f"/api/licenses/{tenants['lic_a']}/credentials", headers=bearer(tenants["admin"]),
                json={"username": "client_a", "password": "brand-new-pass-1"})
    assert r.status_code == 200
    assert api.get("/api/cameras", headers=bearer(tenants["a"])).status_code == 401
