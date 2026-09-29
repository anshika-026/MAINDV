"""/api/settings: admin only, validated, audited."""

import pytest

from app import audit
from tests.conftest import bearer


def test_admin_can_read_and_update(api, admin_token):
    h = bearer(admin_token)
    r = api.put("/api/settings", headers=h, json={"detection_fps": 2.5})
    assert r.status_code == 200 and r.json()["detection_fps"] == 2.5
    assert api.get("/api/settings", headers=h).json()["detection_fps"] == 2.5


@pytest.mark.parametrize("body", [
    {"detection_fps": 0},
    {"detection_fps": 1000},
    {"detection_fps": "fast"},
    {"detection_fps": 2, "admin": True},
    {"__proto__": {"x": 1}},
    {},
])
def test_invalid_or_unknown_settings_are_rejected(api, admin_token, body):
    assert api.put("/api/settings", headers=bearer(admin_token), json=body).status_code == 422


def test_settings_changes_are_audited(api, admin_token):
    api.put("/api/settings", headers=bearer(admin_token), json={"detection_fps": 3})
    rows = audit.recent()
    assert any(r["action"] == "settings.update" and r["actor"] == "admin@example.com" for r in rows)
    assert all("password" not in (r["details"] or "") for r in rows)


def test_the_old_unauthenticated_alert_stubs_are_gone(api, admin_token):
    # /api/alerts is served by alerts_routes (admin-only), not an open stub.
    assert api.get("/api/alerts").status_code == 401
    assert api.post("/api/alerts/1/resolve", json={"reason": "x"}).status_code == 401
