"""/health (liveness) and /ready (readiness)."""

from app import auth, models, resilience
from tests.conftest import bearer


def test_health_is_always_ok(api):
    r = api.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_not_ready_without_an_admin_account(api, monkeypatch):
    monkeypatch.setattr(models, "missing_required", lambda: [])
    r = api.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["admin_account"]["ok"] is False


def test_not_ready_when_a_required_model_is_missing(api, fast_hashing, monkeypatch):
    auth.create_admin_user("ops@example.com", "Ops-Password-1")
    monkeypatch.setattr(models, "missing_required", lambda: ["yolo_face"])
    r = api.get("/ready")
    assert r.status_code == 503 and r.json()["checks"]["models"]["missing"] == ["yolo_face"]


def test_ready_when_everything_is_in_place(api, fast_hashing, monkeypatch):
    auth.create_admin_user("ops@example.com", "Ops-Password-1")
    monkeypatch.setattr(models, "missing_required", lambda: [])
    r = api.get("/ready")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_ready_reports_degraded_features_without_failing(api, fast_hashing, monkeypatch):
    auth.create_admin_user("ops@example.com", "Ops-Password-1")
    monkeypatch.setattr(models, "missing_required", lambda: [])
    h = resilience.health_for("footfall", 77)
    h.failure(RuntimeError("secret stuff at rtsp://admin:Admin@123@10.0.0.1"))
    try:
        body = api.get("/ready").json()
        assert "footfall" in body["degraded"]
        assert "Admin@123" not in str(body) and "rtsp://" not in str(body)
    finally:
        resilience.unregister("footfall", 77)


def test_public_readiness_carries_no_sensitive_details(api, fast_hashing, monkeypatch):
    auth.create_admin_user("ops@example.com", "Ops-Password-1")
    text = api.get("/ready").text
    for secret in ("ops@example.com", "password", "host", "path", "C:\\", "/home/"):
        assert secret not in text


def test_detailed_health_is_admin_only(api, admin_token):
    assert api.get("/api/health/details").status_code == 401
    r = api.get("/api/health/details", headers=bearer(admin_token))
    assert r.status_code == 200 and "models" in r.json()
