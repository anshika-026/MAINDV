"""Shared test setup.

Environment is fixed BEFORE any app module is imported, so module-level
DB_PATH / DATA_DIR values point at a throwaway directory and a test run can
never touch the real backend/data/app.db.
"""

import os
import tempfile

_SESSION_DATA = tempfile.mkdtemp(prefix="maindv-test-")
os.environ["APP_ENV"] = "test"
os.environ["DATA_DIR"] = _SESSION_DATA
os.environ["LOG_TO_FILE"] = "false"
os.environ.pop("ADMIN_EMAIL", None)
os.environ.pop("ADMIN_PASSWORD", None)

import pytest  # noqa: E402

# Every module that keeps its own module-level DB_PATH.
_DB_MODULES = (
    "app.alerts", "app.analytics_settings", "app.attendance", "app.audit", "app.auth", "app.camera_db",
    "app.desk_db", "app.face_db", "app.insights", "app.intrusion", "app.license_db",
    "app.reid.peopleid_db", "app.reid.reid_db", "app.staff.occupancy",
)


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """A fresh SQLite file and DATA_DIR per test, with every table created."""
    import importlib

    from app import config

    db_file = tmp_path / "app.db"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", db_file)
    for name in _DB_MODULES:
        mod = importlib.import_module(name)
        monkeypatch.setattr(mod, "DB_PATH", db_file)

    from app import analytics_settings, audit, auth, camera_db, face_db, license_db
    from app.staff import occupancy

    camera_db.init_db()
    analytics_settings.init_db()
    face_db.init_face_tables()
    license_db.init_db()
    auth.init_db()
    audit.init_db()
    occupancy.init_db()
    return db_file


@pytest.fixture
def fast_hashing(monkeypatch):
    """Real PBKDF2 at 600k iterations takes ~0.5 s per call; tests use 1k.
    Production strength is covered by its own test."""
    from app import passwords

    monkeypatch.setattr(passwords, "ITERATIONS", 1000)
    monkeypatch.setattr(passwords, "_DUMMY_HASH", None)


@pytest.fixture
def fresh_rate_limits(monkeypatch):
    from app import ratelimit

    monkeypatch.setattr(ratelimit, "admin_login_guard", ratelimit.LoginGuard("admin-login", per_ip=20, per_user=5, window=300))
    monkeypatch.setattr(ratelimit, "client_login_guard", ratelimit.LoginGuard("client-login", per_ip=20, per_user=5, window=300))


@pytest.fixture
def api(isolated_db, fast_hashing, fresh_rate_limits):
    """TestClient on the real app WITHOUT running startup (no cameras,
    models or background threads) — `with TestClient(...)` would run it."""
    from fastapi.testclient import TestClient

    from app import main

    return TestClient(main.app)


ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "Correct-Horse-9"


@pytest.fixture
def admin_token(api):
    from app import auth

    auth.create_admin_user(ADMIN_EMAIL, ADMIN_PASSWORD, "Admin")
    r = api.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
