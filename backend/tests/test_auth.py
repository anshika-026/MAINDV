"""Admin authentication: real passwords, session validation, authorization."""

import time

import pytest

from app import auth, passwords
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, bearer


def test_valid_login_returns_a_working_session(api, admin_token):
    r = api.get("/api/auth/me", headers=bearer(admin_token))
    assert r.status_code == 200
    assert r.json() == {"email": ADMIN_EMAIL, "name": "Admin"}


def test_email_is_case_insensitive(api, admin_token):
    r = api.post("/api/auth/login", json={"email": ADMIN_EMAIL.upper(), "password": ADMIN_PASSWORD})
    assert r.status_code == 200


def test_wrong_password_is_rejected(api, admin_token):
    r = api.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong-password-1"})
    assert r.status_code == 401
    assert "token" not in r.text


def test_unknown_user_gets_the_same_generic_error(api, admin_token):
    wrong_pw = api.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong-password-1"})
    unknown = api.post("/api/auth/login", json={"email": "nobody@example.com", "password": "whatever-123"})
    assert unknown.status_code == 401
    assert unknown.json() == wrong_pw.json() == {"detail": "Invalid email or password"}


@pytest.mark.parametrize("body", [
    {"email": ADMIN_EMAIL},
    {"email": ADMIN_EMAIL, "password": ""},
    {"password": ADMIN_PASSWORD},
    {},
])
def test_missing_credentials_are_rejected(api, admin_token, body):
    r = api.post("/api/auth/login", json=body)
    assert r.status_code == 422


def test_the_old_email_only_login_no_longer_works(api, admin_token):
    assert api.post("/api/auth/login", json={"email": "anyone@anywhere.com"}).status_code == 422


def test_expired_token_is_rejected_and_removed(api, admin_token):
    with auth.get_connection() as conn:
        conn.execute("UPDATE sessions SET expires_at = ? WHERE token = ?", (time.time() - 1, admin_token))
    assert api.get("/api/auth/me", headers=bearer(admin_token)).status_code == 401
    with auth.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE token = ?", (admin_token,)).fetchone()[0] == 0


@pytest.mark.parametrize("header", [
    "Bearer",
    "Bearer ",
    "Basic YWRtaW46YWRtaW4=",
    "Bearer not-a-real-token",
    "Bearer " + "x" * 5000,
    "token-without-scheme",
])
def test_malformed_or_unknown_tokens_are_rejected(api, admin_token, header):
    assert api.get("/api/auth/me", headers={"Authorization": header}).status_code == 401


@pytest.mark.parametrize("method,path,body", [
    ("get", "/api/settings", None),
    ("put", "/api/settings", {"detection_fps": 2}),
    ("get", "/api/stats", None),
    ("get", "/api/cameras", None),
    ("get", "/api/licenses", None),
    ("get", "/api/faces/pending", None),
    ("get", "/api/faces/training/stats", None),
    ("get", "/api/footfall/summary", None),
    ("get", "/api/alerts", None),
    ("get", "/api/staff/count", None),
    ("get", "/api/audit", None),
])
def test_protected_endpoints_require_authentication(api, method, path, body):
    r = getattr(api, method)(path, json=body) if body is not None else getattr(api, method)(path)
    assert r.status_code == 401, (path, r.status_code)


def test_authorized_admin_can_use_admin_endpoints(api, admin_token):
    h = bearer(admin_token)
    assert api.get("/api/settings", headers=h).status_code == 200
    assert api.get("/api/stats", headers=h).status_code == 200
    assert api.get("/api/licenses", headers=h).status_code == 200


def test_logout_revokes_the_session(api, admin_token):
    assert api.post("/api/auth/logout", headers=bearer(admin_token)).status_code == 200
    assert api.get("/api/auth/me", headers=bearer(admin_token)).status_code == 401


def test_disabling_an_admin_ends_their_sessions(api, admin_token):
    auth.set_admin_active(ADMIN_EMAIL, False)
    assert api.get("/api/auth/me", headers=bearer(admin_token)).status_code == 401
    r = api.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert r.status_code == 401


def test_password_change_ends_existing_sessions(api, admin_token):
    auth.set_admin_password(ADMIN_EMAIL, "A-new-Password-42")
    assert api.get("/api/auth/me", headers=bearer(admin_token)).status_code == 401
    assert api.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}).status_code == 401
    assert api.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "A-new-Password-42"}).status_code == 200


def test_passwords_are_never_stored_in_plaintext(api, admin_token):
    with auth.get_connection() as conn:
        stored = conn.execute("SELECT password_hash FROM admin_users").fetchone()[0]
    assert ADMIN_PASSWORD not in stored
    assert stored.startswith("pbkdf2_sha256$")
    assert passwords.verify_password(ADMIN_PASSWORD, stored)


@pytest.mark.parametrize("bad", ["short", "          x", "aaaaaaaaaaaa", " leading-space-pass"])
def test_weak_passwords_are_refused(isolated_db, fast_hashing, bad):
    with pytest.raises(ValueError):
        auth.create_admin_user("weak@example.com", bad)


def test_duplicate_admin_email_is_refused(isolated_db, fast_hashing):
    auth.create_admin_user("dup@example.com", "Some-Password-1")
    with pytest.raises(ValueError):
        auth.create_admin_user("DUP@example.com", "Some-Password-2")


def test_sessions_from_the_passwordless_era_are_revoked_on_upgrade(tmp_path, monkeypatch):
    import sqlite3

    db = tmp_path / "legacy.db"
    monkeypatch.setattr(auth, "DB_PATH", db)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sessions (token TEXT PRIMARY KEY, principal_type TEXT NOT NULL, license_id TEXT, "
                 "email TEXT, created_at REAL NOT NULL, expires_at REAL NOT NULL)")
    now = time.time()
    conn.execute("INSERT INTO sessions VALUES ('old-admin', 'admin', NULL, 'x@y.z', ?, ?)", (now, now + 999))
    conn.execute("INSERT INTO sessions VALUES ('a-client', 'client', 'lic1', NULL, ?, ?)", (now, now + 999))
    conn.commit()
    conn.close()
    auth.init_db()
    with auth.get_connection() as c:
        tokens = {r[0] for r in c.execute("SELECT token FROM sessions")}
    assert tokens == {"a-client"}


def test_bootstrap_admin_only_when_none_exists(isolated_db, fast_hashing, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "ADMIN_EMAIL", "boot@example.com")
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "Bootstrap-Pass-1")
    auth.bootstrap_admin_from_env()
    auth.bootstrap_admin_from_env()  # idempotent
    assert [a["email"] for a in auth.list_admin_users()] == ["boot@example.com"]
    assert auth.authenticate_admin("boot@example.com", "Bootstrap-Pass-1") is not None


def test_production_hash_strength():
    assert passwords.ITERATIONS >= 600_000
