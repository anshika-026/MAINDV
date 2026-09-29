"""
auth.py

Session-token authentication shared by the admin portal and the client
portal.

Two principal types:
  - "admin": a row in `admin_users`, authenticated with email + password
    (PBKDF2-SHA256, see passwords.py). There is no self-signup: the first
    admin is created with `python -m app.manage create-admin` (or the
    ADMIN_EMAIL / ADMIN_PASSWORD bootstrap in config.py), and further admins
    by an existing one via the same command.
  - "client": a license's own username/password, verified against a PBKDF2
    hash (license_db.verify_license_login). A client session is re-validated
    against the live license status/expiry on every request (see
    load_active_client_license), not just at login.

Sessions are opaque random tokens stored server-side in `sessions` — trivially
revocable (delete the row), unlike a self-contained JWT. Every lookup checks
expiry, and admin sessions also check that the admin account still exists
and is active, so disabling an admin cuts off their open sessions at once.
"""

from __future__ import annotations

import logging
import re
import secrets
import sqlite3
import time
import uuid

from fastapi import Depends, Header, HTTPException

from . import config, db, license_db, passwords

log = logging.getLogger("auth")

DB_PATH = config.DB_PATH

SESSION_TTL_SECONDS = config.SESSION_TTL_SECONDS

ADMIN = "admin"
CLIENT = "client"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def get_connection():
    return db.connect(DB_PATH, row_factory=sqlite3.Row)


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                principal_type TEXT NOT NULL,   -- 'admin' | 'client'
                license_id TEXT,                -- set only for principal_type='client'
                email TEXT,                     -- set only for principal_type='admin'
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_license ON sessions (license_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                last_login_at REAL,
                password_changed_at REAL NOT NULL
            )
            """
        )
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
        if "admin_user_id" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN admin_user_id TEXT")
            # Every admin session that exists at this point was issued by the
            # old password-less login, so none of them proves anything.
            n = conn.execute("DELETE FROM sessions WHERE principal_type = ?", (ADMIN,)).rowcount
            if n:
                log.warning("revoked %d admin session(s) issued before password login existed", n)


# --- Admin accounts -----------------------------------------------------------

def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_new_password(password: str) -> None:
    if not isinstance(password, str) or len(password) < config.PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {config.PASSWORD_MIN_LENGTH} characters")
    if password.strip() != password:
        raise ValueError("Password must not start or end with whitespace")
    if len(set(password)) < 4:
        raise ValueError("Password is too simple")


def create_admin_user(email: str, password: str, name: str | None = None) -> dict:
    email = normalize_email(email)
    if not _EMAIL_RE.match(email):
        raise ValueError("A valid email address is required")
    validate_new_password(password)
    now = time.time()
    user_id = uuid.uuid4().hex
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO admin_users (id, email, name, password_hash, is_active, created_at, password_changed_at) "
                "VALUES (?, ?, ?, ?, 1, ?, ?)",
                (user_id, email, (name or email.split("@")[0]).strip(), passwords.hash_password(password), now, now),
            )
    except sqlite3.IntegrityError as e:
        raise ValueError(f"An admin with email {email} already exists") from e
    log.info("admin account created: %s", email)
    return get_admin_user(user_id)


def set_admin_password(email: str, password: str) -> None:
    validate_new_password(password)
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE admin_users SET password_hash = ?, password_changed_at = ? WHERE email = ?",
            (passwords.hash_password(password), time.time(), normalize_email(email)),
        )
        if cur.rowcount == 0:
            raise ValueError(f"No admin with email {normalize_email(email)}")
        row = conn.execute("SELECT id FROM admin_users WHERE email = ?", (normalize_email(email),)).fetchone()
        # A password change ends every other session for that account.
        conn.execute("DELETE FROM sessions WHERE admin_user_id = ?", (row["id"],))


def set_admin_active(email: str, active: bool) -> None:
    with get_connection() as conn:
        cur = conn.execute("UPDATE admin_users SET is_active = ? WHERE email = ?", (1 if active else 0, normalize_email(email)))
        if cur.rowcount == 0:
            raise ValueError(f"No admin with email {normalize_email(email)}")
        if not active:
            row = conn.execute("SELECT id FROM admin_users WHERE email = ?", (normalize_email(email),)).fetchone()
            conn.execute("DELETE FROM sessions WHERE admin_user_id = ?", (row["id"],))


def _public_admin(row) -> dict:
    return {"id": row["id"], "email": row["email"], "name": row["name"], "is_active": bool(row["is_active"]),
            "created_at": row["created_at"], "last_login_at": row["last_login_at"]}


def get_admin_user(user_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM admin_users WHERE id = ?", (user_id,)).fetchone()
    return _public_admin(row) if row else None


def list_admin_users() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM admin_users ORDER BY created_at").fetchall()
    return [_public_admin(r) for r in rows]


def count_admin_users() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM admin_users WHERE is_active = 1").fetchone()["c"]


def authenticate_admin(email: str, password: str) -> dict | None:
    """The admin dict if email/password are right and the account is active,
    else None. Same work either way, so response time doesn't reveal whether
    an email exists."""
    email = normalize_email(email)
    if not email or not password:
        passwords.burn_time(password or "")
        return None
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM admin_users WHERE email = ?", (email,)).fetchone()
    if row is None:
        passwords.burn_time(password)
        return None
    if not passwords.verify_password(password, row["password_hash"]) or not row["is_active"]:
        return None
    with get_connection() as conn:
        if passwords.needs_rehash(row["password_hash"]):
            conn.execute("UPDATE admin_users SET password_hash = ? WHERE id = ?", (passwords.hash_password(password), row["id"]))
        conn.execute("UPDATE admin_users SET last_login_at = ? WHERE id = ?", (time.time(), row["id"]))
    return _public_admin(row)


def bootstrap_admin_from_env() -> None:
    """Creates the first admin from ADMIN_EMAIL / ADMIN_PASSWORD, only when no
    admin exists yet. Never overwrites or re-creates an existing account."""
    if not config.ADMIN_EMAIL or not config.ADMIN_PASSWORD:
        return
    with get_connection() as conn:
        exists = conn.execute("SELECT 1 FROM admin_users LIMIT 1").fetchone()
    if exists:
        return
    create_admin_user(config.ADMIN_EMAIL, config.ADMIN_PASSWORD)
    log.warning("bootstrap admin %s created from ADMIN_EMAIL/ADMIN_PASSWORD; remove ADMIN_PASSWORD from the environment now",
                normalize_email(config.ADMIN_EMAIL))


# --- Sessions -------------------------------------------------------------------

def _new_token() -> str:
    return secrets.token_urlsafe(32)


def create_admin_session(admin: dict) -> str:
    token = _new_token()
    now = time.time()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (token, principal_type, email, admin_user_id, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
            (token, ADMIN, admin["email"], admin["id"], now, now + SESSION_TTL_SECONDS),
        )
    return token


def create_client_session(license_id: str) -> str:
    token = _new_token()
    now = time.time()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (token, principal_type, license_id, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (token, CLIENT, license_id, now, now + SESSION_TTL_SECONDS),
        )
    return token


def revoke_session(token: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def revoke_license_sessions(license_id: str) -> int:
    """Ends every open client session of one license (credential reset,
    suspension, deletion)."""
    with get_connection() as conn:
        return conn.execute("DELETE FROM sessions WHERE license_id = ?", (license_id,)).rowcount


def purge_expired_sessions(now: float | None = None) -> int:
    with get_connection() as conn:
        return conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now or time.time(),)).rowcount


def get_session(token: str | None) -> dict | None:
    """Resolves a token to a live session, or None if it is missing,
    malformed, unknown, expired, or belongs to a disabled/deleted admin.
    Public because the websocket routes pass tokens as a query param
    (browsers can't set headers on a WebSocket handshake)."""
    if not token or not isinstance(token, str) or len(token) > 256:
        return None
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()
        if row is None:
            return None
        session = dict(row)
        if session["expires_at"] < time.time():
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None
        if session["principal_type"] == ADMIN:
            admin = conn.execute(
                "SELECT is_active FROM admin_users WHERE id = ?", (session.get("admin_user_id"),)
            ).fetchone()
            if admin is None or not admin["is_active"]:
                conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                return None
    return session


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def get_principal(authorization: str | None = Header(None)) -> dict:
    """FastAPI dependency: any authenticated caller, admin or client —
    use this on an endpoint that behaves differently per role (e.g.
    GET /api/cameras returns everything for an admin, only assigned
    cameras for a client) rather than requiring exactly one role."""
    token = _extract_bearer_token(authorization)
    session = get_session(token)
    if session is None:
        raise HTTPException(status_code=401, detail="Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    return session


def require_admin(principal: dict = Depends(get_principal)) -> dict:
    """FastAPI dependency: admin-only endpoint. A client session reaching
    one of these gets a 403."""
    if principal["principal_type"] != ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return principal


def load_active_client_license(principal: dict) -> dict:
    """Re-reads the license fresh from the DB on every call — this is
    what makes a suspend/deactivate/expiry take effect immediately on a
    client's very next request, not just at their next login. Raises 403
    with a clear reason if the license is no longer usable."""
    if principal["principal_type"] != CLIENT:
        raise HTTPException(status_code=403, detail="Client access required")
    lic = license_db.get_license(principal["license_id"])
    if lic is None:
        raise HTTPException(status_code=401, detail="License no longer exists")
    effective = license_db.effective_status(lic)
    if effective != license_db.STATUS_ACTIVE:
        raise HTTPException(status_code=403, detail=f"This license is {effective} — contact your administrator")
    return lic


def get_current_client(principal: dict = Depends(get_principal)) -> dict:
    """FastAPI dependency: client-only endpoint. Returns
    {**principal, "license": <fresh license dict>}."""
    lic = load_active_client_license(principal)
    return {**principal, "license": lic}


def allowed_camera_ids(principal: dict) -> set[int] | None:
    """Cameras a principal may see: None = all (admin), else the set of camera
    ids assigned to the client's active license. The single tenant-scoping
    rule every per-camera endpoint applies."""
    if principal["principal_type"] == ADMIN:
        return None
    lic = load_active_client_license(principal)
    return {c["id"] for c in license_db.list_cameras_for_license(lic["id"])}


def can_access_camera(session: dict | None, camera_id: int) -> bool:
    """Websocket-friendly (no exceptions): may this session see this camera?"""
    if session is None:
        return False
    if session["principal_type"] == ADMIN:
        return True
    try:
        lic = load_active_client_license(session)
    except HTTPException:
        return False
    return license_db.is_camera_assigned(lic["id"], camera_id)


def require_feature(feature_key: str):
    """Dependency factory: admin always passes; a client needs this feature
    enabled on their license."""
    def dep(principal: dict = Depends(get_principal)) -> dict:
        if principal["principal_type"] == ADMIN:
            return principal
        lic = load_active_client_license(principal)
        if feature_key not in license_db.list_license_features(lic["id"]):
            raise HTTPException(status_code=403, detail="This feature is not included in your license")
        return principal
    return dep
