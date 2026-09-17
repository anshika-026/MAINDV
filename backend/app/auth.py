"""
auth.py

Session-token authentication shared by the admin portal and the client
portal. This is the missing foundation the rest of the client-license
system depends on: before this file existed, `POST /api/auth/login`
issued no real token at all (the frontend just hardcoded the string
"local-session" client-side — see BACKEND_HANDOFF.md), and every other
endpoint in this app (cameras, sites, licenses, face pipeline) accepted
requests from anyone, with zero verification. Tenant isolation (a client
only seeing their own cameras) was previously enforced ONLY in the React
UI, which is not a security boundary — anyone could call the same APIs
directly and see everything.

Two principal types:
  - "admin": anyone who has completed POST /api/auth/login. IMPORTANT
    LIMITATION: that endpoint still does not check a password (see its
    own docstring in main.py) — a valid admin session proves "this
    browser completed the existing login flow", not "this is a verified
    administrator". Building real admin passwords is a separate, larger
    piece of work (a user/roles table, signup, password reset, ...) that
    is out of scope here; see the accompanying report for why this is
    called out explicitly rather than silently left as-is.
  - "client": a license's own username/password, verified against a
    PBKDF2 hash (license_db.verify_license_login) — this IS a real
    credential check, and a client session is re-validated against the
    live license status/expiry on every single request (see
    load_active_client_license), not just at login.

Sessions are opaque random tokens (not JWTs) stored server-side in a
`sessions` table, matching the rest of this codebase's plain-sqlite3
style (camera_db.py / face_db.py / license_db.py) — no new dependency,
and trivially revocable (delete the row), unlike a self-contained JWT.
"""

from __future__ import annotations

import contextlib
import secrets
import sqlite3
import time
from pathlib import Path

from fastapi import Depends, Header, HTTPException

from . import license_db

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"

SESSION_TTL_SECONDS = 7 * 86400  # 7 days

ADMIN = "admin"
CLIENT = "client"


@contextlib.contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


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


def _new_token() -> str:
    return secrets.token_urlsafe(32)


def create_admin_session(email: str) -> str:
    token = _new_token()
    now = time.time()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (token, principal_type, email, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (token, ADMIN, email, now, now + SESSION_TTL_SECONDS),
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


def get_session(token: str) -> dict | None:
    """Public (unlike a leading-underscore helper) because the websocket
    routes in main.py need to resolve a token passed as a query param —
    browsers cannot attach an Authorization header to a WebSocket
    handshake, so those routes can't use the get_principal dependency
    below and look tokens up directly instead."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()
    if row is None:
        return None
    session = dict(row)
    if session["expires_at"] < time.time():
        revoke_session(token)
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
    session = get_session(token) if token else None
    if session is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session


def require_admin(principal: dict = Depends(get_principal)) -> dict:
    """FastAPI dependency: admin-only endpoint (camera/site/license
    management, the face pipeline's admin tooling). A client session
    reaching one of these gets a 403, not a silent frontend redirect."""
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
