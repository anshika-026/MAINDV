"""
audit.py

Append-only record of security-relevant admin actions (logins, settings and
analytics switches, camera and license changes), queryable later and also
written to the application log. Never stores passwords, tokens or images.
"""

from __future__ import annotations

import json
import logging
import time

from . import config, db

log = logging.getLogger("audit")

DB_PATH = config.DB_PATH


def init_db() -> None:
    with db.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                actor TEXT,
                action TEXT NOT NULL,
                target TEXT,
                details TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log (ts)")


def _actor(principal: dict | None) -> str | None:
    if not principal:
        return None
    return principal.get("email") or (f"license:{principal['license_id']}" if principal.get("license_id") else None)


def record(action: str, principal: dict | None = None, target: str | None = None, **details) -> None:
    """Best effort by design: an audit write failing must never fail the
    action itself, but it is logged loudly."""
    actor = _actor(principal)
    log.info("audit action=%s actor=%s target=%s details=%s", action, actor, target, details or "")
    try:
        with db.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO audit_log (ts, actor, action, target, details) VALUES (?, ?, ?, ?, ?)",
                (time.time(), actor, action, target, json.dumps(details, default=str) if details else None),
            )
    except Exception:
        log.exception("could not write audit record for %s", action)


def recent(limit: int = 200) -> list[dict]:
    import sqlite3

    with db.connect(DB_PATH, row_factory=sqlite3.Row) as conn:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY ts DESC LIMIT ?", (max(1, min(limit, 1000)),)).fetchall()
    return [dict(r) for r in rows]
