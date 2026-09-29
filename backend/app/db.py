"""
db.py

The single way this app opens its SQLite database. Every module used to call
sqlite3.connect() itself with its own (or no) timeout; under load that
surfaced as "database is locked" and as connections left open when a query
raised. `connect()` gives every caller the same behaviour:

  - a generous lock timeout plus busy_timeout, so concurrent camera threads
    wait for each other instead of failing,
  - WAL-friendly synchronous=NORMAL,
  - commit on success, rollback on any exception, and ALWAYS close.

Usage:
    with db.connect(DB_PATH) as conn:
        conn.execute(...)

Modules keep their own module-level DB_PATH (tests point it at a temp file);
it defaults to config.DB_PATH.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Iterator

from . import config

LOCK_TIMEOUT_SECONDS = 30.0


def open_connection(path: str | Path | None = None, row_factory=None, check_same_thread: bool = True) -> sqlite3.Connection:
    """A configured connection the caller must close. Prefer connect()."""
    p = Path(path or config.DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=LOCK_TIMEOUT_SECONDS, check_same_thread=check_same_thread)
    try:
        conn.execute(f"PRAGMA busy_timeout = {int(LOCK_TIMEOUT_SECONDS * 1000)}")
        conn.execute("PRAGMA synchronous = NORMAL")
    except sqlite3.Error:
        conn.close()
        raise
    if row_factory is not None:
        conn.row_factory = row_factory
    return conn


@contextlib.contextmanager
def connect(path: str | Path | None = None, row_factory=None) -> Iterator[sqlite3.Connection]:
    """Transaction-scoped connection: commits if the block succeeds, rolls
    back if it raises, and is closed either way."""
    conn = open_connection(path, row_factory=row_factory)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def check(path: str | Path | None = None) -> tuple[bool, str]:
    """Readiness probe: can the database be opened and queried?"""
    try:
        with connect(path) as conn:
            conn.execute("SELECT 1").fetchone()
        return True, "ok"
    except Exception as e:  # reported, never raised: this feeds /ready
        return False, type(e).__name__
