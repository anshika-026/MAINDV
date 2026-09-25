"""SQLite storage for the body-appearance Re-ID unique-footfall engine —
mirrors peopleid_db.py's structure/conventions (same shared `app.db` file,
same get_connection()/init_db() pattern) but with its OWN tables. Kept
separate from peopleid_persons/peopleid_embeddings deliberately: a face
embedding (peopleid_*) and a body-appearance embedding (reid_*) are
different-dimension, different-space vectors that must never be compared
against each other, and this keeps the existing face-based People
Identification feature completely untouched.

Unlike peopleid_persons (always human-named via enrollment), a person here
is auto-created with no human input — `label` is a generated "PERSON_001"
style identifier, not a real name (spec sections 2/7/31: identity creation
is fully automatic; a human can rename a person later via update_person,
same as any auto-generated label).
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"
SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reid_snapshots"

TRACK_STATE_UNKNOWN = "unknown"
TRACK_STATE_CANDIDATE = "candidate"
TRACK_STATE_CONFIRMED = "confirmed"

EVENT_NEW_PERSON = "new_person"
EVENT_SIGHTING = "sighting"


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
            CREATE TABLE IF NOT EXISTS reid_persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL UNIQUE,
                display_name TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reid_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL REFERENCES reid_persons(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                quality_score REAL,
                source_camera INTEGER,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reid_embeddings_person ON reid_embeddings(person_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reid_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL REFERENCES reid_persons(id) ON DELETE CASCADE,
                camera_id INTEGER,
                file_path TEXT NOT NULL,
                quality_score REAL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reid_snapshots_person ON reid_snapshots(person_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reid_tracks (
                camera_id INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                person_id INTEGER,
                state TEXT NOT NULL DEFAULT 'unknown',
                current_confidence REAL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                PRIMARY KEY (camera_id, track_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reid_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                track_id INTEGER,
                camera_id INTEGER NOT NULL,
                confidence REAL,
                ts REAL NOT NULL,
                event_type TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reid_events_camera_ts ON reid_events(camera_id, ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reid_events_person_ts ON reid_events(person_id, ts)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reid_camera_config (
                camera_id INTEGER PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 0,
                similarity_threshold REAL,
                quality_min_score REAL,
                mot_interval_seconds REAL,
                roi TEXT,
                updated_at REAL NOT NULL
            )
            """
        )


# --- Persons ----------------------------------------------------------------

def create_person(now: float | None = None) -> int:
    """Auto-creates a new identity with a generated PERSON_NNN label — no
    human input required (spec section 2/31). The label is derived from the
    row id AFTER insert (SQLite's AUTOINCREMENT), so it's assigned in a
    second UPDATE rather than guessed beforehand."""
    now = now if now is not None else time.time()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO reid_persons (label, status, first_seen, last_seen, created_at, updated_at) "
            "VALUES ('', 'active', ?, ?, ?, ?)",
            (now, now, now, now),
        )
        person_id = cur.lastrowid
        label = f"PERSON_{person_id:03d}"
        conn.execute("UPDATE reid_persons SET label = ? WHERE id = ?", (label, person_id))
        return person_id


def touch_person(person_id: int, now: float | None = None) -> None:
    now = now if now is not None else time.time()
    with get_connection() as conn:
        conn.execute("UPDATE reid_persons SET last_seen = ?, updated_at = ? WHERE id = ?", (now, now, person_id))


def update_person(person_id: int, **fields) -> None:
    allowed = {"display_name", "status"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with get_connection() as conn:
        conn.execute(f"UPDATE reid_persons SET {set_clause}, updated_at = ? WHERE id = ?", (*updates.values(), time.time(), person_id))


def delete_person(person_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM reid_embeddings WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM reid_snapshots WHERE person_id = ?", (person_id,))
        conn.execute("UPDATE reid_tracks SET person_id = NULL WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM reid_persons WHERE id = ?", (person_id,))


def get_person(person_id: int) -> dict | None:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM reid_persons WHERE id = ?", (person_id,)).fetchone()
    return dict(row) if row else None


def list_persons(status: str | None = "active") -> list[dict]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        query = (
            "SELECT p.*, "
            "(SELECT COUNT(*) FROM reid_embeddings e WHERE e.person_id = p.id) AS embedding_count, "
            "(SELECT COUNT(*) FROM reid_snapshots s WHERE s.person_id = p.id) AS snapshot_count "
            "FROM reid_persons p"
        )
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY last_seen DESC"
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# --- Embeddings ---------------------------------------------------------------

def add_embedding(person_id: int, embedding: np.ndarray, quality_score: float | None = None, source_camera: int | None = None) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO reid_embeddings (person_id, embedding, quality_score, source_camera, created_at) VALUES (?, ?, ?, ?, ?)",
            (person_id, embedding.astype(np.float32).tobytes(), quality_score, source_camera, time.time()),
        )
        return cur.lastrowid


def load_all_embeddings() -> list[tuple[int, int, np.ndarray]]:
    """(embedding_row_id, person_id, embedding) for every stored sample —
    the raw material reid_gallery's VectorGallery builds its search matrix
    from. Only ever called in-process, never returned from an API handler."""
    with get_connection() as conn:
        rows = conn.execute("SELECT id, person_id, embedding FROM reid_embeddings").fetchall()
    return [(row_id, person_id, np.frombuffer(blob, dtype=np.float32)) for row_id, person_id, blob in rows]


def embedding_count_for_person(person_id: int) -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM reid_embeddings WHERE person_id = ?", (person_id,)).fetchone()[0]


# --- Snapshots ------------------------------------------------------------

def add_snapshot(person_id: int, camera_id: int | None, file_path: str, quality_score: float | None = None) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO reid_snapshots (person_id, camera_id, file_path, quality_score, created_at) VALUES (?, ?, ?, ?, ?)",
            (person_id, camera_id, file_path, quality_score, time.time()),
        )
        return cur.lastrowid


def list_snapshots_for_person(person_id: int) -> list[dict]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM reid_snapshots WHERE person_id = ? ORDER BY created_at", (person_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def list_snapshots_bulk(person_ids: list[int], per_person: int = 4) -> dict[int, list[dict]]:
    """Up to `per_person` snapshots for each of `person_ids`, oldest first.

    Avoids the N+1 the acceptance-test screen would otherwise cause: it
    re-reads every identity every few seconds, and one query per person
    would mean dozens of round trips per refresh.
    """
    if not person_ids:
        return {}
    placeholders = ",".join("?" * len(person_ids))
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM reid_snapshots WHERE person_id IN ({placeholders}) ORDER BY person_id, created_at",
            person_ids,
        ).fetchall()
    out: dict[int, list[dict]] = {pid: [] for pid in person_ids}
    for row in rows:
        bucket = out[row["person_id"]]
        if len(bucket) < per_person:
            bucket.append(dict(row))
    return out


def prune_expired_snapshots(retention_days: int) -> int:
    """Deletes snapshot ROWS older than retention_days — caller is
    responsible for also deleting the referenced file on disk (this module
    doesn't touch the filesystem beyond SNAPSHOTS_DIR's own path
    convention). Returns the deleted row count. 0/negative retention_days
    means "keep forever" (mirrors CLIP_RETENTION_DAYS's own convention) —
    no-op."""
    if retention_days <= 0:
        return 0
    cutoff = time.time() - retention_days * 86400
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, file_path FROM reid_snapshots WHERE created_at < ?", (cutoff,)).fetchall()
        conn.execute("DELETE FROM reid_snapshots WHERE created_at < ?", (cutoff,))
    return len(rows)


# --- Tracks -----------------------------------------------------------------

def upsert_track(camera_id: int, track_id: int, state: str, person_id: int | None, confidence: float | None, now: float | None = None) -> None:
    now = now if now is not None else time.time()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO reid_tracks (camera_id, track_id, person_id, state, current_confidence, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(camera_id, track_id) DO UPDATE SET "
            "person_id=excluded.person_id, state=excluded.state, current_confidence=excluded.current_confidence, "
            "last_seen=excluded.last_seen",
            (camera_id, track_id, person_id, state, confidence, now, now),
        )


def list_active_tracks(camera_id: int | None = None, within_seconds: float = 30.0) -> list[dict]:
    cutoff = time.time() - within_seconds
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        if camera_id is not None:
            rows = conn.execute("SELECT * FROM reid_tracks WHERE camera_id = ? AND last_seen >= ?", (camera_id, cutoff)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM reid_tracks WHERE last_seen >= ?", (cutoff,)).fetchall()
    return [dict(r) for r in rows]


def prune_stale_tracks(older_than_seconds: float) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM reid_tracks WHERE last_seen < ?", (time.time() - older_than_seconds,))


# --- Events / unique footfall -----------------------------------------------

def log_event(camera_id: int, track_id: int, event_type: str, person_id: int | None = None, confidence: float | None = None, now: float | None = None) -> int:
    now = now if now is not None else time.time()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO reid_events (person_id, track_id, camera_id, confidence, ts, event_type) VALUES (?, ?, ?, ?, ?, ?)",
            (person_id, track_id, camera_id, confidence, now, event_type),
        )
        return cur.lastrowid


def list_events(camera_id: int | None = None, person_id: int | None = None, start_ts: float | None = None, end_ts: float | None = None, limit: int = 200) -> list[dict]:
    clauses, params = [], []
    if camera_id is not None:
        clauses.append("camera_id = ?")
        params.append(camera_id)
    if person_id is not None:
        clauses.append("person_id = ?")
        params.append(person_id)
    if start_ts is not None:
        clauses.append("ts >= ?")
        params.append(start_ts)
    if end_ts is not None:
        clauses.append("ts < ?")
        params.append(end_ts)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT e.*, p.label AS person_label, p.display_name FROM reid_events e "
            f"LEFT JOIN reid_persons p ON p.id = e.person_id {where} ORDER BY ts DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def _day_start(ts: float | None = None) -> float:
    import datetime

    dt = datetime.datetime.fromtimestamp(ts if ts is not None else time.time())
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def count_unique_today(now: float | None = None) -> int:
    """Spec sections 12/13's "daily unique footfall": distinct CONFIRMED
    person_id with at least one sighting since local midnight. Resets
    naturally every day just by the WHERE clause moving forward — no
    separate reset job needed (unlike the old face-based footfall_counter.py
    + footfall_report.py's explicit end-of-day finalize job, which exists
    because THAT module's dedup window is short-lived in-memory state, not
    a durable per-day count derived straight from a timestamped event log)."""
    day_start = _day_start(now)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT person_id) FROM reid_events WHERE person_id IS NOT NULL AND ts >= ?", (day_start,)
        ).fetchone()
    return row[0] or 0


def reset_identities(delete_snapshot_files: bool = True) -> dict[str, int]:
    """Clears the whole identity registry so counting restarts from
    PERSON_001. Returns how many rows each table lost.

    Exists for two real needs, not as a convenience: acceptance testing
    (a UAT run has to start from a known-empty state to be meaningful),
    and changing the embedding model (embeddings from a different model
    are in a different vector space, so old identities can never be
    matched again and would sit there inflating the count forever).

    Touches only the reid_* tables — cameras, enrolled faces, attendance
    and alerts are untouched. Per-camera Re-ID config is deliberately
    KEPT, so a reset doesn't silently turn the engine off.
    """
    removed: dict[str, int] = {}
    files_removed = 0
    if delete_snapshot_files:
        with get_connection() as conn:
            paths = [r[0] for r in conn.execute("SELECT file_path FROM reid_snapshots")]
        for raw in paths:
            path = Path(raw)
            if not path.is_absolute():
                path = SNAPSHOTS_DIR / path.name
            try:
                path.unlink(missing_ok=True)
                files_removed += 1
            except OSError:
                pass  # a locked/missing file must not abort the reset

    with get_connection() as conn:
        for table in ("reid_events", "reid_tracks", "reid_snapshots", "reid_embeddings", "reid_persons"):
            removed[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            conn.execute(f"DELETE FROM {table}")
        # Restart labelling at PERSON_001 rather than continuing the old run.
        conn.execute("DELETE FROM sqlite_sequence WHERE name LIKE 'reid_%'")
    removed["snapshot_files"] = files_removed
    return removed


def count_unique_lifetime() -> int:
    """Spec section 13's "lifetime unique footfall": every person this
    engine has ever confirmed, ever. Equivalent to counting distinct
    person_id across all of reid_events, but reading reid_persons directly
    is cheaper and exactly equivalent under this module's own invariant
    that a reid_persons row is only ever created together with its first
    confirmed sighting (see reid_worker.py)."""
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM reid_persons").fetchone()[0]


def count_new_today(now: float | None = None) -> int:
    day_start = _day_start(now)
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM reid_persons WHERE first_seen >= ?", (day_start,)).fetchone()[0]


def count_returning_today(now: float | None = None) -> int:
    """Seen today (>=1 event today) but NOT new today (first_seen before
    today) — spec section 19's "Returning People Today" stat tile."""
    day_start = _day_start(now)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT e.person_id) FROM reid_events e JOIN reid_persons p ON p.id = e.person_id "
            "WHERE e.ts >= ? AND p.first_seen < ?",
            (day_start, day_start),
        ).fetchone()
    return row[0] or 0


def footfall_history(days: int = 14) -> list[dict]:
    """[{"date": "YYYY-MM-DD", "unique_count": N}, ...] for the last `days`
    days, oldest first — backs the /api/reid/footfall/history chart. Uses
    SQLite's own date() function on the stored unix timestamp rather than
    a separate finalized-report file (see count_unique_today's docstring on
    why this module doesn't need footfall_report.py's finalize-job
    pattern)."""
    cutoff = time.time() - days * 86400
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT date(ts, 'unixepoch', 'localtime') AS day, COUNT(DISTINCT person_id) AS unique_count "
            "FROM reid_events WHERE person_id IS NOT NULL AND ts >= ? GROUP BY day ORDER BY day",
            (cutoff,),
        ).fetchall()
    return [{"date": r[0], "unique_count": r[1]} for r in rows]


# --- Per-camera config (incl. ROI) ------------------------------------------

def get_camera_config(camera_id: int) -> dict:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM reid_camera_config WHERE camera_id = ?", (camera_id,)).fetchone()
    if row is None:
        return {
            "camera_id": camera_id, "enabled": False, "similarity_threshold": None,
            "quality_min_score": None, "mot_interval_seconds": None, "roi": None,
        }
    d = dict(row)
    d["enabled"] = bool(d["enabled"])
    d["roi"] = json.loads(d["roi"]) if d["roi"] else None
    return d


def list_camera_configs() -> list[dict]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM reid_camera_config").fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["enabled"] = bool(d["enabled"])
        d["roi"] = json.loads(d["roi"]) if d["roi"] else None
        out.append(d)
    return out


def set_camera_config(
    camera_id: int, enabled: bool, similarity_threshold: float | None = None,
    quality_min_score: float | None = None, mot_interval_seconds: float | None = None,
    roi: list[list[float]] | None = None,
) -> None:
    """roi: a polygon as [[x, y], ...] in fractional (0..1) frame
    coordinates (same convention as desk_db.py's zones — survives a camera
    resolution change), or None to count the whole frame (no ROI
    restriction, the default)."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO reid_camera_config (camera_id, enabled, similarity_threshold, quality_min_score, mot_interval_seconds, roi, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(camera_id) DO UPDATE SET enabled=excluded.enabled, similarity_threshold=excluded.similarity_threshold, "
            "quality_min_score=excluded.quality_min_score, mot_interval_seconds=excluded.mot_interval_seconds, "
            "roi=excluded.roi, updated_at=excluded.updated_at",
            (camera_id, int(enabled), similarity_threshold, quality_min_score, mot_interval_seconds,
             json.dumps(roi) if roi else None, time.time()),
        )
