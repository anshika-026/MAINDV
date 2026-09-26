"""SQLite storage for the People Identification module — a NEW, parallel
set of tables in the same shared `backend/data/app.db` file every other
`*_db.py` module already uses (see face_db.py's DB_PATH), following the
identical get_connection()/init_db() convention. Deliberately separate from
`enrolled_faces`/`detection_events` (face_db.py): those tables back the
existing Honeywell-Allow-List-driven Attendance/Analytics feature and must
not be touched or duplicated-into by this module (see the People
Identification plan's Context section).

A person here can have many `peopleid_embeddings` rows (the whole point —
see peopleid_enrollment.py's curation pipeline, which aims for 10-20
diverse, quality-gated reference embeddings per person instead of the single
low-quality photo enrolled_faces often has today).
"""

import contextlib
import sqlite3
import time
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"
# Unknown-person review photos (spec section 25) — a NEW, separate
# directory from ENROLLMENT_PHOTOS_DIR (main.py), which backs the
# UNRELATED existing People/enrolled_faces feature. Not mounted as public
# StaticFiles like that one — served through an authenticated
# peopleid_api.py endpoint instead (spec section 34: access-controlled
# images, not a public URL).
PHOTOS_DIR = Path(__file__).resolve().parent.parent / "data" / "peopleid_photos"

TRACK_STATE_UNKNOWN = "unknown"
TRACK_STATE_CANDIDATE = "candidate"
TRACK_STATE_CONFIRMED = "confirmed"


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
            CREATE TABLE IF NOT EXISTS peopleid_persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                employee_id TEXT,
                department TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS peopleid_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL REFERENCES peopleid_persons(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                quality_score REAL,
                pose_label TEXT,
                source TEXT,
                photo_path TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_peopleid_embeddings_person ON peopleid_embeddings(person_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS peopleid_tracks (
                track_id INTEGER NOT NULL,
                camera_id INTEGER NOT NULL,
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
            CREATE TABLE IF NOT EXISTS peopleid_events (
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_peopleid_events_camera_ts ON peopleid_events(camera_id, ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_peopleid_events_person ON peopleid_events(person_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS peopleid_unknown_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                best_photo TEXT,
                embedding_centroid BLOB,
                observation_count INTEGER NOT NULL DEFAULT 0,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                assigned_person_id INTEGER
            )
            """
        )
        # Per-camera overrides (spec section 29's "camera-specific
        # calibration") — a row only ever means "this camera diverges from
        # config.py's PEOPLEID_* defaults"; a camera with no row here uses
        # the global defaults untouched. `enabled` gates whether
        # PipelineManager._start_worker turns People-ID on for that camera
        # at all — off (no row / enabled=0) is the safe default, matching
        # FOOTFALL_CAMERAS' existing opt-in-per-camera convention.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS peopleid_camera_config (
                camera_id INTEGER PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 0,
                similarity_threshold REAL,
                quality_min_score REAL,
                mot_interval_seconds REAL,
                updated_at REAL NOT NULL
            )
            """
        )


# --- Persons --------------------------------------------------------------

def create_person(name: str, employee_id: str | None = None, department: str | None = None) -> int:
    now = time.time()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO peopleid_persons (name, employee_id, department, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'active', ?, ?)",
            (name, employee_id, department, now, now),
        )
        return cur.lastrowid


def update_person(person_id: int, **fields) -> None:
    allowed = {"name", "employee_id", "department", "status"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with get_connection() as conn:
        conn.execute(
            f"UPDATE peopleid_persons SET {set_clause}, updated_at = ? WHERE id = ?",
            (*updates.values(), time.time(), person_id),
        )


def delete_person(person_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM peopleid_embeddings WHERE person_id = ?", (person_id,))
        conn.execute("UPDATE peopleid_tracks SET person_id = NULL WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM peopleid_persons WHERE id = ?", (person_id,))


def get_person(person_id: int) -> dict | None:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM peopleid_persons WHERE id = ?", (person_id,)).fetchone()
    return dict(row) if row else None


def list_persons(status: str | None = "active") -> list[dict]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        if status:
            rows = conn.execute(
                "SELECT p.*, (SELECT COUNT(*) FROM peopleid_embeddings e WHERE e.person_id = p.id) AS embedding_count "
                "FROM peopleid_persons p WHERE status = ? ORDER BY name", (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT p.*, (SELECT COUNT(*) FROM peopleid_embeddings e WHERE e.person_id = p.id) AS embedding_count "
                "FROM peopleid_persons p ORDER BY name",
            ).fetchall()
    return [dict(r) for r in rows]


# --- Embeddings -------------------------------------------------------------

def add_embedding(
    person_id: int, embedding: np.ndarray, quality_score: float | None = None,
    pose_label: str | None = None, source: str = "upload", photo_path: str | None = None,
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO peopleid_embeddings (person_id, embedding, quality_score, pose_label, source, photo_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (person_id, embedding.astype(np.float32).tobytes(), quality_score, pose_label, source, photo_path, time.time()),
        )
        return cur.lastrowid


def delete_embedding(embedding_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM peopleid_embeddings WHERE id = ?", (embedding_id,))


def list_embeddings_for_person(person_id: int) -> list[dict]:
    """Metadata only — deliberately never includes the raw embedding BLOB,
    which should never leave this process over the API (see the module
    plan's security requirement)."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, person_id, quality_score, pose_label, source, photo_path, created_at "
            "FROM peopleid_embeddings WHERE person_id = ? ORDER BY created_at",
            (person_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_all_embeddings() -> list[tuple[int, int, np.ndarray]]:
    """(embedding_row_id, person_id, embedding) for every enrolled sample —
    the raw material peopleid_gallery.VectorGallery builds its search matrix
    from. Only ever called in-process (worker/gallery build), never returned
    from an API handler."""
    with get_connection() as conn:
        rows = conn.execute("SELECT id, person_id, embedding FROM peopleid_embeddings").fetchall()
    return [(row_id, person_id, np.frombuffer(blob, dtype=np.float32)) for row_id, person_id, blob in rows]


# --- Tracks -----------------------------------------------------------------

def upsert_track(
    camera_id: int, track_id: int, state: str, person_id: int | None,
    confidence: float | None, now: float | None = None,
) -> None:
    now = now if now is not None else time.time()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO peopleid_tracks (camera_id, track_id, person_id, state, current_confidence, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(camera_id, track_id) DO UPDATE SET "
            "person_id=excluded.person_id, state=excluded.state, current_confidence=excluded.current_confidence, "
            "last_seen=excluded.last_seen",
            (camera_id, track_id, person_id, state, confidence, now, now),
        )


def list_active_tracks(camera_id: int, within_seconds: float = 30.0) -> list[dict]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM peopleid_tracks WHERE camera_id = ? AND last_seen >= ?",
            (camera_id, time.time() - within_seconds),
        ).fetchall()
    return [dict(r) for r in rows]


def prune_stale_tracks(older_than_seconds: float) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM peopleid_tracks WHERE last_seen < ?", (time.time() - older_than_seconds,))


# --- Events -------------------------------------------------------------

def log_event(
    camera_id: int, track_id: int, event_type: str,
    person_id: int | None = None, confidence: float | None = None,
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO peopleid_events (person_id, track_id, camera_id, confidence, ts, event_type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (person_id, track_id, camera_id, confidence, time.time(), event_type),
        )
        return cur.lastrowid


def list_events(
    camera_id: int | None = None, person_id: int | None = None,
    start_ts: float | None = None, end_ts: float | None = None, limit: int = 200,
) -> list[dict]:
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
            f"SELECT e.*, p.name AS person_name FROM peopleid_events e "
            f"LEFT JOIN peopleid_persons p ON p.id = e.person_id {where} "
            f"ORDER BY ts DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# --- Unknown-person review ------------------------------------------------

def upsert_unknown_cluster(
    cluster_id: int | None, camera_id: int, embedding_centroid: np.ndarray,
    best_photo: str | None, now: float | None = None,
) -> int:
    now = now if now is not None else time.time()
    with get_connection() as conn:
        if cluster_id is None:
            cur = conn.execute(
                "INSERT INTO peopleid_unknown_clusters "
                "(camera_id, best_photo, embedding_centroid, observation_count, first_seen, last_seen) "
                "VALUES (?, ?, ?, 1, ?, ?)",
                (camera_id, best_photo, embedding_centroid.astype(np.float32).tobytes(), now, now),
            )
            return cur.lastrowid
        conn.execute(
            "UPDATE peopleid_unknown_clusters SET embedding_centroid = ?, observation_count = observation_count + 1, "
            "last_seen = ?, best_photo = COALESCE(?, best_photo) WHERE id = ?",
            (embedding_centroid.astype(np.float32).tobytes(), now, best_photo, cluster_id),
        )
        return cluster_id


def list_unknown_clusters(include_assigned: bool = False) -> list[dict]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        if include_assigned:
            rows = conn.execute("SELECT * FROM peopleid_unknown_clusters ORDER BY last_seen DESC").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM peopleid_unknown_clusters WHERE assigned_person_id IS NULL ORDER BY last_seen DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def get_unknown_cluster(cluster_id: int) -> dict | None:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM peopleid_unknown_clusters WHERE id = ?", (cluster_id,)).fetchone()
    return dict(row) if row else None


def assign_unknown_cluster(cluster_id: int, person_id: int) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE peopleid_unknown_clusters SET assigned_person_id = ? WHERE id = ?", (person_id, cluster_id))


def get_camera_config(camera_id: int) -> dict:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM peopleid_camera_config WHERE camera_id = ?", (camera_id,)).fetchone()
    if row is None:
        return {
            "camera_id": camera_id, "enabled": False, "similarity_threshold": None,
            "quality_min_score": None, "mot_interval_seconds": None,
        }
    d = dict(row)
    d["enabled"] = bool(d["enabled"])
    return d


def list_camera_configs() -> list[dict]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM peopleid_camera_config").fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["enabled"] = bool(d["enabled"])
        out.append(d)
    return out


def set_camera_config(
    camera_id: int, enabled: bool, similarity_threshold: float | None = None,
    quality_min_score: float | None = None, mot_interval_seconds: float | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO peopleid_camera_config "
            "(camera_id, enabled, similarity_threshold, quality_min_score, mot_interval_seconds, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(camera_id) DO UPDATE SET enabled=excluded.enabled, "
            "similarity_threshold=excluded.similarity_threshold, quality_min_score=excluded.quality_min_score, "
            "mot_interval_seconds=excluded.mot_interval_seconds, updated_at=excluded.updated_at",
            (camera_id, int(enabled), similarity_threshold, quality_min_score, mot_interval_seconds, time.time()),
        )


def find_unknown_cluster_near(camera_id: int, embedding: np.ndarray, similarity_threshold: float) -> int | None:
    """Best-matching NOT-yet-assigned unknown cluster on this camera whose
    centroid is within similarity_threshold of `embedding`, or None — lets
    peopleid_worker fold repeat sightings of the same unrecognized person
    into one cluster instead of creating a new row every observation."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, embedding_centroid FROM peopleid_unknown_clusters "
            "WHERE camera_id = ? AND assigned_person_id IS NULL", (camera_id,),
        ).fetchall()
    best_id, best_sim = None, similarity_threshold
    query = embedding / (np.linalg.norm(embedding) + 1e-8)
    for row in rows:
        centroid = np.frombuffer(row["embedding_centroid"], dtype=np.float32)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        sim = float(np.dot(query, centroid))
        if sim >= best_sim:
            best_id, best_sim = row["id"], sim
    return best_id
