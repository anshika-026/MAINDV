"""
face_db.py

Raw sqlite3 storage for:
  - employees: local, minimal roster (employee_id, name) used to validate
    labels server-side without depending on the external face-enrollment
    service at request time. See FACE_TRAINING.md for why this exists.
  - face_embeddings: the enrollment "gallery" (many embeddings per person_id,
    since each accepted capture adds another vector — no retraining needed).
  - face_pending: unrecognized / low-confidence tracked faces waiting for a
    human to assign a person_id (live-correction queue, unchanged).
  - face_training_captures: EVERY finished track's best frame, regardless of
    match confidence — the bulk dataset for the manual labeling UI + the
    classifier training pipeline. Deliberately separate from face_pending,
    which only ever sees low-confidence captures.

Matches the style of camera_db.py — no ORM, CREATE TABLE IF NOT EXISTS,
module-level connection helper.
"""

import sqlite3
import json
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_face_tables():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            embedding TEXT NOT NULL,        -- JSON list[float], 512-dim ArcFace vector
            source_image_path TEXT,
            enrolled_at REAL NOT NULL
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_face_embeddings_person
        ON face_embeddings(person_id)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_pending (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            captured_at REAL NOT NULL,
            image_path TEXT NOT NULL,
            embedding TEXT NOT NULL,        -- JSON list[float]
            best_match_person_id TEXT,      -- nullable: top candidate below threshold
            best_match_score REAL,
            assigned_person_id TEXT,        -- nullable until a human assigns it
            status TEXT NOT NULL DEFAULT 'pending'  -- pending | assigned | ignored
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_face_pending_status
        ON face_pending(status)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_face_pending_track
        ON face_pending(camera_id, track_id)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_training_captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL,
            track_id INTEGER,
            captured_at REAL NOT NULL,
            image_path TEXT NOT NULL,
            embedding TEXT,                  -- JSON list[float], null if embedding failed
            detection_confidence REAL,
            blur_score REAL,                 -- variance of Laplacian; higher = sharper
            brightness REAL,                 -- mean grayscale intensity, 0-255
            employee_id TEXT,                -- nullable until labeled
            label_status TEXT NOT NULL DEFAULT 'unlabeled',  -- unlabeled | labeled | skipped
            labeled_at REAL
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_captures_status
        ON face_training_captures(label_status, captured_at)
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Gallery (enrolled embeddings)
# ---------------------------------------------------------------------------

def add_embedding(person_id: str, embedding: list[float], source_image_path: str | None = None):
    conn = get_conn()
    conn.execute(
        """INSERT INTO face_embeddings (person_id, embedding, source_image_path, enrolled_at)
           VALUES (?, ?, ?, ?)""",
        (person_id, json.dumps(embedding), source_image_path, time.time()),
    )
    conn.commit()
    conn.close()


def get_all_embeddings() -> list[dict]:
    """Returns the full gallery as [{person_id, embedding: list[float]}, ...].
    Loaded into memory and refreshed periodically by the recognition worker —
    fine at this scale (a plain cosine-similarity loop), no vector DB needed
    unless enrollment grows into the thousands."""
    conn = get_conn()
    rows = conn.execute("SELECT person_id, embedding FROM face_embeddings").fetchall()
    conn.close()
    return [{"person_id": r["person_id"], "embedding": json.loads(r["embedding"])} for r in rows]


def count_embeddings_for_person(person_id: str) -> int:
    conn = get_conn()
    n = conn.execute(
        "SELECT COUNT(*) AS c FROM face_embeddings WHERE person_id = ?", (person_id,)
    ).fetchone()["c"]
    conn.close()
    return n


# ---------------------------------------------------------------------------
# Pending review queue
# ---------------------------------------------------------------------------

def add_pending(
    camera_id: int,
    track_id: int,
    image_path: str,
    embedding: list[float],
    best_match_person_id: str | None = None,
    best_match_score: float | None = None,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO face_pending
           (camera_id, track_id, captured_at, image_path, embedding,
            best_match_person_id, best_match_score, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
        (
            camera_id,
            track_id,
            time.time(),
            image_path,
            json.dumps(embedding),
            best_match_person_id,
            best_match_score,
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def has_pending_for_track(camera_id: int, track_id: int, since_ts: float) -> bool:
    """Avoid spamming the review queue with the same track every frame —
    check whether this track already has a pending/assigned entry recently."""
    conn = get_conn()
    row = conn.execute(
        """SELECT 1 FROM face_pending
           WHERE camera_id = ? AND track_id = ? AND captured_at > ?
           LIMIT 1""",
        (camera_id, track_id, since_ts),
    ).fetchone()
    conn.close()
    return row is not None


def get_pending(status: str = "pending", hours: int = 24) -> list[dict]:
    cutoff = time.time() - hours * 3600
    conn = get_conn()
    rows = conn.execute(
        """SELECT id, camera_id, track_id, captured_at, image_path,
                  best_match_person_id, best_match_score, assigned_person_id, status
           FROM face_pending
           WHERE status = ? AND captured_at > ?
           ORDER BY captured_at DESC""",
        (status, cutoff),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def assign_pending(pending_id: int, person_id: str):
    """Human confirms identity: mark this capture assigned AND fold its
    embedding into the gallery so future frames match faster/more confidently.
    This is the entire 'training' step — no model retraining involved."""
    conn = get_conn()
    row = conn.execute(
        "SELECT embedding, image_path FROM face_pending WHERE id = ?", (pending_id,)
    ).fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"No pending capture with id={pending_id}")

    embedding = json.loads(row["embedding"])

    conn.execute(
        """UPDATE face_pending SET status = 'assigned', assigned_person_id = ?
           WHERE id = ?""",
        (person_id, pending_id),
    )
    conn.commit()
    conn.close()

    add_embedding(person_id, embedding, source_image_path=row["image_path"])


def ignore_pending(pending_id: int):
    conn = get_conn()
    conn.execute("UPDATE face_pending SET status = 'ignored' WHERE id = ?", (pending_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Employees — local roster, used only to validate labels server-side.
# Not the system of record for HR data; just enough to reject a typo'd ID
# before it silently starts a new, wrong training class.
# ---------------------------------------------------------------------------

def upsert_employee(employee_id: str, name: str) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO employees (employee_id, name) VALUES (?, ?)
           ON CONFLICT(employee_id) DO UPDATE SET name = excluded.name""",
        (employee_id, name),
    )
    conn.commit()
    conn.close()


def employee_exists(employee_id: str) -> bool:
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
    conn.close()
    return row is not None


def list_employees() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT employee_id, name FROM employees ORDER BY employee_id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_employees() -> int:
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) AS c FROM employees").fetchone()["c"]
    conn.close()
    return n


# ---------------------------------------------------------------------------
# Training captures — the bulk manual-labeling dataset.
# ---------------------------------------------------------------------------

def add_training_capture(
    camera_id: int,
    track_id: int | None,
    image_path: str,
    embedding: list[float] | None,
    detection_confidence: float | None = None,
    blur_score: float | None = None,
    brightness: float | None = None,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO face_training_captures
           (camera_id, track_id, captured_at, image_path, embedding,
            detection_confidence, blur_score, brightness, label_status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unlabeled')""",
        (
            camera_id,
            track_id,
            time.time(),
            image_path,
            json.dumps(embedding) if embedding is not None else None,
            detection_confidence,
            blur_score,
            brightness,
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_training_capture(capture_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM face_training_captures WHERE id = ?", (capture_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_next_unlabeled_capture() -> dict | None:
    """Oldest-first, so a long backlog gets worked through in capture order
    rather than newest-first (which would let old captures rot forever)."""
    conn = get_conn()
    row = conn.execute(
        """SELECT * FROM face_training_captures
           WHERE label_status = 'unlabeled'
           ORDER BY captured_at ASC
           LIMIT 1"""
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_training_stats() -> dict:
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) AS c FROM face_training_captures").fetchone()["c"]
    reviewed = conn.execute(
        "SELECT COUNT(*) AS c FROM face_training_captures WHERE label_status != 'unlabeled'"
    ).fetchone()["c"]
    conn.close()
    return {"reviewed": reviewed, "total": total}


def label_training_capture(capture_id: int, employee_id: str, new_image_path: str) -> None:
    """Single transaction: a capture never ends up half-labeled (e.g. row
    updated but file move failed, or vice versa) — the caller moves the file
    and passes the already-final path in; if this raises, the caller is
    expected to not have left the file moved (see face_training_routes.py)."""
    conn = get_conn()
    try:
        cur = conn.execute(
            "SELECT label_status FROM face_training_captures WHERE id = ?", (capture_id,)
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"No training capture with id={capture_id}")
        if row["label_status"] != "unlabeled":
            raise ValueError(f"Capture {capture_id} is already {row['label_status']}")

        conn.execute(
            """UPDATE face_training_captures
               SET label_status = 'labeled', employee_id = ?, labeled_at = ?, image_path = ?
               WHERE id = ?""",
            (employee_id, time.time(), new_image_path, capture_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def skip_training_capture(capture_id: int) -> None:
    conn = get_conn()
    try:
        cur = conn.execute(
            "SELECT label_status FROM face_training_captures WHERE id = ?", (capture_id,)
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"No training capture with id={capture_id}")
        conn.execute(
            "UPDATE face_training_captures SET label_status = 'skipped' WHERE id = ?",
            (capture_id,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_labeled_training_embeddings() -> list[dict]:
    """Camera-derived, human-labeled embeddings only — the actual training
    set for the classifier. Distinct from get_all_embeddings(), which is the
    live-recognition gallery (enrollment photos + assigned review captures)."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT employee_id, embedding FROM face_training_captures
           WHERE label_status = 'labeled' AND embedding IS NOT NULL"""
    ).fetchall()
    conn.close()
    return [{"person_id": r["employee_id"], "embedding": json.loads(r["embedding"])} for r in rows]
