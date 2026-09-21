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
  - collection_sessions: persistent state for a multi-day background
    collection run (start/end time, which cameras, status, activity) so a
    backend restart can detect and resume an unfinished session instead of
    silently losing it. See FACE_TRAINING.md "7-day collection sessions".

Matches the style of camera_db.py — no ORM, CREATE TABLE IF NOT EXISTS,
module-level connection helper.
"""

import sqlite3
import json
import threading
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"

# Guards the "count all captures, then insert" sequence in
# face_pipeline.py's _save_training_capture() so MAX_TRAINING_CAPTURES can
# never be exceeded even though multiple camera threads write concurrently.
# One process, one lock — sufficient here since this app runs as a single
# uvicorn worker (see BACKEND_HANDOFF.md), not multiple processes.
capture_limit_lock = threading.Lock()


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

    # The People page's roster (name, photos, enrollment) is read live from a
    # separate external service (see face_routes.py/client.js) that this app
    # has no write access to — there's no update API for it. Its employee_id
    # field is frequently null/stale, and the People page's own "Save" button
    # previously only updated in-memory React state (never persisted
    # anywhere), so an edited ID reverted on the next refresh. This table is
    # the actual persistence for that ID field: keyed by the external
    # service's name string (the only stable shared identifier available),
    # it overrides whatever employee_id that service reports.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS people_employee_id_overrides (
            name TEXT PRIMARY KEY,
            employee_id TEXT NOT NULL,
            updated_at REAL NOT NULL
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
            -- unlabeled | labeled | skipped: normal, human-facing queue states.
            -- no_embedding | rejected: never shown in /face-training, never
            -- trainable — set at insert time by face_pipeline.py, see
            -- FACE_TRAINING.md's "Quality filtering & capture statuses".
            label_status TEXT NOT NULL DEFAULT 'unlabeled',
            labeled_at REAL
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_captures_status
        ON face_training_captures(label_status, captured_at)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_captures_camera
        ON face_training_captures(camera_id)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS collection_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at REAL NOT NULL,
            planned_end_at REAL NOT NULL,
            camera_ids TEXT NOT NULL,        -- JSON list[int]
            status TEXT NOT NULL DEFAULT 'running',  -- running | completed | stopped
            last_activity_at REAL,
            duplicates_rejected INTEGER NOT NULL DEFAULT 0,
            completed_at REAL
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_collection_sessions_status
        ON collection_sessions(status)
    """)

    # One row per completed `python -m app.train_faces` / POST /training/train
    # run — see face_training.train_classifier(). Never written on a failed
    # run (nothing to report), never written to except by that one call site.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at REAL NOT NULL,
            sample_count INTEGER NOT NULL,
            class_count INTEGER NOT NULL,
            -- excluded_* are nullable: a run backfilled from before this
            -- history table existed may not have these diagnostic counts
            -- available — NULL means "not recorded", never a guessed 0.
            excluded_no_embedding INTEGER,
            excluded_rejected INTEGER,
            excluded_too_few_samples INTEGER,
            validation_samples INTEGER,
            validation_classes INTEGER,
            validation_accuracy REAL,
            model_path TEXT NOT NULL,
            per_employee_counts TEXT NOT NULL  -- JSON {employee_id: count}
        )
    """)
    # Added later: per-class validation breakdown (accuracy/false-accept/
    # false-reject per employee, plus a confusion matrix) — a real run from
    # before this existed just won't have one (NULL), never backfilled with
    # a guessed value. SQLite has no "ADD COLUMN IF NOT EXISTS"; catching
    # the duplicate-column error is the standard idempotent pattern here.
    try:
        cur.execute("ALTER TABLE training_runs ADD COLUMN per_class_validation TEXT")
    except sqlite3.OperationalError:
        pass

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


def set_person_employee_id(name: str, employee_id: str) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO people_employee_id_overrides (name, employee_id, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET employee_id = excluded.employee_id, updated_at = excluded.updated_at""",
        (name, employee_id, time.time()),
    )
    conn.commit()
    conn.close()


def get_person_employee_id_overrides() -> dict[str, str]:
    conn = get_conn()
    rows = conn.execute("SELECT name, employee_id FROM people_employee_id_overrides").fetchall()
    conn.close()
    return {r["name"]: r["employee_id"] for r in rows}


def delete_employees_not_in(employee_ids: set[str]) -> int:
    """Removes local roster rows whose ID is no longer returned by the
    external face-enrollment service — otherwise a renamed/renumbered ID
    (e.g. 018 -> 118 for the same person) would leave the old ID stuck in
    the datalist/validation forever, since upsert_employee only ever adds or
    updates, never removes. Safe to call: this table is just a local mirror
    for validating labels server-side (see the module docstring), not the
    source of truth, and it has no foreign key into face_training_captures —
    a capture already labeled under a since-removed ID keeps that label
    untouched, this only prunes the roster used for *new* labels.

    Caller's responsibility: pass an empty set only when that's genuinely
    correct (the external service really has zero IDs) — this will delete
    every local row in that case. sync_employees() in face_training_routes.py
    guards against calling this with an accidentally-empty set from a
    malformed/partial API response."""
    conn = get_conn()
    if not employee_ids:
        n = conn.execute("SELECT COUNT(*) AS c FROM employees").fetchone()["c"]
        conn.execute("DELETE FROM employees")
    else:
        placeholders = ",".join("?" * len(employee_ids))
        rows = conn.execute(
            f"SELECT employee_id FROM employees WHERE employee_id NOT IN ({placeholders})",
            tuple(employee_ids),
        ).fetchall()
        n = len(rows)
        if n:
            conn.execute(
                f"DELETE FROM employees WHERE employee_id NOT IN ({placeholders})",
                tuple(employee_ids),
            )
    conn.commit()
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
    label_status: str = "unlabeled",
) -> int:
    """`label_status` defaults to 'unlabeled' (the normal case) but
    face_pipeline.py passes 'no_embedding' or 'rejected' for captures that
    should never reach the labeling queue — see the label_status comment on
    the table definition above."""
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO face_training_captures
           (camera_id, track_id, captured_at, image_path, embedding,
            detection_confidence, blur_score, brightness, label_status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            camera_id,
            track_id,
            time.time(),
            image_path,
            json.dumps(embedding) if embedding is not None else None,
            detection_confidence,
            blur_score,
            brightness,
            label_status,
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def count_training_captures() -> int:
    """Total across ALL cameras and ALL statuses — the number
    MAX_TRAINING_CAPTURES is checked against. Must be called while holding
    capture_limit_lock when used as part of the insert-or-reject decision."""
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) AS c FROM face_training_captures").fetchone()["c"]
    conn.close()
    return n


def get_capture_counts_by_status() -> dict:
    """Diagnostic breakdown, e.g. {"unlabeled": 12, "labeled": 3, "skipped": 1,
    "no_embedding": 8, "rejected": 2}. Used for verification/reporting, not
    by the labeling UI itself."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT label_status, COUNT(*) AS c FROM face_training_captures GROUP BY label_status"
    ).fetchall()
    conn.close()
    return {r["label_status"]: r["c"] for r in rows}


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
    """`reviewed`/`total` are scoped to the human-facing queue only
    ('unlabeled'/'labeled'/'skipped') — 'no_embedding' and 'rejected'
    captures were never shown to a human, so counting them here would
    inflate the /face-training "X / Y reviewed" counter with captures
    nobody actually reviewed. `all_captures`/`by_status` cover everything,
    for diagnostics."""
    conn = get_conn()
    total = conn.execute(
        "SELECT COUNT(*) AS c FROM face_training_captures WHERE label_status IN ('unlabeled','labeled','skipped')"
    ).fetchone()["c"]
    reviewed = conn.execute(
        "SELECT COUNT(*) AS c FROM face_training_captures WHERE label_status IN ('labeled','skipped')"
    ).fetchone()["c"]
    all_captures = conn.execute("SELECT COUNT(*) AS c FROM face_training_captures").fetchone()["c"]
    conn.close()
    return {
        "reviewed": reviewed,
        "total": total,
        "all_captures": all_captures,
        "by_status": get_capture_counts_by_status(),
    }


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


def relabel_training_capture(capture_id: int, employee_id: str, new_image_path: str) -> dict:
    """Corrects an already-labeled or skipped capture to a different
    employee_id — the undo/fix for a mistyped ID. Same file-then-DB contract
    as label_training_capture: the caller (face_training_routes.py) has
    already moved the file to new_image_path before calling this. Returns
    the row as it was BEFORE the update, so the caller (which already did
    the move) can react if this raises."""
    conn = get_conn()
    try:
        cur = conn.execute("SELECT * FROM face_training_captures WHERE id = ?", (capture_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"No training capture with id={capture_id}")
        if row["label_status"] not in ("labeled", "skipped"):
            raise ValueError(
                f"Capture {capture_id} is {row['label_status']} — only a labeled or skipped capture can be corrected"
            )
        conn.execute(
            """UPDATE face_training_captures
               SET label_status = 'labeled', employee_id = ?, labeled_at = ?, image_path = ?
               WHERE id = ?""",
            (employee_id, time.time(), new_image_path, capture_id),
        )
        conn.commit()
        return dict(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def unlabel_training_capture(capture_id: int, new_image_path: str) -> dict:
    """Reverts a labeled/skipped capture back to 'unlabeled' so it re-enters
    the queue for someone to redo — plain undo, no employee_id guessed.
    Same file-then-DB contract as label_training_capture. Returns the row as
    it was BEFORE the update."""
    conn = get_conn()
    try:
        cur = conn.execute("SELECT * FROM face_training_captures WHERE id = ?", (capture_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"No training capture with id={capture_id}")
        if row["label_status"] not in ("labeled", "skipped"):
            raise ValueError(f"Capture {capture_id} is {row['label_status']} — nothing to undo")
        conn.execute(
            """UPDATE face_training_captures
               SET label_status = 'unlabeled', employee_id = NULL, labeled_at = NULL, image_path = ?
               WHERE id = ?""",
            (new_image_path, capture_id),
        )
        conn.commit()
        return dict(row)
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


def count_labeled_since(ts: float) -> int:
    """Number of captures labeled strictly after `ts` — used by
    face_training_scheduler.py to decide whether enough new human-labeled
    data has accumulated since the last training run to justify another one.
    ts=0 (no prior run) counts every labeled row ever."""
    conn = get_conn()
    n = conn.execute(
        "SELECT COUNT(*) AS c FROM face_training_captures WHERE label_status = 'labeled' AND labeled_at > ?",
        (ts,),
    ).fetchone()["c"]
    conn.close()
    return n


def count_captures_since(ts: float) -> int:
    """Total new raw captures (any label_status) since `ts` — distinct from
    count_labeled_since, which only counts ones a human has since labeled.
    Used for the "samples added today" status field: collection can add
    captures much faster than they get labeled, so both numbers matter."""
    conn = get_conn()
    n = conn.execute(
        "SELECT COUNT(*) AS c FROM face_training_captures WHERE captured_at > ?", (ts,)
    ).fetchone()["c"]
    conn.close()
    return n


def get_recent_labeled_captures(limit: int = 8) -> list[dict]:
    """Newest-first labeled captures, for the labeling page's correction
    list (see /recent-labels) — lets someone fix a just-typed wrong ID
    without hunting through the dataset."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT * FROM face_training_captures
           WHERE label_status = 'labeled'
           ORDER BY labeled_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_labeled_training_embeddings() -> list[dict]:
    """Camera-derived, human-labeled embeddings only — the actual training
    set for the classifier. Distinct from get_all_embeddings(), which is the
    live-recognition gallery (enrollment photos + assigned review captures).

    Includes camera_id/track_id (not just person_id/embedding) so a
    train/validation split can group same-track captures together instead of
    splitting them across both sides — several captures of one track are
    near-duplicates of the same appearance, so putting some in train and
    others in validation would leak information rather than measure
    generalization. See face_training.py's split logic."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT id, employee_id, embedding, camera_id, track_id FROM face_training_captures
           WHERE label_status = 'labeled' AND embedding IS NOT NULL"""
    ).fetchall()
    conn.close()
    return [
        {
            "capture_id": r["id"],
            "person_id": r["employee_id"],
            "embedding": json.loads(r["embedding"]),
            "camera_id": r["camera_id"],
            "track_id": r["track_id"],
        }
        for r in rows
    ]


def count_training_captures_for_camera(camera_id: int) -> int:
    """Same atomicity contract as count_training_captures() — call while
    holding capture_limit_lock when used as part of an insert-or-reject
    decision (see MAX_CAPTURES_PER_CAMERA in face_pipeline.py)."""
    conn = get_conn()
    n = conn.execute(
        "SELECT COUNT(*) AS c FROM face_training_captures WHERE camera_id = ?", (camera_id,)
    ).fetchone()["c"]
    conn.close()
    return n


def get_per_camera_capture_breakdown() -> dict:
    """{camera_id: {"total":.., "usable":.., "no_embedding":.., "rejected":..,
    "labeled":.., "skipped":.., "last_capture_at":..}} — for the monitoring
    endpoint. "usable" means label_status='unlabeled' (awaiting review, has
    a real embedding, passed quality) to match the wording operators care
    about, distinct from the DB's internal 'unlabeled' status name."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT camera_id, label_status, COUNT(*) AS c, MAX(captured_at) AS last_at
           FROM face_training_captures
           GROUP BY camera_id, label_status"""
    ).fetchall()
    conn.close()

    breakdown: dict[int, dict] = {}
    for r in rows:
        cam = breakdown.setdefault(
            r["camera_id"],
            {"total": 0, "usable": 0, "no_embedding": 0, "rejected": 0,
             "labeled": 0, "skipped": 0, "last_capture_at": None},
        )
        status = r["label_status"]
        key = "usable" if status == "unlabeled" else status
        if key in cam:
            cam[key] = r["c"]
        cam["total"] += r["c"]
        if cam["last_capture_at"] is None or (r["last_at"] or 0) > cam["last_capture_at"]:
            cam["last_capture_at"] = r["last_at"]
    return breakdown


# ---------------------------------------------------------------------------
# Collection sessions — persistent state for a multi-day background run.
# ---------------------------------------------------------------------------

def _session_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["camera_ids"] = json.loads(d["camera_ids"])
    return d


def create_collection_session(started_at: float, planned_end_at: float, camera_ids: list[int]) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO collection_sessions
           (started_at, planned_end_at, camera_ids, status, last_activity_at)
           VALUES (?, ?, ?, 'running', ?)""",
        (started_at, planned_end_at, json.dumps(camera_ids), started_at),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_collection_session(session_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM collection_sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return _session_row_to_dict(row) if row else None


def get_running_collection_session() -> dict | None:
    """At most one row should ever be 'running' at a time — enforced by
    application logic (start_session() reuses/extends it instead of
    creating a second one), not a DB constraint."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM collection_sessions WHERE status = 'running' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return _session_row_to_dict(row) if row else None


def update_collection_session_plan(session_id: int, camera_ids: list[int], planned_end_at: float) -> None:
    """Extending/reusing an already-running session (e.g. calling
    /collection/start again with a new camera list or duration) rather than
    creating a second concurrent 'running' row."""
    conn = get_conn()
    conn.execute(
        "UPDATE collection_sessions SET camera_ids = ?, planned_end_at = ? WHERE id = ?",
        (json.dumps(camera_ids), planned_end_at, session_id),
    )
    conn.commit()
    conn.close()


def touch_running_collection_session() -> None:
    """Called on every successful capture insert — 'last successful
    activity' for monitoring/health, independent of label_status."""
    conn = get_conn()
    conn.execute(
        "UPDATE collection_sessions SET last_activity_at = ? WHERE status = 'running'",
        (time.time(),),
    )
    conn.commit()
    conn.close()


def increment_running_session_duplicates_rejected() -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE collection_sessions SET duplicates_rejected = duplicates_rejected + 1 WHERE status = 'running'"
    )
    conn.commit()
    conn.close()


def add_training_run(
    sample_count: int,
    class_count: int,
    excluded_no_embedding: int,
    excluded_rejected: int,
    excluded_too_few_samples: int,
    model_path: str,
    per_employee_counts: dict[str, int],
    validation_samples: int | None = None,
    validation_classes: int | None = None,
    validation_accuracy: float | None = None,
    per_class_validation: dict | None = None,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO training_runs
           (trained_at, sample_count, class_count, excluded_no_embedding,
            excluded_rejected, excluded_too_few_samples, validation_samples,
            validation_classes, validation_accuracy, model_path, per_employee_counts,
            per_class_validation)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            time.time(), sample_count, class_count, excluded_no_embedding,
            excluded_rejected, excluded_too_few_samples, validation_samples,
            validation_classes, validation_accuracy, model_path, json.dumps(per_employee_counts),
            json.dumps(per_class_validation) if per_class_validation is not None else None,
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def list_training_runs(limit: int = 20) -> list[dict]:
    """Newest-first training history — see /training-history."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM training_runs ORDER BY trained_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["per_employee_counts"] = json.loads(d["per_employee_counts"])
        d["per_class_validation"] = json.loads(d["per_class_validation"]) if d.get("per_class_validation") else None
        out.append(d)
    return out


def finish_collection_session(session_id: int, status: str) -> None:
    """status: 'completed' (planned end time reached) or 'stopped' (manual)."""
    conn = get_conn()
    conn.execute(
        "UPDATE collection_sessions SET status = ?, completed_at = ? WHERE id = ?",
        (status, time.time(), session_id),
    )
    conn.commit()
    conn.close()
