"""
retention.py

Keeps runtime data from growing forever. Runs hourly (RETENTION_INTERVAL_SECONDS)
in a background thread, and on demand via `python -m app.manage retention`.

What it removes (each rule disabled by setting its setting to 0):

  - expired login sessions                                  (always)
  - review-queue captures (face_pending) that were never
    assigned, older than FACE_PENDING_RETENTION_DAYS         row + image
  - training captures that are NOT part of the training set
    (unlabeled / rejected / no_embedding), older than
    FACE_CAPTURE_RETENTION_DAYS                               row + image
  - Re-ID body snapshots older than SNAPSHOT_RETENTION_DAYS   row + image
  - snapshots of alerts resolved more than
    ALERT_SNAPSHOT_RETENTION_DAYS ago                         image (alert row kept)
  - classifier backups beyond MODEL_BACKUP_RETENTION_COUNT
  - orphaned image files (no DB row points at them) older than a day, in the
    capture/snapshot folders only

What it NEVER touches: labeled or skipped training captures (the training
set), enrollment photos, the active classifier, assigned review captures
(their image is the source of an enrolled embedding), or any file outside
DATA_DIR. Deletions are batched so a big first run can't hold the database
lock for long.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from . import config, db, lifecycle, storage

log = logging.getLogger("retention")

BATCH = 500
MAX_DELETIONS_PER_RULE = 20_000
ORPHAN_MIN_AGE_SECONDS = 86_400
ORPHAN_DIRS = ("face_captures", "face_training/_unlabeled", "reid_snapshots", "alert_snapshots")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _db_path():
    return config.DB_PATH


def _inside_data_dir(p: Path) -> bool:
    try:
        p.resolve().relative_to(Path(config.DATA_DIR).resolve())
        return True
    except ValueError:
        return False


def _delete_file(stored: str | None) -> bool:
    p = storage.resolve(stored)
    if p is None or not _inside_data_dir(p):
        return False
    try:
        p.unlink(missing_ok=True)
        return True
    except OSError as e:
        log.warning("could not delete %s: %s", p.name, e)
        return False


def _table_exists(conn, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _purge_rows(table: str, path_col: str, where: str, params: tuple) -> int:
    """Deletes matching rows and their image files, in batches."""
    removed = 0
    while removed < MAX_DELETIONS_PER_RULE:
        with db.connect(_db_path()) as conn:
            if not _table_exists(conn, table):
                return 0
            rows = conn.execute(f"SELECT id, {path_col} FROM {table} WHERE {where} LIMIT {BATCH}", params).fetchall()
            if not rows:
                break
            conn.executemany(f"DELETE FROM {table} WHERE id = ?", [(r[0],) for r in rows])
        for _, path in rows:  # files after the rows are gone: never a row pointing at a missing file
            _delete_file(path)
        removed += len(rows)
        if len(rows) < BATCH or lifecycle.stopping():
            break
    return removed


def _days_ago(days: int, now: float) -> float:
    return now - days * 86_400


def purge_face_pending(now: float) -> int:
    if config.FACE_PENDING_RETENTION_DAYS <= 0:
        return 0
    return _purge_rows("face_pending", "image_path", "status IN ('pending', 'ignored') AND captured_at < ?",
                       (_days_ago(config.FACE_PENDING_RETENTION_DAYS, now),))


def purge_training_captures(now: float) -> int:
    if config.FACE_CAPTURE_RETENTION_DAYS <= 0:
        return 0
    return _purge_rows("face_training_captures", "image_path",
                       "label_status IN ('unlabeled', 'rejected', 'no_embedding') AND captured_at < ?",
                       (_days_ago(config.FACE_CAPTURE_RETENTION_DAYS, now),))


def purge_reid_snapshots(now: float) -> int:
    if config.SNAPSHOT_RETENTION_DAYS <= 0:
        return 0
    return _purge_rows("reid_snapshots", "file_path", "created_at < ?", (_days_ago(config.SNAPSHOT_RETENTION_DAYS, now),))


def purge_alert_snapshots(now: float) -> int:
    if config.ALERT_SNAPSHOT_RETENTION_DAYS <= 0:
        return 0
    cutoff = _days_ago(config.ALERT_SNAPSHOT_RETENTION_DAYS, now)
    with db.connect(_db_path()) as conn:
        if not _table_exists(conn, "alerts"):
            return 0
        rows = conn.execute(
            "SELECT id, snapshot_path FROM alerts WHERE snapshot_path IS NOT NULL AND status = 'Resolved' "
            "AND resolved_at IS NOT NULL AND resolved_at < ? LIMIT ?", (cutoff, MAX_DELETIONS_PER_RULE)).fetchall()
        conn.executemany("UPDATE alerts SET snapshot_path = NULL WHERE id = ?", [(r[0],) for r in rows])
    for _, path in rows:
        _delete_file(path)
    return len(rows)


def purge_classifier_backups() -> int:
    from . import classifier_io
    from .face_pipeline import CLASSIFIER_PATH

    return len(classifier_io.rotate_backups(CLASSIFIER_PATH))


def _referenced_files() -> set[Path]:
    refs: set[Path] = set()
    with db.connect(_db_path()) as conn:
        for table, col in (("face_pending", "image_path"), ("face_training_captures", "image_path"),
                           ("face_embeddings", "source_image_path"), ("reid_snapshots", "file_path"),
                           ("alerts", "snapshot_path")):
            if not _table_exists(conn, table):
                continue
            for (v,) in conn.execute(f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL"):
                p = storage.resolve(v)
                if p is not None:
                    refs.add(p.resolve())
    return refs


def purge_orphans(now: float) -> int:
    refs = _referenced_files()
    removed = 0
    data = Path(config.DATA_DIR)
    for sub in ORPHAN_DIRS:
        d = data / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if removed >= MAX_DELETIONS_PER_RULE:
                return removed
            if not f.is_file() or f.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                if now - f.stat().st_mtime < ORPHAN_MIN_AGE_SECONDS or f.resolve() in refs:
                    continue
                f.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def run_once(now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    from . import auth

    results: dict[str, int | str] = {}
    # Expired sessions are always purged; everything else only when retention
    # is enabled (RETENTION_ENABLED, on by default).
    rules = [("sessions", lambda: auth.purge_expired_sessions(now))]
    if config.RETENTION_ENABLED:
        rules += [
            ("face_pending", lambda: purge_face_pending(now)),
            ("training_captures", lambda: purge_training_captures(now)),
            ("reid_snapshots", lambda: purge_reid_snapshots(now)),
            ("alert_snapshots", lambda: purge_alert_snapshots(now)),
            ("classifier_backups", purge_classifier_backups),
            ("orphan_files", lambda: purge_orphans(now)),
        ]
    for name, fn in rules:
        try:
            results[name] = fn()
        except Exception as e:  # one failing rule must not stop the others
            log.exception("retention rule %s failed", name)
            results[name] = f"error: {type(e).__name__}"
    if any(isinstance(v, int) and v for v in results.values()):
        log.info("retention: %s", ", ".join(f"{k}={v}" for k, v in results.items()))
    return results


_thread: threading.Thread | None = None


def start() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return

    def loop():
        # First pass shortly after startup, then on the interval.
        if lifecycle.wait(60):
            return
        while True:
            run_once()
            if lifecycle.wait(config.RETENTION_INTERVAL_SECONDS):
                return

    _thread = threading.Thread(target=loop, daemon=True, name="retention")
    _thread.start()
