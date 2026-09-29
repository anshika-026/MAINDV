"""
One-off migration: rewrite image paths stored in the database to the
portable form (relative to DATA_DIR, see app/storage.py).

Rows written before the storage change hold absolute paths from whichever
machine wrote them (C:\\Users\\<someone>\\...\\backend\\data\\face_training\\...).
The app reads those fine as long as the file is under the current DATA_DIR,
but they break again the next time the data directory moves. This makes
them relative once.

Run from backend/ with the backend STOPPED:

    python -m scripts.migrate_paths            # dry run: counts only
    python -m scripts.migrate_paths --apply    # backs up app.db, then rewrites

Idempotent. Values whose file is outside DATA_DIR are left unchanged.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, storage  # noqa: E402

COLUMNS = (
    ("face_training_captures", "image_path"),
    ("face_pending", "image_path"),
    ("face_embeddings", "source_image_path"),
    ("reid_snapshots", "file_path"),
    ("alerts", "snapshot_path"),
)


def migrate(db_path: Path, apply: bool) -> dict:
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        report = {}
        for table, col in COLUMNS:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                continue
            rows = conn.execute(f"SELECT rowid, {col} FROM {table} WHERE {col} IS NOT NULL").fetchall()
            changes = []
            missing = 0
            for rowid, value in rows:
                new = storage.normalize(value)
                if new != value:
                    changes.append((new, rowid))
                if not storage.exists(new):
                    missing += 1
            report[table] = {"rows": len(rows), "rewritten": len(changes), "file_missing": missing}
            if apply and changes:
                conn.executemany(f"UPDATE {table} SET {col} = ? WHERE rowid = ?", changes)
        if apply:
            conn.commit()
        return report
    finally:
        conn.close()


def backup(db_path: Path) -> Path:
    target = db_path.parent / "backups" / f"app-before-path-migration-{time.strftime('%Y%m%d-%H%M%S')}.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--db", default=str(config.DB_PATH))
    args = ap.parse_args(argv)
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"No database at {db_path}")
        return 1
    print(f"DATA_DIR={config.DATA_DIR}\nDB={db_path}\n{'APPLY' if args.apply else 'DRY RUN'}")
    if args.apply:
        print(f"backup: {backup(db_path)}")
    for table, r in migrate(db_path, args.apply).items():
        print(f"  {table:24} rows={r['rows']:6}  rewritten={r['rewritten']:6}  file_missing={r['file_missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
