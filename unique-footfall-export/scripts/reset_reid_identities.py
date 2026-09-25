"""Clears the Re-ID identity registry (persons, embeddings, snapshots,
tracks, events) so unique footfall starts counting from zero.

Run from backend/:
    python -m scripts.reset_reid_identities --yes

When you need this: identities are only as good as the embedding model that
created them. After changing REID_MODEL_PATH (or the similarity threshold),
every previously stored identity was built in a DIFFERENT vector space —
old embeddings and new ones are not comparable, so returning visitors can
never match their old identity and the historical counts stay wrong
forever. Resetting is part of changing the model, not an optional cleanup.

This does NOT touch cameras, enrolled faces, attendance, alerts or any
other table — only the reid_* ones. Snapshot image files on disk are left
in place by default (they're pruned on the normal retention schedule);
pass --delete-snapshots to remove them too.

Take a copy of backend/data/app.db first if you might want the old
identities back — this is not reversible from inside the app.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import reid_db  # noqa: E402

# Child rows first — embeddings/snapshots/events reference a person.
TABLES = ["reid_events", "reid_tracks", "reid_snapshots", "reid_embeddings", "reid_persons"]


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="actually delete (otherwise just reports)")
    parser.add_argument("--delete-snapshots", action="store_true", help="also delete snapshot .jpg files on disk")
    args = parser.parse_args()

    with sqlite3.connect(reid_db.DB_PATH) as conn:
        before = counts(conn)
        print(f"database: {reid_db.DB_PATH}")
        print("current rows:")
        for table, n in before.items():
            print(f"  {table:<18} {n:>6}")

        if not any(before.values()):
            print("\nAlready empty — nothing to do.")
            return 0

        if not args.yes:
            print("\nDry run. Re-run with --yes to delete the rows above.")
            return 0

        for table in TABLES:
            conn.execute(f"DELETE FROM {table}")
        # Restart PERSON numbering at 1 so the next identity is PERSON_001.
        conn.execute("DELETE FROM sqlite_sequence WHERE name LIKE 'reid_%'")
        after = counts(conn)

    print("\nafter:")
    for table, n in after.items():
        print(f"  {table:<18} {n:>6}")

    if args.delete_snapshots:
        removed = 0
        for path in Path(reid_db.SNAPSHOTS_DIR).glob("*.jpg"):
            path.unlink()
            removed += 1
        print(f"\ndeleted {removed} snapshot file(s) from {reid_db.SNAPSHOTS_DIR}")

    print(f"\nunique today   : {reid_db.count_unique_today()}")
    print(f"unique lifetime: {reid_db.count_unique_lifetime()}")
    print("\nRestart the backend so each camera worker rebuilds its in-memory gallery.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
