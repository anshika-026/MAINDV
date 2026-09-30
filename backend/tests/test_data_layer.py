"""Database helper, portable image paths, retention, path migration."""

import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from app import config, db, storage


# --- db.connect ---------------------------------------------------------------

def test_successful_block_commits_and_closes(tmp_path):
    path = tmp_path / "t.db"
    with db.connect(path) as conn:
        conn.execute("CREATE TABLE t (x)")
        conn.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")  # closed
    with db.connect(path) as c2:
        assert c2.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1


def test_failed_block_rolls_back_and_still_closes(tmp_path):
    path = tmp_path / "t.db"
    with db.connect(path) as conn:
        conn.execute("CREATE TABLE t (x)")
    with pytest.raises(RuntimeError):
        with db.connect(path) as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            raise RuntimeError("mid-transaction failure")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
    with db.connect(path) as c2:
        assert c2.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0


def test_failed_query_rolls_back(tmp_path):
    path = tmp_path / "t.db"
    with db.connect(path) as conn:
        conn.execute("CREATE TABLE t (x UNIQUE)")
        conn.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect(path) as conn:
            conn.execute("INSERT INTO t VALUES (2)")
            conn.execute("INSERT INTO t VALUES (1)")
    with db.connect(path) as c2:
        assert [r[0] for r in c2.execute("SELECT x FROM t")] == [1]


def test_connections_get_busy_timeout(tmp_path):
    with db.connect(tmp_path / "t.db") as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 30_000


def test_concurrent_writers_wait_instead_of_failing(tmp_path):
    path = tmp_path / "t.db"
    with db.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (x)")
    errors = []

    def writer(n):
        try:
            for i in range(25):
                with db.connect(path) as c:
                    c.execute("INSERT INTO t VALUES (?)", (n * 1000 + i,))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    with db.connect(path) as c:
        assert c.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 200


def test_connection_failure_is_reported_by_check(tmp_path):
    ok, _ = db.check(tmp_path / "ok.db")
    assert ok
    bad = tmp_path / "is_a_dir.db"
    bad.mkdir()
    ok, detail = db.check(bad)
    assert not ok and detail


def test_face_db_functions_close_connections_on_error(isolated_db, monkeypatch):
    """A failing query inside a converted face_db function must not leave a
    write transaction open (which would block every other writer)."""
    from app import face_db

    face_db.add_embedding("018", [0.1] * 4, "face_enroll/x.jpg")
    with pytest.raises(ValueError):
        face_db.assign_pending(99999, "018")
    # Another writer can proceed immediately.
    face_db.upsert_employee("019", "Someone")
    assert face_db.employee_exists("019")


# --- storage: portable paths ------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    (d / "face_training" / "018").mkdir(parents=True)
    (d / "face_training" / "018" / "a.jpg").write_bytes(b"x")
    monkeypatch.setattr(config, "DATA_DIR", d)
    return d


def test_paths_are_stored_relative(data_dir):
    assert storage.to_stored(data_dir / "face_training" / "018" / "a.jpg") == "face_training/018/a.jpg"


def test_relative_paths_resolve_under_data_dir(data_dir):
    assert storage.resolve("face_training/018/a.jpg") == (data_dir / "face_training" / "018" / "a.jpg").resolve()


@pytest.mark.parametrize("legacy", [
    r"C:\Users\lenovo\Desktop\DECO_VISION\backend\data\face_training\018\a.jpg",
    "/home/ubuntu/Deco-vision/backend/data/face_training/018/a.jpg",
    r"backend/data/face_training\018\a.jpg",
])
def test_legacy_paths_from_other_machines_resolve(data_dir, legacy):
    assert storage.exists(legacy)
    assert storage.normalize(legacy) == "face_training/018/a.jpg"


@pytest.mark.parametrize("evil", ["../../etc/passwd", "face_training/../../../secret", "..\\..\\x"])
def test_relative_values_can_never_escape_data_dir(data_dir, evil):
    assert storage.resolve(evil) is None


def test_files_outside_data_dir_are_kept_absolute(data_dir, tmp_path):
    outside = tmp_path / "elsewhere.jpg"
    outside.write_bytes(b"x")
    assert storage.to_stored(outside) == str(outside.resolve())
    assert storage.resolve(str(outside)) == outside


def test_moving_the_data_directory_keeps_every_image_reachable(data_dir, tmp_path, monkeypatch):
    stored = storage.to_stored(data_dir / "face_training" / "018" / "a.jpg")
    moved = tmp_path / "somewhere" / "else" / "data"
    shutil.move(str(data_dir), str(moved))
    monkeypatch.setattr(config, "DATA_DIR", moved)
    assert storage.exists(stored)


# --- retention ---------------------------------------------------------------------

@pytest.fixture
def retention_env(isolated_db, tmp_path, monkeypatch):
    from app import face_db, retention

    for k, v in dict(FACE_PENDING_RETENTION_DAYS=30, FACE_CAPTURE_RETENTION_DAYS=60, SNAPSHOT_RETENTION_DAYS=30,
                     ALERT_SNAPSHOT_RETENTION_DAYS=90, RETENTION_ENABLED=True).items():
        monkeypatch.setattr(config, k, v)
    now = time.time()
    old, recent = now - 100 * 86400, now - 86400

    def img(rel):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"jpg")
        return rel

    with db.connect(isolated_db) as c:
        for status, ts, name in (("pending", old, "old_pending"), ("ignored", old, "old_ignored"),
                                 ("assigned", old, "old_assigned"), ("pending", recent, "new_pending")):
            c.execute("INSERT INTO face_pending (camera_id, track_id, captured_at, image_path, embedding, status) VALUES (1,1,?,?, '[]', ?)",
                      (ts, img(f"face_captures/{name}.jpg"), status))
        for status, ts, name in (("labeled", old, "lab"), ("skipped", old, "skip"), ("unlabeled", old, "unl"),
                                 ("rejected", old, "rej"), ("no_embedding", old, "noemb"), ("unlabeled", recent, "unl_new")):
            c.execute("INSERT INTO face_training_captures (camera_id, captured_at, image_path, label_status) VALUES (1,?,?,?)",
                      (ts, img(f"face_training/_unlabeled/{name}.jpg"), status))
    face_db  # noqa: B018 (imported for table creation via isolated_db)
    return retention, tmp_path, now


def _exists(root, rel):
    return (root / rel).exists()


def test_retention_removes_only_old_non_training_data(retention_env):
    retention, root, now = retention_env
    result = retention.run_once(now)
    assert result["face_pending"] == 2 and result["training_captures"] == 3
    with db.connect(config.DB_PATH) as c:
        pending = {r[0] for r in c.execute("SELECT image_path FROM face_pending")}
        caps = {r[0] for r in c.execute("SELECT image_path FROM face_training_captures")}
    assert pending == {"face_captures/old_assigned.jpg", "face_captures/new_pending.jpg"}
    assert caps == {"face_training/_unlabeled/lab.jpg", "face_training/_unlabeled/skip.jpg", "face_training/_unlabeled/unl_new.jpg"}
    assert not _exists(root, "face_captures/old_pending.jpg")
    assert _exists(root, "face_captures/old_assigned.jpg")          # source of an enrolled embedding
    assert _exists(root, "face_training/_unlabeled/lab.jpg")        # training set
    assert not _exists(root, "face_training/_unlabeled/rej.jpg")


def test_retention_is_idempotent(retention_env):
    retention, _, now = retention_env
    retention.run_once(now)
    second = retention.run_once(now)
    assert second["face_pending"] == 0 and second["training_captures"] == 0


def test_retention_can_be_switched_off(retention_env, monkeypatch):
    retention, root, now = retention_env
    monkeypatch.setattr(config, "RETENTION_ENABLED", False)
    result = retention.run_once(now)
    assert set(result) == {"sessions"}
    assert _exists(root, "face_captures/old_pending.jpg")


def test_retention_never_deletes_outside_data_dir(retention_env, tmp_path_factory):
    retention, _, now = retention_env
    outside = tmp_path_factory.mktemp("outside") / "keep.jpg"
    outside.write_bytes(b"x")
    with db.connect(config.DB_PATH) as c:
        c.execute("INSERT INTO face_pending (camera_id, track_id, captured_at, image_path, embedding, status) VALUES (1,1,?,?, '[]','pending')",
                  (now - 100 * 86400, str(outside)))
    retention.run_once(now)
    assert outside.exists()


def test_orphan_files_are_kept_unless_orphan_deletion_is_enabled(retention_env, monkeypatch):
    retention, root, now = retention_env
    stray = root / "face_captures" / "restored_by_hand.jpg"
    stray.write_bytes(b"x")
    os.utime(stray, (now - 3 * 86400, now - 3 * 86400))
    assert "orphan_files" not in retention.run_once(now)
    assert stray.exists()


def test_orphan_files_are_removed_but_referenced_and_new_files_kept(retention_env, monkeypatch):
    retention, root, now = retention_env
    monkeypatch.setattr(config, "RETENTION_DELETE_ORPHANS", True)
    orphan = root / "face_captures" / "orphan.jpg"
    fresh = root / "face_captures" / "fresh_orphan.jpg"
    orphan.write_bytes(b"x")
    fresh.write_bytes(b"x")
    os.utime(orphan, (now - 3 * 86400, now - 3 * 86400))
    for f in (root / "face_captures").glob("*.jpg"):
        if f.name != "fresh_orphan.jpg":
            os.utime(f, (now - 3 * 86400, now - 3 * 86400))
    retention.run_once(now)
    assert not orphan.exists() and fresh.exists()
    assert (root / "face_captures" / "new_pending.jpg").exists()


def test_expired_sessions_are_purged(isolated_db):
    from app import auth, retention

    with auth.get_connection() as c:
        c.execute("INSERT INTO sessions (token, principal_type, created_at, expires_at) VALUES ('t1','client',0,1)")
        c.execute("INSERT INTO sessions (token, principal_type, created_at, expires_at) VALUES ('t2','client',0,?)", (time.time() + 999,))
    assert retention.run_once()["sessions"] == 1


# --- path migration ------------------------------------------------------------------

def test_migration_dry_run_changes_nothing_and_apply_is_idempotent(tmp_path, monkeypatch):
    from scripts import migrate_paths

    data = tmp_path / "data"
    (data / "face_enroll").mkdir(parents=True)
    (data / "face_enroll" / "018_x.png").write_bytes(b"x")
    monkeypatch.setattr(config, "DATA_DIR", data)
    dbp = data / "app.db"
    c = sqlite3.connect(dbp)
    c.execute("CREATE TABLE face_embeddings (id INTEGER PRIMARY KEY, person_id TEXT, embedding TEXT, source_image_path TEXT, enrolled_at REAL)")
    legacy = r"C:\Users\lenovo\Desktop\DECO_VISION\backend\data\face_enroll\018_x.png"
    c.execute("INSERT INTO face_embeddings VALUES (1, '018', '[]', ?, 0)", (legacy,))
    c.commit()
    c.close()

    report = migrate_paths.migrate(dbp, apply=False)
    assert report["face_embeddings"]["rewritten"] == 1
    assert sqlite3.connect(dbp).execute("SELECT source_image_path FROM face_embeddings").fetchone()[0] == legacy

    migrate_paths.migrate(dbp, apply=True)
    assert sqlite3.connect(dbp).execute("SELECT source_image_path FROM face_embeddings").fetchone()[0] == "face_enroll/018_x.png"
    assert migrate_paths.migrate(dbp, apply=True)["face_embeddings"]["rewritten"] == 0


def test_migration_backup_is_a_complete_copy(tmp_path):
    from scripts import migrate_paths

    dbp = tmp_path / "app.db"
    c = sqlite3.connect(dbp)
    c.execute("CREATE TABLE t (x)")
    c.execute("INSERT INTO t VALUES (42)")
    c.commit()
    c.close()
    b = migrate_paths.backup(dbp)
    assert Path(b).exists() and sqlite3.connect(b).execute("SELECT x FROM t").fetchone()[0] == 42
