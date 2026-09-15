"""Camera + site registry, SQLite-backed. Mirrors the shape the frontend's
api.js already expects (listCameras/createCamera/... and listSites/...)."""

import contextlib
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"


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
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                site TEXT NOT NULL DEFAULT '',
                cam_code TEXT DEFAULT '',
                purpose TEXT DEFAULT 'GENERAL',
                host TEXT DEFAULT '',
                port INTEGER DEFAULT 554,
                user TEXT DEFAULT '',
                password TEXT DEFAULT '',
                stream_path TEXT DEFAULT '',
                vendor TEXT DEFAULT '',
                status TEXT DEFAULT 'inactive',
                live_feed_enabled INTEGER DEFAULT 1
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT DEFAULT ''
            )
            """
        )
        _seed_sites_if_empty(conn)


def _seed_sites_if_empty(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) FROM sites").fetchone()[0]
    if count > 0:
        return
    conn.execute("INSERT OR IGNORE INTO sites (name, description) VALUES (?, '')", ("Noida Site",))


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["is_configured"] = bool(d["host"])
    d["live"] = bool(d["live_feed_enabled"]) and d["status"] == "active" and bool(d["host"])
    d.pop("password", None)  # never sent to the frontend
    return d


def list_cameras() -> list[dict]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM cameras ORDER BY id").fetchall()
    return [_row_to_dict(r) for r in rows]


def get_camera(camera_id: int) -> dict | None:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
    return _row_to_dict(row) if row else None


def get_camera_connection(camera_id: int) -> dict | None:
    """Includes the password — internal use only (building the RTSP URL)."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
    return dict(row) if row else None


def add_camera(name: str, site: str, **fields) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO cameras
               (name, site, cam_code, purpose, host, port, user, password, stream_path, vendor,
                status, live_feed_enabled)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                site,
                fields.get("cam_code", ""),
                fields.get("purpose", "GENERAL"),
                fields.get("host", ""),
                fields.get("port", 554),
                fields.get("user", ""),
                fields.get("password", ""),
                fields.get("stream_path", "/h264/ch1/sub/av_stream"),
                fields.get("vendor", ""),
                "active" if fields.get("host") else "inactive",
                int(fields.get("live_feed_enabled", True)),
            ),
        )
        return cur.lastrowid


def update_camera(camera_id: int, **fields) -> None:
    if not fields:
        return
    if "host" in fields:
        fields["status"] = "active" if fields["host"] else "inactive"
    columns = ", ".join(f"{k} = ?" for k in fields)
    with get_connection() as conn:
        conn.execute(f"UPDATE cameras SET {columns} WHERE id = ?", (*fields.values(), camera_id))


def delete_camera(camera_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))


def list_sites() -> list[dict]:
    cameras = list_cameras()
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM sites ORDER BY id").fetchall()
    result = []
    for r in rows:
        site = dict(r)
        site_cameras = [c for c in cameras if c["site"] == site["name"]]
        site["cameras"] = site_cameras
        site["active_count"] = sum(1 for c in site_cameras if c["status"] == "active")
        result.append(site)
    return result


def add_site(name: str, description: str = "") -> int:
    with get_connection() as conn:
        cur = conn.execute("INSERT INTO sites (name, description) VALUES (?, ?)", (name, description))
        return cur.lastrowid


def update_site(site_id: int, name: str | None = None, description: str | None = None) -> None:
    with get_connection() as conn:
        if name is not None:
            old_name = conn.execute("SELECT name FROM sites WHERE id = ?", (site_id,)).fetchone()
            if old_name and old_name[0] != name:
                conn.execute("UPDATE cameras SET site = ? WHERE site = ?", (name, old_name[0]))
            conn.execute("UPDATE sites SET name = ? WHERE id = ?", (name, site_id))
        if description is not None:
            conn.execute("UPDATE sites SET description = ? WHERE id = ?", (description, site_id))


def delete_site(site_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM sites WHERE id = ?", (site_id,))
