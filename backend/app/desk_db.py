"""
desk_db.py

Storage for desk analytics (workforce insights). Ported from the Deco-vision
project's desk_db.py, keyed by employee_id (this app's face recognition
identifies employees by ID, see employee_directory.py) instead of by name.
desk_tracker.py decides who is at which desk; this module only records it.

  - desk_zones: admin-drawn desk outlines, a polygon of [x, y] points in
    0..1 fractions of the frame (so they survive a resolution change), per
    camera. Not tied to an employee: who sits there is worked out live.
  - desk_sessions: one row per continuous stretch an employee was seen at a
    specific desk (start_ts fixed, end_ts moved forward on every sighting).
  - away_sessions: one row per stretch an employee was away from every desk.
  - desk_movement_events: append-only log of every change (session start /
    end, away start / end, desk switch) with the confidence behind it.
"""

import contextlib
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"


@contextlib.contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS desk_zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                zone_label TEXT NOT NULL,
                polygon TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS desk_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_id INTEGER NOT NULL,
                employee_id TEXT NOT NULL,
                camera_id INTEGER NOT NULL,
                start_ts REAL NOT NULL,
                end_ts REAL NOT NULL,
                last_confidence REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS away_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                start_ts REAL NOT NULL,
                end_ts REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS desk_movement_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                zone_id INTEGER,
                ts REAL NOT NULL,
                confidence REAL,
                details TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_desk_zones_camera ON desk_zones (camera_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_desk_sessions_start ON desk_sessions (start_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_away_sessions_start ON away_sessions (start_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_desk_events_ts ON desk_movement_events (ts)")


# ---- Zones ------------------------------------------------------------------

def _zone(row) -> dict:
    d = dict(row)
    d["polygon"] = json.loads(d["polygon"])
    return d


def list_zones(camera_id: int | None = None) -> list[dict]:
    query, params = "SELECT * FROM desk_zones", []
    if camera_id is not None:
        query += " WHERE camera_id = ?"
        params.append(camera_id)
    with get_connection() as conn:
        rows = conn.execute(query + " ORDER BY id", params).fetchall()
    return [_zone(r) for r in rows]


def get_zone(zone_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM desk_zones WHERE id = ?", (zone_id,)).fetchone()
    return _zone(row) if row else None


def create_zone(camera_id: int, polygon: list, label: str | None = None) -> int:
    """With no label, auto-names the desk "Desk N" using the lowest number not
    already taken on this camera, so deleting and re-adding desks never
    produces duplicate names."""
    with get_connection() as conn:
        if not label:
            used = set()
            for (existing,) in conn.execute("SELECT zone_label FROM desk_zones WHERE camera_id = ?", (camera_id,)):
                if existing and existing.startswith("Desk "):
                    try:
                        used.add(int(existing.removeprefix("Desk ")))
                    except ValueError:
                        pass
            n = 1
            while n in used:
                n += 1
            label = f"Desk {n}"
        cur = conn.execute(
            "INSERT INTO desk_zones (camera_id, zone_label, polygon, created_at) VALUES (?, ?, ?, ?)",
            (camera_id, label, json.dumps(polygon), time.time()),
        )
        return cur.lastrowid


def delete_zone(zone_id: int) -> None:
    # Sessions already recorded at this desk are kept: "someone sat at Desk 3
    # from 9 to 11" stays true even after the desk outline is removed.
    with get_connection() as conn:
        conn.execute("DELETE FROM desk_zones WHERE id = ?", (zone_id,))


# ---- Desk and away sessions -------------------------------------------------

def start_session(zone_id: int, employee_id: str, camera_id: int, ts: float, confidence: float | None) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO desk_sessions (zone_id, employee_id, camera_id, start_ts, end_ts, last_confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (zone_id, employee_id, camera_id, ts, ts, confidence),
        )
        return cur.lastrowid


def touch_session(session_id: int, ts: float, confidence: float | None = None) -> None:
    with get_connection() as conn:
        if confidence is not None:
            conn.execute("UPDATE desk_sessions SET end_ts = ?, last_confidence = ? WHERE id = ?", (ts, confidence, session_id))
        else:
            conn.execute("UPDATE desk_sessions SET end_ts = ? WHERE id = ?", (ts, session_id))


def start_away(employee_id: str, ts: float) -> int:
    with get_connection() as conn:
        cur = conn.execute("INSERT INTO away_sessions (employee_id, start_ts, end_ts) VALUES (?, ?, ?)", (employee_id, ts, ts))
        return cur.lastrowid


def touch_away(away_id: int, ts: float) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE away_sessions SET end_ts = ? WHERE id = ?", (ts, away_id))


def load_open_sessions(grace_seconds: float, now: float | None = None) -> list[dict]:
    cutoff = (now if now is not None else time.time()) - grace_seconds
    with get_connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM desk_sessions WHERE end_ts >= ?", (cutoff,))]


def load_open_away(grace_seconds: float, now: float | None = None) -> list[dict]:
    cutoff = (now if now is not None else time.time()) - grace_seconds
    with get_connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM away_sessions WHERE end_ts >= ?", (cutoff,))]


def log_event(employee_id: str, event_type: str, ts: float, zone_id: int | None = None,
              confidence: float | None = None, details: str | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO desk_movement_events (employee_id, event_type, zone_id, ts, confidence, details) VALUES (?, ?, ?, ?, ?, ?)",
            (employee_id, event_type, zone_id, ts, confidence, details),
        )


# ---- Report -----------------------------------------------------------------

def _day_bounds(date: str) -> tuple[float, float]:
    start = datetime.strptime(date, "%Y-%m-%d").timestamp()
    return start, start + 86400


def get_daily_report(date: str, open_away_until: dict[str, float] | None = None) -> list[dict]:
    """Per employee seen at a desk (or away) on `date`: desk_seconds, away_seconds,
    movements (desk switches), first/last session. `open_away_until` lets the
    caller count an away stretch that's still going (employee_id -> now);
    otherwise an open away period would read as 0 s until they come back."""
    start, end = _day_bounds(date)
    with get_connection() as conn:
        desk = conn.execute("SELECT * FROM desk_sessions WHERE start_ts >= ? AND start_ts < ?", (start, end)).fetchall()
        away = conn.execute("SELECT * FROM away_sessions WHERE start_ts >= ? AND start_ts < ?", (start, end)).fetchall()
        switches = conn.execute(
            "SELECT employee_id, COUNT(*) FROM desk_movement_events WHERE event_type = 'desk_switch' AND ts >= ? AND ts < ? GROUP BY employee_id",
            (start, end),
        ).fetchall()
    open_away_until = open_away_until or {}
    latest_away: dict[str, sqlite3.Row] = {}
    for a in away:
        if a["employee_id"] not in latest_away or a["start_ts"] > latest_away[a["employee_id"]]["start_ts"]:
            latest_away[a["employee_id"]] = a

    by_emp: dict[str, dict] = {}

    def bucket(emp):
        return by_emp.setdefault(emp, {"desk": 0.0, "away": 0.0, "first": None, "last": None})

    def span(b, s, e):
        b["first"] = s if b["first"] is None else min(b["first"], s)
        b["last"] = e if b["last"] is None else max(b["last"], e)

    for s in desk:
        b = bucket(s["employee_id"])
        b["desk"] += s["end_ts"] - s["start_ts"]
        span(b, s["start_ts"], s["end_ts"])
    for a in away:
        b = bucket(a["employee_id"])
        stop = a["end_ts"]
        if latest_away[a["employee_id"]]["id"] == a["id"] and a["employee_id"] in open_away_until:
            stop = max(stop, min(open_away_until[a["employee_id"]], end))
        b["away"] += stop - a["start_ts"]
        span(b, a["start_ts"], stop)
    moves = {emp: n for emp, n in switches}

    return sorted(
        (
            {"employee_id": emp, "desk_seconds": round(b["desk"]), "away_seconds": round(b["away"]),
             "movements": moves.get(emp, 0), "first_session": b["first"], "last_session": b["last"]}
            for emp, b in by_emp.items()
        ),
        key=lambda r: -r["desk_seconds"],
    )
