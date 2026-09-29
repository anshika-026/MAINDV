"""
occupancy.py

StaffOccupancyManager: the single source of truth for who is inside the office.
Every entrance camera reports line crossings here; the count is derived from
this state (visits marked PRESENT), never from how many people a camera sees.

A *visit* is one stay inside, from ENTRY to EXIT, with its own id. A visit
belongs to an employee when face recognition identified them, or is anonymous
(an unknown person, or anyone while face recognition is off). One employee can
have at most one PRESENT visit, across all cameras, so the same person entering
by two doors, or a repeated crossing, never counts twice.

Exits: people leaving usually have their back to an entrance camera, so an exit
without a face is matched on body appearance (the Re-ID embedding stored at
entry) against ONLY the people currently inside, which is a small, easy set.

Every decision, including ignored ones (duplicate entry, exit with nobody to
match), is written to staff_events for auditing.
"""

import datetime
import logging
import sqlite3
import threading
import time

import numpy as np

from app.staff import config as staff_config
from app import config, db

log = logging.getLogger("staff")

DB_PATH = config.DB_PATH

PRESENT, EXITED = "PRESENT", "EXITED"


def _conn():
    """Shared connection policy (app/db.py): lock timeout + busy_timeout,
    commit on success, rollback on error, and always closed."""
    return db.connect(DB_PATH, row_factory=sqlite3.Row)


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS staff_camera_config (
                camera_id INTEGER PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1,
                line TEXT,              -- [[x1, y1], [x2, y2]] fractions of the frame
                inside_sign INTEGER NOT NULL DEFAULT 1,  -- which side of the line is the office
                roi TEXT,               -- polygon [[x, y], ...] or NULL = whole frame
                updated_at REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS staff_occupancy (
                visit_id TEXT PRIMARY KEY,
                employee_id TEXT,       -- NULL = anonymous / unknown person
                status TEXT NOT NULL,   -- PRESENT | EXITED
                entry_time REAL NOT NULL,
                exit_time REAL,
                last_seen REAL NOT NULL,
                camera_id INTEGER,
                track_id INTEGER,
                confidence REAL,
                identity_source TEXT,   -- face | none
                embedding BLOB          -- body appearance at entry, for matching the exit
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS staff_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                event_type TEXT NOT NULL,   -- ENTRY | EXIT | AUTO_EXIT | IDENTIFIED | DUPLICATE_ENTRY | EXIT_UNMATCHED
                visit_id TEXT,
                employee_id TEXT,
                camera_id INTEGER,
                track_id INTEGER,
                confidence REAL,
                identity_source TEXT,   -- face | reid | fallback | none
                details TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_staff_events_ts ON staff_events (ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_staff_occupancy_status ON staff_occupancy (status)")


def _day_start(ts: float) -> float:
    return datetime.datetime.fromtimestamp(ts).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _norm(v):
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else None


class StaffOccupancyManager:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or staff_config.SETTINGS
        self._lock = threading.RLock()
        self._present: dict[str, dict] = {}  # visit_id -> row (+ "emb": np.ndarray | None)
        self._seq = 0
        self._last_auto_exit_day: str | None = None
        self._load()

    # ---- persistence ----------------------------------------------------------

    def _load(self) -> None:
        with _conn() as conn:
            rows = conn.execute("SELECT * FROM staff_occupancy WHERE status = ?", (PRESENT,)).fetchall()
            self._seq = conn.execute("SELECT COUNT(*) FROM staff_occupancy").fetchone()[0]
        for r in rows:
            d = dict(r)
            d["emb"] = _norm(np.frombuffer(d.pop("embedding"), dtype=np.float32)) if d.get("embedding") else None
            self._present[d["visit_id"]] = d
        if rows:
            log.info("staff: restored %d people still inside from before the restart", len(rows))

    def _save_visit(self, v: dict) -> None:
        emb = v.get("emb")
        with _conn() as conn:
            conn.execute(
                "INSERT INTO staff_occupancy (visit_id, employee_id, status, entry_time, exit_time, last_seen, camera_id, track_id, confidence, identity_source, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(visit_id) DO UPDATE SET employee_id=excluded.employee_id, "
                "status=excluded.status, exit_time=excluded.exit_time, last_seen=excluded.last_seen, confidence=excluded.confidence, "
                "identity_source=excluded.identity_source, embedding=COALESCE(excluded.embedding, staff_occupancy.embedding)",
                (v["visit_id"], v.get("employee_id"), v["status"], v["entry_time"], v.get("exit_time"), v["last_seen"],
                 v.get("camera_id"), v.get("track_id"), v.get("confidence"), v.get("identity_source"),
                 emb.astype(np.float32).tobytes() if emb is not None else None),
            )

    def _event(self, ts, event_type, visit=None, camera_id=None, track_id=None, confidence=None,
               source=None, details=None, employee_id=None) -> dict:
        ev = {
            "ts": ts, "event_type": event_type,
            "visit_id": visit["visit_id"] if visit else None,
            "employee_id": employee_id if employee_id is not None else (visit or {}).get("employee_id"),
            "camera_id": camera_id, "track_id": track_id, "confidence": confidence,
            "identity_source": source, "details": details,
        }
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO staff_events (ts, event_type, visit_id, employee_id, camera_id, track_id, confidence, identity_source, details) "
                "VALUES (:ts, :event_type, :visit_id, :employee_id, :camera_id, :track_id, :confidence, :identity_source, :details)", ev)
            ev["id"] = cur.lastrowid
        log.info("staff: %s %s camera %s track %s (%s)", event_type, ev["employee_id"] or "unknown", camera_id, track_id, details or source or "")
        return ev

    # ---- queries ----------------------------------------------------------------

    def present_visit_of(self, employee_id: str) -> dict | None:
        return next((v for v in self._present.values() if v.get("employee_id") == employee_id), None)

    def _match_body(self, emb, candidates: list[dict]) -> dict | None:
        """Best body-appearance match among `candidates`, if clearly the best."""
        emb = _norm(emb)
        scored = sorted(((float(emb @ v["emb"]), v) for v in candidates if v.get("emb") is not None and emb is not None),
                        key=lambda x: -x[0])
        if not scored or scored[0][0] < self.cfg["reid_match_threshold"]:
            return None
        if len(scored) > 1 and scored[0][0] - scored[1][0] < self.cfg["reid_min_margin"]:
            return None
        return scored[0][1]

    # ---- state changes ----------------------------------------------------------

    def entry(self, camera_id, track_id, employee_id=None, confidence=None, embedding=None, ts=None) -> dict:
        ts = ts if ts is not None else time.time()
        with self._lock:
            if employee_id:
                existing = self.present_visit_of(employee_id)
                if existing:
                    existing["last_seen"] = ts
                    self._save_visit(existing)
                    return self._event(ts, "DUPLICATE_ENTRY", existing, camera_id, track_id, confidence, "face",
                                       "already inside; not counted again")
            self._seq += 1
            visit = {
                "visit_id": f"V{int(ts)}-{self._seq}", "employee_id": employee_id, "status": PRESENT,
                "entry_time": ts, "exit_time": None, "last_seen": ts, "camera_id": camera_id, "track_id": track_id,
                "confidence": confidence, "identity_source": "face" if employee_id else "none", "emb": _norm(embedding),
            }
            self._present[visit["visit_id"]] = visit
            self._save_visit(visit)
            return self._event(ts, "ENTRY", visit, camera_id, track_id, confidence, visit["identity_source"])

    def identify_late(self, visit_id, employee_id, confidence, ts=None) -> dict | None:
        """A face confirmed shortly after an anonymous ENTRY on the same track."""
        ts = ts if ts is not None else time.time()
        with self._lock:
            visit = self._present.get(visit_id)
            if not visit or visit.get("employee_id"):
                return None
            other = self.present_visit_of(employee_id)
            if other:
                # They were already inside (e.g. entry by another door): merge.
                visit.update(status=EXITED, exit_time=ts, last_seen=ts)
                del self._present[visit_id]
                self._save_visit(visit)
                return self._event(ts, "DUPLICATE_ENTRY", visit, visit["camera_id"], visit["track_id"], confidence, "face",
                                   "identified as someone already inside; merged", employee_id=employee_id)
            visit.update(employee_id=employee_id, confidence=confidence, identity_source="face", last_seen=ts)
            self._save_visit(visit)
            return self._event(ts, "IDENTIFIED", visit, visit["camera_id"], visit["track_id"], confidence, "face",
                               "identified after entering")

    def exit(self, camera_id, track_id, employee_id=None, confidence=None, embedding=None, ts=None,
             visit_id=None) -> dict:
        ts = ts if ts is not None else time.time()
        with self._lock:
            visit, source = None, None
            if visit_id and visit_id in self._present:
                visit, source = self._present[visit_id], "track"      # same track that entered
            elif employee_id:
                visit, source = self.present_visit_of(employee_id), "face"
                if visit is None:
                    return self._event(ts, "EXIT_UNMATCHED", None, camera_id, track_id, confidence, "face",
                                       "recognised leaving but wasn't marked inside", employee_id=employee_id)
            elif embedding is not None:
                visit = self._match_body(embedding, list(self._present.values()))
                source = "reid" if visit else None
            if visit is None:
                unknowns = sorted((v for v in self._present.values() if not v.get("employee_id")), key=lambda v: v["entry_time"])
                if unknowns:
                    visit, source = unknowns[0], "fallback"
            if visit is None:
                return self._event(ts, "EXIT_UNMATCHED", None, camera_id, track_id, confidence, "none",
                                   "nobody inside matched this person leaving; count unchanged")
            visit.update(status=EXITED, exit_time=ts, last_seen=ts)
            del self._present[visit["visit_id"]]
            self._save_visit(visit)
            return self._event(ts, "EXIT", visit, camera_id, track_id, confidence, source)

    def auto_exit_all(self, ts=None, reason="end of day") -> int:
        ts = ts if ts is not None else time.time()
        with self._lock:
            visits = list(self._present.values())
            for v in visits:
                v.update(status=EXITED, exit_time=ts, last_seen=ts)
                self._save_visit(v)
                self._event(ts, "AUTO_EXIT", v, v.get("camera_id"), v.get("track_id"), None, None, reason)
            self._present.clear()
            return len(visits)

    def maybe_end_of_day(self, now=None) -> int:
        now = now if now is not None else time.time()
        dt = datetime.datetime.fromtimestamp(now)
        day = dt.date().isoformat()
        if dt.strftime("%H:%M") >= self.cfg["end_of_day"] and self._last_auto_exit_day != day:
            self._last_auto_exit_day = day
            return self.auto_exit_all(now)
        return 0

    # ---- reporting --------------------------------------------------------------

    def counts(self, anonymous_mode: bool = False, now=None, camera_ids: set[int] | None = None) -> dict:
        """camera_ids: restrict to visits/events seen on these cameras (a
        client's licensed cameras); None = everything (admin)."""
        now = now if now is not None else time.time()
        start = _day_start(now)
        with self._lock:
            present = [v for v in self._present.values() if camera_ids is None or v.get("camera_id") in camera_ids]
            known = sum(1 for v in present if v.get("employee_id"))
            unknown = len(present) - known
        q, p = "SELECT event_type, COUNT(*) FROM staff_events WHERE ts >= ?", [start]
        if camera_ids is not None:
            q, p = q + f" AND camera_id IN ({','.join('?' * len(camera_ids)) or 'NULL'})", p + sorted(camera_ids)
        with _conn() as conn:
            by_type = dict(conn.execute(q + " GROUP BY event_type", p).fetchall())
        return {
            # Face recognition off: nobody can be told apart from a visitor, so
            # everyone inside is counted as staff (but never named).
            "current_staff_count": known + unknown if anonymous_mode else known,
            "known_staff": known,
            "unknown_persons": unknown,
            "total_persons": known + unknown,
            "total_entries_today": by_type.get("ENTRY", 0),
            "total_exits_today": by_type.get("EXIT", 0) + by_type.get("AUTO_EXIT", 0),
            "duplicates_prevented_today": by_type.get("DUPLICATE_ENTRY", 0),
            "unmatched_exits_today": by_type.get("EXIT_UNMATCHED", 0),
            "mode": "anonymous" if anonymous_mode else "identified",
            "timestamp": datetime.datetime.fromtimestamp(now).isoformat(timespec="seconds"),
        }

    def employees_today(self, now=None) -> list[dict]:
        """Every employee with a visit today: PRESENT or EXITED, latest visit."""
        now = now if now is not None else time.time()
        with _conn() as conn:
            rows = conn.execute(
                "SELECT * FROM staff_occupancy WHERE employee_id IS NOT NULL AND (entry_time >= ? OR status = ?) ORDER BY entry_time",
                (_day_start(now), PRESENT)).fetchall()
        latest: dict[str, dict] = {}
        for r in rows:
            latest[r["employee_id"]] = {k: r[k] for k in ("employee_id", "status", "entry_time", "exit_time", "last_seen", "camera_id", "confidence")}
        return sorted(latest.values(), key=lambda r: (r["status"] != PRESENT, -r["last_seen"]))

    def events(self, event_type: str | None = None, limit: int = 200, since: float | None = None,
               camera_ids: set[int] | None = None) -> list[dict]:
        q, p = "SELECT * FROM staff_events WHERE 1=1", []
        if event_type:
            q, p = q + " AND event_type = ?", p + [event_type]
        if since is not None:
            q, p = q + " AND ts >= ?", p + [since]
        if camera_ids is not None:
            q, p = q + f" AND camera_id IN ({','.join('?' * len(camera_ids)) or 'NULL'})", p + sorted(camera_ids)
        with _conn() as conn:
            return [dict(r) for r in conn.execute(q + " ORDER BY ts DESC LIMIT ?", (*p, limit))]
