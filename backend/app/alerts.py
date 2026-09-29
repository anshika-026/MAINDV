"""
alerts.py

Real alerts for the Alerts & Events page, raised by what the system actually
detects (nothing here is simulated):

  - Unknown Person (High): a face at an Entry/Exit camera that face
    recognition can't match to any employee (best classifier probability
    under UNKNOWN_MAX_PROBA, so poorly-captured employees in the 0.4-0.6
    "not sure" band don't raise one). A snapshot of the face is kept. At
    most one per camera every UNKNOWN_COOLDOWN_SECONDS so a crowd doesn't
    flood the list.
  - Camera Offline (Critical): a streaming camera sent no video for 30 s.
    Resolves itself when video comes back.
  - Footfall Stopped (Critical): counting failed on a gate camera.
  - Late Arrival (Low): an employee's first sighting today was after
    attendance.ATTENDANCE_LATE_AFTER. One per employee per day.

An alert with the same `dedupe_key` that's still open is updated (count and
last_seen) instead of creating a duplicate row.
"""

import datetime
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from app import lifecycle

log = logging.getLogger("alerts")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "alert_snapshots"

UNKNOWN_MAX_PROBA = float(os.environ.get("ALERT_UNKNOWN_MAX_PROBA", "0.4"))
UNKNOWN_COOLDOWN_SECONDS = float(os.environ.get("ALERT_UNKNOWN_COOLDOWN_SECONDS", "300"))
MONITOR_INTERVAL_SECONDS = 30

SEVERITIES = ("Critical", "High", "Medium", "Low")
_lock = threading.Lock()


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,              -- first raised
                last_seen REAL NOT NULL,       -- last re-raise while open
                occurrences INTEGER NOT NULL DEFAULT 1,
                event TEXT NOT NULL,
                severity TEXT NOT NULL,
                camera_id INTEGER,
                message TEXT,
                confidence REAL,
                employee_id TEXT,
                snapshot_path TEXT,
                dedupe_key TEXT,
                status TEXT NOT NULL DEFAULT 'Active',   -- Active | Acknowledged | Resolved
                acknowledged_by TEXT,
                acknowledged_at REAL,
                resolved_by TEXT,
                resolved_at REAL,
                resolution_reason TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts (ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_open_key ON alerts (dedupe_key, status)")


def raise_alert(event: str, severity: str, camera_id: int | None = None, message: str = "",
                dedupe_key: str | None = None, confidence: float | None = None,
                employee_id: str | None = None, snapshot_jpeg: bytes | None = None,
                now: float | None = None) -> int:
    from app import analytics_settings

    if not analytics_settings.enabled("alerts"):
        return 0  # Alerts switched off: nothing new is raised (existing alerts stay listed)
    now = now if now is not None else time.time()
    with _lock, _conn() as conn:
        if dedupe_key:
            row = conn.execute(
                "SELECT id FROM alerts WHERE dedupe_key = ? AND status != 'Resolved' ORDER BY id DESC LIMIT 1", (dedupe_key,)
            ).fetchone()
            if row:
                conn.execute("UPDATE alerts SET last_seen = ?, occurrences = occurrences + 1 WHERE id = ?", (now, row["id"]))
                return row["id"]
        cur = conn.execute(
            "INSERT INTO alerts (ts, last_seen, event, severity, camera_id, message, confidence, employee_id, dedupe_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now, now, event, severity, camera_id, message, confidence, employee_id, dedupe_key),
        )
        alert_id = cur.lastrowid
        if snapshot_jpeg:
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            path = SNAPSHOT_DIR / f"alert_{alert_id}.jpg"
            path.write_bytes(snapshot_jpeg)
            conn.execute("UPDATE alerts SET snapshot_path = ? WHERE id = ?", (str(path), alert_id))
    log.info("alert %s: %s (%s) camera %s %s", alert_id, event, severity, camera_id, message)
    return alert_id


def resolve_open(dedupe_key: str, reason: str, by: str = "System", now: float | None = None) -> None:
    now = now if now is not None else time.time()
    with _lock, _conn() as conn:
        conn.execute(
            "UPDATE alerts SET status = 'Resolved', resolved_by = ?, resolved_at = ?, resolution_reason = ? "
            "WHERE dedupe_key = ? AND status != 'Resolved'",
            (by, now, reason, dedupe_key),
        )


def acknowledge(alert_id: int, by: str) -> bool:
    with _lock, _conn() as conn:
        cur = conn.execute(
            "UPDATE alerts SET status = 'Acknowledged', acknowledged_by = ?, acknowledged_at = ? WHERE id = ? AND status = 'Active'",
            (by, time.time(), alert_id),
        )
        return cur.rowcount > 0


def resolve(alert_id: int, by: str, reason: str) -> bool:
    with _lock, _conn() as conn:
        cur = conn.execute(
            "UPDATE alerts SET status = 'Resolved', resolved_by = ?, resolved_at = ?, resolution_reason = ? WHERE id = ? AND status != 'Resolved'",
            (by, time.time(), reason, alert_id),
        )
        return cur.rowcount > 0


def _range_start(range_: str, now: float) -> tuple[float | None, float | None]:
    today = datetime.datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
    if range_ == "today":
        return today.timestamp(), None
    if range_ == "yesterday":
        return (today - datetime.timedelta(days=1)).timestamp(), today.timestamp()
    if range_ == "week":
        return (today - datetime.timedelta(days=6)).timestamp(), None
    return None, None


def list_alerts(range_: str = "today", limit: int = 500, now: float | None = None) -> list[dict]:
    from app import camera_db

    now = now if now is not None else time.time()
    start, end = _range_start(range_, now)
    q, params = "SELECT * FROM alerts WHERE 1=1", []
    if start is not None:
        q += " AND ts >= ?"
        params.append(start)
    if end is not None:
        q += " AND ts < ?"
        params.append(end)
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(q + " ORDER BY ts DESC LIMIT ?", (*params, limit))]
    cams = {c["id"]: c for c in camera_db.list_cameras()}
    for r in rows:
        cam = cams.get(r["camera_id"]) or {}
        r["camera_name"] = cam.get("name") or (f"Camera {r['camera_id']}" if r["camera_id"] else "-")
        r["location"] = cam.get("site") or "-"
        r["has_snapshot"] = bool(r.pop("snapshot_path", None))
    return rows


def summary(now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    today, _ = _range_start("today", now)
    yesterday, _ = _range_start("yesterday", now)
    with _conn() as conn:
        one = lambda q, *p: conn.execute(q, p).fetchone()[0]  # noqa: E731
        return {
            "active": one("SELECT COUNT(*) FROM alerts WHERE status = 'Active'"),
            "acknowledged": one("SELECT COUNT(*) FROM alerts WHERE status = 'Acknowledged'"),
            "resolved_today": one("SELECT COUNT(*) FROM alerts WHERE status = 'Resolved' AND resolved_at >= ?", today),
            "resolved_yesterday": one("SELECT COUNT(*) FROM alerts WHERE status = 'Resolved' AND resolved_at >= ? AND resolved_at < ?", yesterday, today),
            "raised_last_hour": one("SELECT COUNT(*) FROM alerts WHERE last_seen >= ?", now - 3600),
            "raised_today": one("SELECT COUNT(*) FROM alerts WHERE ts >= ?", today),
        }


def snapshot_path(alert_id: int) -> Path | None:
    with _conn() as conn:
        row = conn.execute("SELECT snapshot_path FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    if not row or not row[0]:
        return None
    p = Path(row[0])
    return p if p.is_file() else None


# ---- hooks called from the rest of the backend -----------------------------

_last_unknown: dict[int, float] = {}


def unknown_face(camera_id: int, best_score: float, crop_jpeg: bytes | None) -> None:
    """face_pipeline: a face track ended without a confident match."""
    from app import camera_db

    if best_score >= UNKNOWN_MAX_PROBA:
        return  # probably an employee captured badly, not a stranger
    cam = camera_db.get_camera(camera_id) or {}
    if "entry" not in (cam.get("purpose") or "").lower():
        return  # only at gates: unrecognised faces at desks are mostly staff seen at odd angles
    now = time.time()
    if now - _last_unknown.get(camera_id, 0) < UNKNOWN_COOLDOWN_SECONDS:
        return
    _last_unknown[camera_id] = now
    raise_alert("Unknown Person", "High", camera_id,
                message=f"A face at {cam.get('name', 'the gate')} didn't match any employee",
                confidence=best_score, snapshot_jpeg=crop_jpeg, now=now)


def late_arrival(employee_id: str, name: str, camera_id: int, first_seen: float) -> None:
    """attendance: first sighting of the day came after the late cutoff."""
    day = datetime.date.fromtimestamp(first_seen).isoformat()
    at = datetime.datetime.fromtimestamp(first_seen).strftime("%I:%M %p")
    raise_alert("Late Arrival", "Low", camera_id, message=f"{name} first seen at {at}",
                dedupe_key=f"late:{employee_id}:{day}", employee_id=employee_id, now=first_seen)


def _monitor_loop() -> None:
    from app import footfall, insights

    while True:
        try:
            for cam in insights.camera_health():
                key = f"offline:{cam['id']}"
                if cam["state"] == "offline":
                    raise_alert("Camera Offline", "Critical", cam["id"], message=f"No video from {cam['name']} for over 30 seconds", dedupe_key=key)
                elif cam["state"] == "online":
                    resolve_open(key, "Video came back")
            for g in footfall.service.summary()["gates"]:
                key = f"footfall:{g['camera_id']}"
                if not g["counting"]:
                    raise_alert("Footfall Stopped", "Critical", g["camera_id"], message=f"Unique footfall stopped counting at {g['name']}", dedupe_key=key)
                else:
                    resolve_open(key, "Counting again")
        except Exception:
            log.exception("alerts: monitor pass failed")
        if lifecycle.wait(MONITOR_INTERVAL_SECONDS):
            return
_started = False


def start() -> None:
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_monitor_loop, daemon=True, name="alerts-monitor").start()
