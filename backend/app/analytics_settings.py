"""
analytics_settings.py

On/off switches for each analytics feature, so CPU can be spent only on
what's needed. Checked on every frame by the code that does the work, so a
switch takes effect within a second or two, and saved in the database so it
survives a restart.

  face_recognition  face detection + recognition on every camera with it
                    enabled (camera_stream.py). Attendance, desk analytics
                    and "Unknown Person" alerts all depend on it.
  desk_analytics    naming people at marked desks (desks.py).
  footfall          unique footfall Re-ID on gate cameras (footfall.py).
  intrusion         person detection in restricted zones (intrusion.py).
  staff_count       entry/exit line counting at entrance cameras (staff/).
  alerts            raising new alerts (alerts.py); existing ones stay listed.
  expression        facial-expression reading on live faces (expression.py).
  appearance_handoff naming faceless people on the live overlay by body
                    appearance learned from today's face matches (appearance.py).
  live_overlay      the person boxes drawn on the AI Analytics view: an extra
                    person-detection pass per watched camera (face_pipeline.py).

Cameras are also only kept streaming in the background while an analytics
feature that needs them is on, so with everything off a camera is only
decoded while someone is watching it.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"

FEATURES = {
    "face_recognition": "Face recognition & attendance",
    "desk_analytics": "Desk analytics",
    "footfall": "Unique footfall",
    "live_overlay": "Live person boxes",
    "intrusion": "Intrusion detection",
    "staff_count": "Staff count",
    "alerts": "Alerts",
    "expression": "Expression detection",
    "appearance_handoff": "Name by appearance",
}
# Expression reads the face crops of the live person boxes, so it needs them on.
DEPENDS_ON = {"desk_analytics": "face_recognition", "expression": "live_overlay", "appearance_handoff": "face_recognition"}

_state = {k: True for k in FEATURES}
_lock = threading.Lock()


def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS analytics_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL)")
        for key, value in conn.execute("SELECT key, value FROM analytics_settings"):
            if key in _state:
                _state[key] = json.loads(value)


def enabled(feature: str) -> bool:
    """Fast in-memory check, safe to call per frame. A feature whose
    dependency is off counts as off."""
    if not _state.get(feature, True):
        return False
    dep = DEPENDS_ON.get(feature)
    return enabled(dep) if dep else True


def set_enabled(feature: str, on: bool) -> None:
    if feature not in FEATURES:
        raise KeyError(feature)
    with _lock:
        _state[feature] = bool(on)
        with _conn() as conn:
            conn.execute(
                "INSERT INTO analytics_settings (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (feature, json.dumps(bool(on)), time.time()),
            )


def snapshot() -> dict:
    return {
        k: {"label": label, "on": _state[k], "effective": enabled(k), "requires": DEPENDS_ON.get(k)}
        for k, label in FEATURES.items()
    }


_proc = None
_children: dict = {}


def cpu_usage() -> dict:
    """Backend CPU (this process plus its camera-reader processes) and whole
    machine, as % of all cores, sampled since the previous call."""
    global _proc
    try:
        import psutil
    except ImportError:
        return {}
    if _proc is None:
        _proc = psutil.Process()
    # psutil measures CPU between two calls on the SAME Process object, so the
    # camera-reader children are cached by pid (a new one reads 0 the first time).
    live = {p.pid: p for p in [_proc] + _proc.children(recursive=True)}
    for pid, p in live.items():
        _children.setdefault(pid, p)
    for pid in [pid for pid in _children if pid not in live]:
        del _children[pid]
    total = 0.0
    for p in _children.values():
        try:
            total += p.cpu_percent(None)
        except psutil.Error:
            pass
    cores = psutil.cpu_count() or 1
    return {
        "backend_pct": round(total / cores, 1),
        "system_pct": psutil.cpu_percent(None),
        "memory_free_gb": round(psutil.virtual_memory().available / 1e9, 1),
        "memory_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }
