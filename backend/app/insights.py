"""
insights.py

Summaries for the Dashboard and the Workforce Insights overview, built only
from data the backend already records: attendance (attendance.py), unique
footfall (footfall.py), desk analytics (desks.py), live camera streams
(camera_stream.py), unrecognised faces (face_db's face_pending queue) and
face-training labels. Nothing here is estimated or invented: a number with
no data behind it comes back as None and the page says so.
"""

import datetime
import sqlite3
import time
from pathlib import Path

from app import attendance, camera_db, camera_stream, employee_directory

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"

# A streaming camera that hasn't delivered a frame for this long is offline.
OFFLINE_AFTER_SECONDS = 30


def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _iso(d: datetime.date) -> str:
    return d.isoformat()


def camera_health(now: float | None = None) -> list[dict]:
    """Per camera: "online" (frame in the last 30 s), "offline" (streaming but
    silent), or "idle" (not streaming: nobody watching and no analytics on it)."""
    now = now if now is not None else time.time()
    out = []
    for c in camera_db.list_cameras():
        last = camera_stream.get_stream(c["id"]).last_frame_at
        if not camera_db.is_streamable(c):
            state = "disabled"
        elif last is None:
            state = "idle"
        elif now - last <= OFFLINE_AFTER_SECONDS:
            state = "online"
        else:
            state = "offline"
        out.append({"id": c["id"], "name": c["name"], "state": state, "last_frame_at": last,
                    "face_recognition": bool(c.get("attendance_tracking"))})
    return out


def unknown_faces_today(now: float | None = None) -> int:
    """Faces captured today that face recognition couldn't confidently name."""
    now = now if now is not None else time.time()
    start = datetime.datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    try:
        with _conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM face_pending WHERE captured_at >= ?", (start,)).fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def enrollment_split() -> dict:
    """Employees with at least one labelled face-training photo (recognisable)
    vs none (the model can't recognise them yet)."""
    try:
        with _conn() as conn:
            labelled = {r[0] for r in conn.execute(
                "SELECT DISTINCT employee_id FROM face_training_captures WHERE label_status = 'labeled' AND employee_id IS NOT NULL")}
    except sqlite3.OperationalError:
        labelled = set()
    enrolled = sum(1 for emp in employee_directory.EMPLOYEES if emp in labelled)
    return {"enrolled": enrolled, "not_enrolled": len(employee_directory.EMPLOYEES) - enrolled}


def workforce_overview(date: str | None = None, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    day = datetime.date.fromisoformat(date) if date else datetime.date.fromtimestamp(now)
    today = attendance.day_report(_iso(day), now=now)
    yesterday = attendance.day_report(_iso(day - datetime.timedelta(days=1)), now=now)
    s = today["stats"]
    exited = sum(1 for r in today["rows"] if r["status"] == "Present")  # seen today, not in the last 15 min
    trend = []
    for i in range(6, -1, -1):
        d = day - datetime.timedelta(days=i)
        rep = today if i == 0 else attendance.day_report(_iso(d), now=now)
        trend.append({"date": _iso(d), "name": d.strftime("%a"), "value": rep["stats"]["present"]})
    return {
        "date": _iso(day),
        "total_employees": s["total"],
        "present": s["present"],
        "on_site": sum(1 for r in today["rows"] if r["status"] == "On site"),
        "exited": exited,
        "not_detected": s["absent"],
        "on_leave": s["on_leave"],
        "late": s["late"],
        "attendance_pct": s["attendance_pct"],
        "attendance_pct_yesterday": yesterday["stats"]["attendance_pct"],
        "present_yesterday": yesterday["stats"]["present"],
        "trend": trend,
        "enrollment": enrollment_split(),
    }


def dashboard_summary(now: float | None = None) -> dict:
    from app import desks, footfall

    now = now if now is not None else time.time()
    day = datetime.date.fromtimestamp(now)
    att = attendance.day_report(_iso(day), now=now)
    att_y = attendance.day_report(_iso(day - datetime.timedelta(days=1)), now=now)
    cams = camera_health(now)
    ff = footfall.service.summary()
    desk_rows = desks.service.report(_iso(day))["employees"] if desks.service.tracker else []

    attention = []
    for c in cams:
        if c["state"] == "offline":
            ago = int((now - c["last_frame_at"]) // 60)
            attention.append({"label": f"{c['name']} is offline", "detail": f"No video for {ago} min" if ago else "No video for under a minute", "tag": "Critical"})
    for g in ff["gates"]:
        if not g["counting"]:
            attention.append({"label": f"Footfall stopped at {g['name']}", "detail": "Counting failed on this gate; check the backend log", "tag": "Critical"})
        elif not g.get("zone"):
            attention.append({"label": f"No counting zone at {g['name']}", "detail": "Footfall counts the whole view, including seating", "tag": "Warning"})
    late = [r for r in att["rows"] if r["arrival"] == "Late arrival"]
    if late:
        names = ", ".join(r["name"] for r in late[:3]) + (f" and {len(late) - 3} more" if len(late) > 3 else "")
        attention.append({"label": f"{len(late)} late arrival{'s' if len(late) != 1 else ''}", "detail": names, "tag": "Warning"})

    insights = []
    s = att["stats"]
    if s["present"]:
        insights.append(f"{s['present']} of {s['total']} employees ({s['attendance_pct']}%) have been seen today; "
                        f"yesterday it was {att_y['stats']['present']}.")
    if ff["busiest_hour"] is not None and ff["unique_today"]:
        h = ff["busiest_hour"]
        insights.append(f"Footfall is busiest at {datetime.time(h).strftime('%I %p').lstrip('0')} so far, "
                        f"with {ff['unique_today']} unique visitors across {len(ff['gates'])} gate{'s' if len(ff['gates']) != 1 else ''}.")
    if desk_rows:
        top = desk_rows[0]
        insights.append(f"{top['name']} has the most desk time today: {top['desk_seconds'] // 3600}h {(top['desk_seconds'] % 3600) // 60:02d}m.")
    unknown = unknown_faces_today(now)
    if unknown:
        insights.append(f"{unknown} face{'s' if unknown != 1 else ''} today couldn't be matched to an employee; "
                        "label them on the Face Training page to improve recognition.")
    if not insights:
        insights.append("No activity recorded yet today.")

    return {
        "people_present": {"value": s["present"], "of": s["total"], "yesterday": att_y["stats"]["present"]},
        "unknown_faces": unknown,
        "cameras": {"online": sum(c["state"] == "online" for c in cams), "total": len(cams),
                    "disabled": sum(c["state"] == "disabled" for c in cams),
                    "offline": sum(c["state"] == "offline" for c in cams), "list": cams},
        "needs_attention": attention,
        "insights": insights,
    }
