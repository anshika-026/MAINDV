"""
intrusion.py

Restricted-area (intrusion) detection. An admin draws a zone (a polygon on
a camera's picture, fractions of the frame) and optionally an active window
("19:00"-"08:00" = overnight only). About once a second, each camera with an
enabled zone gets tracked people from the shared per-camera detector
(person_detection.py); a person whose FEET — the
bottom-centre of their box — are inside an active zone is an intrusion. The
feet, not the box centre, so someone walking past in front of the area
doesn't count as being in it.

Each intrusion raises an "Intrusion Detected" alert (alerts.py, High) with an
annotated snapshot. While that alert is still open, further detections in
the same zone count against it (occurrences / last seen) instead of creating
new ones. Detections are also logged per zone (at most once per
EVENT_LOG_SECONDS) for the Intrusion page's charts.

Switched on and off with the "intrusion" analytics switch; cameras with
zones are kept streaming only while it's on.
"""

import concurrent.futures
import datetime
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from app import alerts, analytics_settings, camera_db
from app.desk_tracker import in_polygon

log = logging.getLogger("intrusion")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"
CYCLE_SECONDS = 1.0
PERSON_CONF = 0.45
EVENT_LOG_SECONDS = 30
SYNC_INTERVAL_SECONDS = 15


def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intrusion_zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                polygon TEXT NOT NULL,
                active_from TEXT,      -- "HH:MM" or NULL = always
                active_to TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intrusion_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_id INTEGER NOT NULL,
                camera_id INTEGER NOT NULL,
                ts REAL NOT NULL,
                people INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_intrusion_events_ts ON intrusion_events (ts)")


# ---- zones ------------------------------------------------------------------

def _zone(row) -> dict:
    d = dict(row)
    d["polygon"] = json.loads(d["polygon"])
    d["enabled"] = bool(d["enabled"])
    return d


def list_zones(camera_id: int | None = None) -> list[dict]:
    q, p = "SELECT * FROM intrusion_zones", []
    if camera_id is not None:
        q, p = q + " WHERE camera_id = ?", [camera_id]
    with _conn() as conn:
        return [_zone(r) for r in conn.execute(q + " ORDER BY id", p)]


def create_zone(camera_id: int, name: str, polygon: list, active_from: str | None, active_to: str | None) -> dict:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO intrusion_zones (camera_id, name, polygon, active_from, active_to, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, name, json.dumps(polygon), active_from, active_to, time.time()),
        )
        row = conn.execute("SELECT * FROM intrusion_zones WHERE id = ?", (cur.lastrowid,)).fetchone()
    service.refresh()
    return _zone(row)


def update_zone(zone_id: int, **fields) -> dict | None:
    allowed = {k: v for k, v in fields.items() if k in ("name", "active_from", "active_to", "enabled")}
    if allowed:
        if "enabled" in allowed:
            allowed["enabled"] = int(bool(allowed["enabled"]))
        with _conn() as conn:
            conn.execute(f"UPDATE intrusion_zones SET {', '.join(f'{k} = ?' for k in allowed)} WHERE id = ?", (*allowed.values(), zone_id))
    service.refresh()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM intrusion_zones WHERE id = ?", (zone_id,)).fetchone()
    return _zone(row) if row else None


def delete_zone(zone_id: int) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM intrusion_zones WHERE id = ?", (zone_id,))
    service.refresh()


def is_active(zone: dict, now: float | None = None) -> bool:
    """Inside the zone's active window (which may wrap past midnight)."""
    if not zone["enabled"]:
        return False
    start, end = zone.get("active_from"), zone.get("active_to")
    if not start or not end:
        return True
    t = datetime.datetime.fromtimestamp(now if now is not None else time.time()).strftime("%H:%M")
    return start <= t < end if start <= end else (t >= start or t < end)


def stats(days: int = 7, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    today = datetime.date.fromtimestamp(now)
    start = datetime.datetime.combine(today - datetime.timedelta(days=days - 1), datetime.time()).timestamp()
    day_start = datetime.datetime.combine(today, datetime.time()).timestamp()
    with _conn() as conn:
        rows = conn.execute("SELECT zone_id, ts FROM intrusion_events WHERE ts >= ?", (start,)).fetchall()
    per_day = {(today - datetime.timedelta(days=i)).isoformat(): 0 for i in range(days)}
    per_zone_today: dict[int, int] = {}
    last_seen: dict[int, float] = {}
    for zone_id, ts in rows:
        d = datetime.date.fromtimestamp(ts).isoformat()
        if d in per_day:
            per_day[d] += 1
        if ts >= day_start:
            per_zone_today[zone_id] = per_zone_today.get(zone_id, 0) + 1
        last_seen[zone_id] = max(last_seen.get(zone_id, 0), ts)
    return {
        "per_day": [{"date": d, "day": datetime.date.fromisoformat(d).strftime("%a"), "count": n} for d, n in sorted(per_day.items())],
        "per_zone_today": per_zone_today,
        "last_seen": last_seen,
        "today": sum(per_zone_today.values()),
    }


# ---- live detection ---------------------------------------------------------

class _KeepAliveSink:
    def put_nowait(self, item):
        pass


class IntrusionService:
    def __init__(self):
        self._zones: dict[int, list[dict]] = {}
        self._executors: dict[int, concurrent.futures.ThreadPoolExecutor] = {}
        self._futures: dict[int, concurrent.futures.Future] = {}
        self._last_logged: dict[int, float] = {}
        self._sinks: dict[int, _KeepAliveSink] = {}
        self._failed: set[int] = set()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.refresh()
        threading.Thread(target=self._sync_loop, daemon=True, name="intrusion-keepalive").start()

    def refresh(self) -> None:
        by_cam: dict[int, list[dict]] = {}
        for z in list_zones():
            if z["enabled"]:
                by_cam.setdefault(z["camera_id"], []).append(z)
        self._zones = by_cam

    def _sync_loop(self) -> None:
        while True:
            try:
                self.sync()
            except Exception:
                log.exception("intrusion: keep-alive sync failed")
            time.sleep(SYNC_INTERVAL_SECONDS)

    def sync(self) -> None:
        from app import camera_stream

        wanted = set(self._zones) if analytics_settings.enabled("intrusion") else set()
        wanted &= {c["id"] for c in camera_db.list_cameras() if camera_db.is_streamable(c)}
        for cid in wanted - self._sinks.keys():
            sink = _KeepAliveSink()
            camera_stream.get_stream(cid).subscribe(sink, is_collector=True, wants_frames=False)
            self._sinks[cid] = sink
        for cid in self._sinks.keys() - wanted:
            camera_stream.get_stream(cid).unsubscribe(self._sinks.pop(cid))

    # People come from the shared per-camera detector (person_detection.py),
    # about once a second on cameras with a zone inside its active hours.
    def wants(self, camera_id: int) -> float:
        zones = self._zones.get(camera_id)
        if not zones or camera_id in self._failed or not analytics_settings.enabled("intrusion"):
            return 0.0
        return 1.0 / CYCLE_SECONDS if any(is_active(z) for z in zones) else 0.0

    def deliver(self, camera_id: int, frame: np.ndarray, people: list[dict], ts: float) -> None:
        fut = self._futures.get(camera_id)
        if fut is not None and not fut.done():
            return
        active = [z for z in self._zones.get(camera_id, []) if is_active(z, ts)]
        if not active:
            return
        boxes = [p["bbox"] for p in people if p["confidence"] >= PERSON_CONF]
        ex = self._executors.setdefault(
            camera_id, concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"intrusion-{camera_id}"))
        self._futures[camera_id] = ex.submit(self._detect, camera_id, frame, active, ts, boxes)

    def _detect(self, camera_id: int, frame: np.ndarray, zones: list[dict], now: float, boxes: list) -> None:
        if not boxes:
            return
        h, w = frame.shape[:2]
        for zone in zones:
            inside = [b for b in boxes if in_polygon((b[0] + b[2]) / 2 / w, b[3] / h, zone["polygon"])]
            if not inside:
                continue
            if now - self._last_logged.get(zone["id"], 0) >= EVENT_LOG_SECONDS:
                self._last_logged[zone["id"]] = now
                with _conn() as conn:
                    conn.execute("INSERT INTO intrusion_events (zone_id, camera_id, ts, people) VALUES (?, ?, ?, ?)",
                                 (zone["id"], camera_id, now, len(inside)))
            alerts.raise_alert(
                "Intrusion Detected", "High", camera_id,
                message=f"{len(inside)} {'person' if len(inside) == 1 else 'people'} inside {zone['name']}",
                dedupe_key=f"intrusion:{zone['id']}", snapshot_jpeg=_annotate(frame, zone, inside), now=now,
            )


def _annotate(frame: np.ndarray, zone: dict, boxes) -> bytes | None:
    img = frame.copy()
    h, w = img.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in zone["polygon"]], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (60, 60, 230))
    img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    cv2.polylines(img, [pts], True, (60, 60, 230), 3)
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    if w > 1280:
        img = cv2.resize(img, (1280, round(h * 1280 / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes() if ok else None


service = IntrusionService()

from app import person_detection  # noqa: E402

person_detection.service.register("intrusion", service.wants, service.deliver)
