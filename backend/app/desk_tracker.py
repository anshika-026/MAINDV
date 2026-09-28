"""
desk_tracker.py

Who is at which desk, live. Ported from the Deco-vision project's
desk_tracker.py: a per-employee state machine UNKNOWN -> AT_DESK(desk) ->
AWAY -> AT_DESK(desk') ..., with every change written straight to desk_db.

Desks aren't assigned to anyone. Each cycle, every face that face
recognition confidently identified (see desks.py) is placed at whichever
desk outline its centre falls inside; that alone opens a desk session or
moves someone to another desk. An employee not confirmed at their desk for
DESK_SESSION_GRACE_SECONDS goes "away" until they're next seen at a desk.

Difference from Deco-vision: it also ran a body-pose model every 20 s to
keep a seated person's session open while their face wasn't visible. That
model isn't loaded here (this box is already CPU-bound), so the grace period
is longer instead — 90 s by default rather than 20 s.
"""

import logging
import os
import time

from app import desk_db

log = logging.getLogger("desk_tracker")

DESK_SESSION_GRACE_SECONDS = float(os.environ.get("DESK_SESSION_GRACE_SECONDS", "90"))
# A desk already held by someone recently seen there isn't taken from them
# when a second person is identified inside the same outline (desks share
# edges, and two people can lean into one): only a holder not seen for
# this long is displaced.
DISPLACE_AFTER_SECONDS = 10.0


def in_polygon(cx: float, cy: float, polygon: list) -> bool:
    """Ray-casting point-in-polygon, fractions of the frame."""
    inside = False
    n = len(polygon)
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if cy > min(y1, y2) and cy <= max(y1, y2) and cx <= max(x1, x2) and y1 != y2:
            x_cross = (cy - y1) * (x2 - x1) / (y2 - y1) + x1
            if x1 == x2 or cx <= x_cross:
                inside = not inside
        x1, y1 = x2, y2
    return inside


class DeskTracker:
    def __init__(self, now: float | None = None):
        self._zones_by_camera: dict[int, list[dict]] = {}
        self._state: dict[str, dict] = {}
        self.refresh_zones()
        self._rehydrate(now if now is not None else time.time())

    def refresh_zones(self) -> None:
        zones = desk_db.list_zones()
        by_camera: dict[int, list[dict]] = {}
        for z in zones:
            by_camera.setdefault(z["camera_id"], []).append(z)
        self._zones_by_camera = by_camera
        live = {z["id"] for z in zones}
        for emp, st in list(self._state.items()):
            if st["status"] == "at_desk" and st["zone_id"] not in live:
                del self._state[emp]

    def zones_for(self, camera_id: int) -> list[dict]:
        return self._zones_by_camera.get(camera_id, [])

    def _rehydrate(self, now: float) -> None:
        """Pick up sessions still inside the grace period after a restart."""
        for row in desk_db.load_open_sessions(DESK_SESSION_GRACE_SECONDS, now):
            self._state[row["employee_id"]] = {"status": "at_desk", "zone_id": row["zone_id"],
                                               "session_id": row["id"], "last_seen": row["end_ts"]}
        for row in desk_db.load_open_away(DESK_SESSION_GRACE_SECONDS, now):
            self._state.setdefault(row["employee_id"], {"status": "away", "away_id": row["id"], "last_seen": row["end_ts"]})

    def _occupant_of(self, zone_id: int) -> str | None:
        for emp, st in self._state.items():
            if st["status"] == "at_desk" and st["zone_id"] == zone_id:
                return emp
        return None

    def process_frame(self, camera_id: int, faces: list[dict], frame_w: int, frame_h: int, now: float | None = None) -> None:
        """faces: [{"employee_id", "bbox": [x1, y1, x2, y2] in pixels, "score"}]"""
        now = now if now is not None else time.time()
        zones = self.zones_for(camera_id)
        if zones and frame_w and frame_h:
            for face in faces:
                emp = face.get("employee_id")
                if not emp:
                    continue
                x1, y1, x2, y2 = face["bbox"]
                cx, cy = (x1 + x2) / 2 / frame_w, (y1 + y2) / 2 / frame_h
                for zone in zones:
                    if in_polygon(cx, cy, zone["polygon"]):
                        self._confirm_at_desk(emp, zone, camera_id, face.get("score"), now)
                        break
        self._evict_stale(now)

    def _confirm_at_desk(self, emp: str, zone: dict, camera_id: int, confidence, now: float) -> None:
        zid = zone["id"]
        st = self._state.get(emp)

        if st and st["status"] == "at_desk" and st["zone_id"] == zid:
            desk_db.touch_session(st["session_id"], now, confidence)
            st["last_seen"] = now
            return

        holder = self._occupant_of(zid)
        if holder is not None and holder != emp and now - self._state[holder]["last_seen"] > DISPLACE_AFTER_SECONDS:
            self._end_desk_session(holder, self._state[holder]["last_seen"], reason="displaced")

        if st and st["status"] == "at_desk":
            old = st["zone_id"]
            desk_db.touch_session(st["session_id"], now, confidence)
            sid = desk_db.start_session(zid, emp, camera_id, now, confidence)
            desk_db.log_event(emp, "desk_switch", now, zone_id=zid, confidence=confidence, details=f"desk {old} -> desk {zid}")
            self._state[emp] = {"status": "at_desk", "zone_id": zid, "session_id": sid, "last_seen": now}
            return

        if st and st["status"] == "away":
            desk_db.touch_away(st["away_id"], now)
            desk_db.log_event(emp, "away_end", now, confidence=confidence)

        sid = desk_db.start_session(zid, emp, camera_id, now, confidence)
        desk_db.log_event(emp, "session_start", now, zone_id=zid, confidence=confidence, details=zone["zone_label"])
        self._state[emp] = {"status": "at_desk", "zone_id": zid, "session_id": sid, "last_seen": now}

    def _end_desk_session(self, emp: str, at: float, reason: str) -> None:
        st = self._state.get(emp)
        if not st or st["status"] != "at_desk":
            return
        desk_db.touch_session(st["session_id"], at)
        desk_db.log_event(emp, "session_end", at, zone_id=st["zone_id"], details=reason)
        away_id = desk_db.start_away(emp, at)
        desk_db.log_event(emp, "away_start", at, details=reason)
        self._state[emp] = {"status": "away", "away_id": away_id, "last_seen": at}

    def _evict_stale(self, now: float) -> None:
        for emp, st in list(self._state.items()):
            if st["status"] == "at_desk" and now - st["last_seen"] > DESK_SESSION_GRACE_SECONDS:
                self._end_desk_session(emp, st["last_seen"], reason="not seen at desk")

    def live_status(self) -> dict[str, dict]:
        """employee_id -> {"status": "at_desk" | "away", "zone_label"}."""
        labels = {z["id"]: z["zone_label"] for zs in self._zones_by_camera.values() for z in zs}
        return {
            emp: {"status": st["status"], "zone_label": labels.get(st.get("zone_id")) if st["status"] == "at_desk" else None}
            for emp, st in self._state.items()
        }
