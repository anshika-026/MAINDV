"""
tracker.py

StaffCameraTracker: turns one entrance camera's tracked people into ENTRY and
EXIT events for the StaffOccupancyManager. It doesn't run the detector itself:
update() takes this frame's tracked people ({track_id, bbox}), so tests can
feed scripted movements and the live service (service.py) feeds YOLO+ByteTrack.

Line crossing, per track, using the bottom-centre of the body box (the feet):
  - The signed distance to the entry line puts the feet "in" (office side),
    "out", or inside a +/- minimum_crossing_distance band around the line.
  - A track's side only changes when its feet land clearly on the other side,
    beyond the band. Standing on the line, stepping back and forth inside the
    band, or detection jitter never counts as a crossing.
  - The crossing must also fall along the line (its length plus line_margin
    at each end), so someone walking past well to the side doesn't count.
  - Out -> in is ENTRY, in -> out is EXIT. A track's first confirmed side is
    only a starting point: someone first seen already inside doesn't enter.
  - A change of side is only confirmed once the person has STAYED on the new
    side for crossing_confirm_seconds, or disappeared while on it (walking out
    of view through the door). Stepping over the line and straight back,
    however far, produces no events at all.
  - Within entry_exit_cooldown of the track's last event a change of side is
    held, not dropped: if they're still on the new side once it's over, the
    event fires then.
  - Only people in the office ROI (if one is set) take part.

A track that vanishes (occlusion, missed detections, camera dropout) is only
forgotten after track_lost_timeout. Forgetting it is never an exit: occupancy
only changes on a real crossing, so ENTRY -> lost -> found can't produce
ENTRY, EXIT, ENTRY.

Identity: while a track is visible, face reads (up to identify_attempts_per_cycle
per frame) vote for an employee. Only recognition_min_votes agreeing reads at
recognition_confidence or more confirm an identity. A face confirmed within
late_identification_seconds after an anonymous ENTRY upgrades that entry.
"""

import time
from collections import Counter

import numpy as np

from app.desk_tracker import in_polygon
from app.staff import config as staff_config

IDENTIFY_INTERVAL = 0.4  # s between face reads of the same track


class _Track:
    __slots__ = ("track_id", "bbox", "foot", "prev_foot", "zone", "side", "pending_side", "last_seen",
                 "first_seen", "votes", "best_conf", "employee_id", "id_conf", "last_id_try",
                 "visit_id", "last_event", "last_event_time", "in_roi_ever", "pending_since", "pending_in_roi", "last_emb")

    def __init__(self, track_id, ts):
        self.track_id = track_id
        self.bbox = None
        self.foot = self.prev_foot = None
        self.zone = None           # this frame: in | out | band
        self.side = None           # last confirmed side: in | out
        self.pending_side = None   # side change not yet confirmed
        self.pending_since = 0.0
        self.pending_in_roi = False
        self.last_emb = None       # body embedding while a crossing is pending, in case they vanish
        self.first_seen = self.last_seen = ts
        self.votes = Counter()
        self.best_conf: dict[str, float] = {}
        self.employee_id = None
        self.id_conf = None
        self.last_id_try = 0.0
        self.visit_id = None       # visit this track opened by entering
        self.last_event = None
        self.last_event_time = -1e9
        self.in_roi_ever = False


class StaffCameraTracker:
    def __init__(self, camera_id: int, manager, line, inside_sign: int = 1, roi=None, cfg: dict | None = None):
        """line: [[x1, y1], [x2, y2]] and roi: [[x, y], ...], fractions of the frame."""
        self.camera_id = camera_id
        self.manager = manager
        self.line = line
        self.inside_sign = 1 if inside_sign >= 0 else -1
        self.roi = roi
        self.cfg = cfg or staff_config.SETTINGS
        self.tracks: dict[int, _Track] = {}
        self.frame_size = None

    # ---- geometry ---------------------------------------------------------------

    def _line_px(self, w, h):
        (x1, y1), (x2, y2) = self.line
        return np.array([x1 * w, y1 * h]), np.array([x2 * w, y2 * h])

    def _zone(self, foot, w, h) -> tuple[str, bool]:
        """(in | out | band, whether the point is alongside the line)."""
        a, b = self._line_px(w, h)
        ab = b - a
        length = float(np.linalg.norm(ab)) or 1.0
        p = np.asarray(foot, dtype=float)
        # Signed perpendicular distance in px, positive on the office side.
        dist = float(ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])) / length * self.inside_sign
        t = float(np.dot(p - a, ab)) / (length * length)
        alongside = -self.cfg["line_margin"] <= t <= 1 + self.cfg["line_margin"]
        band = self.cfg["minimum_crossing_distance"]
        return ("in" if dist > band else "out" if dist < -band else "band"), alongside

    # ---- per frame --------------------------------------------------------------

    def update(self, people: list[dict], frame_shape, ts: float | None = None, identify=None, embed=None) -> list[dict]:
        """people: [{"track_id": int, "bbox": [x1, y1, x2, y2]}] in pixels.
        identify(bbox) -> (employee_id | None, confidence); embed(bbox) -> body embedding | None."""
        ts = ts if ts is not None else time.time()
        h, w = frame_shape[:2]
        self.frame_size = (w, h)
        events = []
        attempts = 0
        for p in people:
            tr = self.tracks.get(p["track_id"])
            if tr is None:
                tr = self.tracks[p["track_id"]] = _Track(p["track_id"], ts)
            x1, y1, x2, y2 = p["bbox"]
            tr.bbox = [float(v) for v in p["bbox"]]
            tr.prev_foot, tr.foot = tr.foot, ((x1 + x2) / 2, y2)
            tr.last_seen = ts
            in_roi = self.roi is None or in_polygon(tr.foot[0] / w, tr.foot[1] / h, self.roi)
            tr.in_roi_ever = tr.in_roi_ever or in_roi

            if identify is not None and tr.employee_id is None and attempts < self.cfg["identify_attempts_per_cycle"] \
                    and ts - tr.last_id_try >= IDENTIFY_INTERVAL:
                attempts += 1
                tr.last_id_try = ts
                events += self._vote(tr, *identify(tr.bbox), ts)

            zone, alongside = self._zone(tr.foot, w, h)
            tr.zone = zone
            if zone == "band" or not alongside:
                continue
            if tr.side is None:
                tr.side = zone  # first confirmed position: a starting point, not a crossing
                continue
            if zone == tr.side:
                tr.pending_side = None  # came back before the change was confirmed
                continue
            if tr.pending_side != zone:
                tr.pending_side, tr.pending_since = zone, ts
                if embed is not None:
                    tr.last_emb = embed(tr.bbox)
            tr.pending_in_roi = tr.pending_in_roi or in_roi
            if ts - tr.pending_since < self.cfg["crossing_confirm_seconds"]:
                continue  # not confirmed yet
            if ts - tr.last_event_time < self.cfg["entry_exit_cooldown"]:
                continue  # held until the cooldown is over (see module docstring)
            ev = self._cross(tr, zone, ts, embed, in_roi or tr.pending_in_roi)
            if ev:
                events.append(ev)

        timeout = self.cfg["track_lost_timeout"]
        for tid in [tid for tid, tr in self.tracks.items() if ts - tr.last_seen > timeout]:
            tr = self.tracks.pop(tid)  # forgotten; that alone is NOT an exit...
            if tr.pending_side and tr.pending_side != tr.side:
                # ...but vanishing right after crossing (out of view through the
                # door) confirms that crossing, timed at the last sighting.
                ev = self._cross(tr, tr.pending_side, tr.last_seen, embed, tr.pending_in_roi, frame_embed=False)
                if ev:
                    events.append(ev)
        return events

    def _cross(self, tr: _Track, new_side: str, ts: float, embed, in_roi: bool, frame_embed: bool = True) -> dict | None:
        """frame_embed=False: the track isn't in this frame (confirmed by
        vanishing), so use the body embedding kept from when it crossed."""
        tr.side, tr.pending_side, tr.pending_in_roi = new_side, None, False
        if self.roi is not None and not (in_roi or tr.in_roi_ever):
            return None
        emb = (embed(tr.bbox) if embed is not None else None) if frame_embed else tr.last_emb
        tr.last_event_time = ts
        if new_side == "in":
            tr.last_event = "ENTRY"
            ev = self.manager.entry(self.camera_id, tr.track_id, tr.employee_id, tr.id_conf, emb, ts)
            tr.visit_id = ev["visit_id"] if ev["event_type"] == "ENTRY" else None
        else:
            tr.last_event = "EXIT"
            ev = self.manager.exit(self.camera_id, tr.track_id, tr.employee_id, tr.id_conf, emb, ts, visit_id=tr.visit_id)
            tr.visit_id = None
        return ev

    def _vote(self, tr: _Track, employee_id, confidence, ts) -> list[dict]:
        if not employee_id or confidence is None or confidence < self.cfg["recognition_confidence"]:
            return []
        tr.votes[employee_id] += 1
        tr.best_conf[employee_id] = max(tr.best_conf.get(employee_id, 0), confidence)
        (top, n), *rest = tr.votes.most_common(2) + [(None, 0)]
        runner = rest[0][1] if rest else 0
        if n < self.cfg["recognition_min_votes"] or n <= runner:
            return []
        tr.employee_id, tr.id_conf = top, tr.best_conf[top]
        if tr.visit_id and tr.last_event == "ENTRY" and ts - tr.last_event_time <= self.cfg["late_identification_seconds"]:
            ev = self.manager.identify_late(tr.visit_id, top, tr.id_conf, ts)
            if ev and ev["event_type"] == "DUPLICATE_ENTRY":
                tr.visit_id = None
            return [ev] if ev else []
        return []

    # ---- debug view -------------------------------------------------------------

    def debug(self, now: float | None = None) -> list[dict]:
        now = now if now is not None else time.time()
        return [
            {"track_id": t.track_id, "bbox": t.bbox, "zone": t.zone, "side": t.side, "employee_id": t.employee_id,
             "confidence": t.id_conf, "last_event": t.last_event,
             "event_age": round(now - t.last_event_time, 1) if t.last_event else None,
             "lost_for": round(now - t.last_seen, 1)}
            for t in self.tracks.values()
        ]
