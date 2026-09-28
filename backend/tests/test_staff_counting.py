"""
Staff Count scenarios (app/staff/tracker.py + occupancy.py), run through the
real line-crossing tracker and occupancy manager with scripted people. Only
the camera side is simulated: tracked boxes arrive at 5 fps exactly as
YOLO + ByteTrack would hand them over, face reads come from a scripted
"identify", and body appearance from a scripted "embed".

Run from backend/ (the -s shows the results table):
    python -m pytest tests/test_staff_counting.py -v -s

Frame is 1000 x 500. The entry line runs across the doorway at y = 250 from
x = 200 to 800; the office is below it (towards the camera).
"""

import numpy as np
import pytest

from app.staff import occupancy
from app.staff.occupancy import StaffOccupancyManager
from app.staff.tracker import StaffCameraTracker

W, H = 1000, 500
LINE = [[0.2, 0.5], [0.8, 0.5]]
FPS = 5
DT = 1 / FPS
CFG = {
    "detection_confidence": 0.45, "tracker": "bytetrack.yaml", "imgsz": 640, "process_fps": FPS,
    "track_lost_timeout": 8, "minimum_crossing_distance": 20, "line_margin": 0.15, "entry_exit_cooldown": 3,
    "crossing_confirm_seconds": 1.0,
    "recognition_confidence": 0.70, "recognition_min_votes": 2, "identify_attempts_per_cycle": 4,
    "late_identification_seconds": 10, "reid_match_threshold": 0.72, "reid_min_margin": 0.05, "end_of_day": "23:59",
}
T0 = 1790600000.0  # a fixed afternoon

RESULTS = []


def appearance(person: str) -> np.ndarray:
    """Stable body-appearance vector per person (different people ~orthogonal)."""
    v = np.random.RandomState(abs(hash(person)) % 2**31).randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


class World:
    """Scripted people in front of one or more entrance cameras."""

    def __init__(self, manager):
        self.m = manager
        self.cams: dict[int, StaffCameraTracker] = {}
        self.t = T0
        self.visible: dict[int, dict[int, dict]] = {}  # camera -> track -> {x, y, person, face}

    def camera(self, cid):
        self.cams[cid] = StaffCameraTracker(cid, self.m, LINE, 1, None, CFG)
        self.visible[cid] = {}
        return self.cams[cid]

    def place(self, cid, track, person, x, y, face=True):
        self.visible[cid][track] = {"x": x, "y": y, "person": person, "face": face}

    def remove(self, cid, track):
        self.visible[cid].pop(track, None)

    def step(self, seconds=DT):
        self.t += seconds
        for cid, cam in self.cams.items():
            vis = self.visible[cid]
            people = [{"track_id": tid, "bbox": [p["x"] - 40, p["y"] - 200, p["x"] + 40, p["y"]]} for tid, p in vis.items()]

            def identify(bbox, vis=vis):
                p = next(p for p in vis.values() if abs((bbox[0] + bbox[2]) / 2 - p["x"]) < 1 and abs(bbox[3] - p["y"]) < 1)
                if not p["face"] or p["person"].startswith("visitor"):
                    return None, 0.0
                return p["person"], 0.92

            def embed(bbox, vis=vis):
                p = next(p for p in vis.values() if abs((bbox[0] + bbox[2]) / 2 - p["x"]) < 1 and abs(bbox[3] - p["y"]) < 1)
                return appearance(p["person"])

            cam.update(people, (H, W), self.t, identify, embed)

    def walk(self, cid, tracks: dict[int, str], y_from, y_to, seconds=1.2, face=True, x=500, then_hold=4.0):
        """Tracks walk together from y_from to y_to, then stay put for `then_hold` s."""
        n = max(2, int(seconds * FPS))
        for i in range(n + 1):
            y = y_from + (y_to - y_from) * i / n
            for k, (tid, person) in enumerate(tracks.items()):
                self.place(cid, tid, person, x + 90 * k, y, face)
            self.step()
        for _ in range(int(then_hold * FPS)):
            self.step()


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(occupancy, "DB_PATH", tmp_path / "staff.db")
    occupancy.init_db()
    w = World(StaffOccupancyManager(CFG))
    w.camera(8)
    return w


def events(m, *types):
    return [e for e in reversed(m.events(limit=1000)) if not types or e["event_type"] in types]


def record(name, m, expected_count, expected_entries, expected_exits):
    c = m.counts()
    entries = len(events(m, "ENTRY"))
    exits = len(events(m, "EXIT"))
    dup_events = sum(1 for e in events(m, "ENTRY") for f in events(m, "ENTRY")
                     if e["id"] < f["id"] and e["employee_id"] and e["employee_id"] == f["employee_id"]
                     and not any(x["event_type"] == "EXIT" and x["employee_id"] == e["employee_id"] and e["ts"] < x["ts"] < f["ts"] for x in events(m, "EXIT")))
    RESULTS.append({
        "scenario": name, "expected": expected_count, "actual": c["total_persons"],
        "false_entry": max(0, entries - expected_entries), "false_exit": max(0, exits - expected_exits),
        "duplicate": dup_events, "missed": max(0, expected_entries - entries) + max(0, expected_exits - exits),
    })
    return c


# ---- the 15 scenarios from the spec (14 and 15 need real footage: see below) ----

def test_01_one_employee_enters(world):
    world.walk(8, {1: "001"}, 150, 380)
    c = record("1. one employee enters", world.m, 1, 1, 0)
    assert (c["current_staff_count"], c["known_staff"]) == (1, 1)
    assert [e["employee_id"] for e in events(world.m, "ENTRY")] == ["001"]


def test_02_one_employee_exits_with_back_to_camera(world):
    world.walk(8, {1: "001"}, 150, 380)
    world.remove(8, 1)
    world.step(12)  # track long gone
    world.walk(8, {5: "001"}, 380, 150, face=False)  # walking away: no face, body only
    c = record("2. one employee exits", world.m, 0, 1, 1)
    assert c["current_staff_count"] == 0
    [ex] = events(world.m, "EXIT")
    assert (ex["employee_id"], ex["identity_source"]) == ("001", "reid")


def test_03_same_employee_stays_30_minutes(world):
    world.walk(8, {1: "001"}, 150, 400)
    for i in range(30 * 60 // 5):  # every 5 s for 30 min, fidgeting inside the office
        world.place(8, 1, "001", 500 + (i % 3) * 10, 400 + (i % 2) * 15)
        world.step(5)
    c = record("3. stays 30 minutes", world.m, 1, 1, 0)
    assert c["current_staff_count"] == 1 and len(events(world.m)) == 1


def test_04_two_enter_simultaneously(world):
    world.walk(8, {1: "001", 2: "006"}, 150, 380)
    c = record("4. two enter together", world.m, 2, 2, 0)
    assert c["current_staff_count"] == 2


def test_05_five_enter_together(world):
    world.walk(8, {i: f"00{i}" for i in range(1, 6)}, 150, 380, x=280)
    c = record("5. five enter together", world.m, 5, 5, 0)
    assert c["current_staff_count"] == 5


def test_06_enters_and_immediately_exits(world):
    world.walk(8, {1: "001"}, 150, 330, seconds=0.8, then_hold=0)
    world.walk(8, {1: "001"}, 330, 120, seconds=0.8, then_hold=5)  # straight back out
    c = record("6. enters then immediately exits", world.m, 0, 0, 0)
    assert c["current_staff_count"] == 0
    assert events(world.m) == []  # never confirmed inside: nothing to count or undo


def test_06b_enters_then_leaves_a_few_seconds_later(world):
    world.walk(8, {1: "001"}, 150, 380, then_hold=2)
    world.walk(8, {1: "001"}, 380, 150, face=False, then_hold=5)
    c = record("6b. enters, leaves 2 s later", world.m, 0, 1, 1)
    assert c["current_staff_count"] == 0
    assert [e["event_type"] for e in events(world.m)] == ["ENTRY", "EXIT"]


def test_06c_crosses_then_walks_out_of_view(world):
    world.walk(8, {1: "001"}, 150, 300, seconds=0.6, then_hold=0)  # crosses, then leaves the picture
    world.remove(8, 1)
    world.step(10)                                                  # track expires: confirms the entry
    c = record("6c. crosses then out of view", world.m, 1, 1, 0)
    assert c["current_staff_count"] == 1


def test_07_turns_around_at_the_entrance(world):
    world.walk(8, {1: "001"}, 120, 260, then_hold=0)  # into the band around the line...
    world.walk(8, {1: "001"}, 260, 120, then_hold=3)  # ...and back out without crossing
    c = record("7. turns around at the door", world.m, 0, 0, 0)
    assert c["current_staff_count"] == 0 and events(world.m) == []


def test_08_temporarily_disappears(world):
    world.walk(8, {1: "001"}, 150, 380)
    world.remove(8, 1)
    world.step(5)                          # hidden 5 s (< track_lost_timeout)
    world.place(8, 1, "001", 500, 385)
    world.step()
    world.remove(8, 1)
    world.step(20)                         # gone long enough to be forgotten
    world.walk(8, {9: "001"}, 390, 400)    # re-detected inside under a new track id
    c = record("8. temporarily disappears", world.m, 1, 1, 0)
    assert c["current_staff_count"] == 1
    assert [e["event_type"] for e in events(world.m)] == ["ENTRY"]  # no EXIT, no second ENTRY


def test_09_unknown_visitor(world):
    world.walk(8, {1: "visitor-1"}, 150, 380)
    c = record("9. unknown visitor enters", world.m, 1, 1, 0)
    assert (c["current_staff_count"], c["unknown_persons"], c["total_persons"]) == (0, 1, 1)


def test_10_employees_cross_each_other(world):
    world.walk(8, {1: "001"}, 150, 380)
    # 001 leaves (back turned) while 006 arrives, passing each other at the line.
    n = 8
    for i in range(n + 1):
        world.place(8, 1, "001", 500, 380 - 260 * i / n, face=False)
        world.place(8, 2, "006", 520, 120 + 260 * i / n)
        world.step()
    world.step(4)
    c = record("10. two cross at the door", world.m, 1, 2, 1)
    assert c["current_staff_count"] == 1
    assert world.m.present_visit_of("006") and not world.m.present_visit_of("001")


def test_11_camera_temporarily_disconnects(world):
    world.walk(8, {1: "001", 2: "006"}, 150, 380)
    world.visible[8].clear()
    world.step(30)                                   # no frames at all for 30 s
    world.walk(8, {11: "001", 12: "006"}, 380, 390)  # back, with new track ids, still inside
    c = record("11. camera disconnects", world.m, 2, 2, 0)
    assert c["current_staff_count"] == 2 and len(events(world.m)) == 2


def test_12_application_restarts(world):
    world.walk(8, {1: "001", 2: "visitor-1"}, 150, 380)
    # Restart: a new manager and tracker from the same database.
    world.m = StaffOccupancyManager(CFG)
    world.camera(8)
    assert world.m.counts()["total_persons"] == 2
    world.walk(8, {7: "001"}, 380, 150, face=False)  # leaves; matched by the stored body embedding
    c = record("12. application restarts", world.m, 1, 2, 1)
    assert (c["known_staff"], c["unknown_persons"]) == (0, 1)


def test_13_multiple_cameras_see_same_employee(world):
    world.camera(9)                            # a second entrance
    world.walk(8, {1: "001"}, 150, 380)
    world.walk(9, {1: "001"}, 150, 380)        # the same person also crosses the other door's line
    c = record("13. two entrances, same employee", world.m, 1, 1, 0)
    assert c["current_staff_count"] == 1
    assert [e["event_type"] for e in events(world.m)] == ["ENTRY", "DUPLICATE_ENTRY"]


# ---- extra edge cases from the spec ----

def test_standing_on_the_line_never_fires(world):
    for i in range(50):                         # 10 s of jitter within +/-15 px of the line
        world.place(8, 1, "001", 500, 250 + (15 if i % 2 else -15))
        world.step()
    record("standing on the line", world.m, 0, 0, 0)
    assert events(world.m) == []


def test_repeated_crossings_in_one_visit_are_never_duplicate_entries(world):
    world.walk(8, {1: "001"}, 150, 380)
    for _ in range(3):                          # steps back out briefly and in again, quickly
        world.walk(8, {1: "001"}, 380, 200, seconds=0.4, then_hold=0)
        world.walk(8, {1: "001"}, 200, 380, seconds=0.4, then_hold=0)
    world.step(5)
    c = record("wobbling in the doorway", world.m, 1, 1, 0)
    assert c["current_staff_count"] == 1
    assert len(events(world.m, "ENTRY")) == 1


def test_face_seen_only_after_entering_upgrades_the_entry(world):
    world.walk(8, {1: "001"}, 150, 380, face=False, then_hold=2)   # back to camera while entering: anonymous ENTRY
    for _ in range(3):
        world.place(8, 1, "001", 500, 385, face=True)              # turns round inside
        world.step(0.5)
    c = record("face seen after entering", world.m, 1, 1, 0)
    assert (c["known_staff"], c["unknown_persons"]) == (1, 0)
    assert [e["event_type"] for e in events(world.m)] == ["ENTRY", "IDENTIFIED"]


def test_count_never_goes_negative_and_end_of_day_closes_everyone(world):
    world.walk(8, {1: "visitor-1"}, 380, 150, face=False)   # someone leaves who was never counted in
    assert world.m.counts()["total_persons"] == 0
    assert events(world.m)[-1]["event_type"] == "EXIT_UNMATCHED"
    world.walk(8, {2: "001", 3: "006"}, 150, 380)
    assert world.m.auto_exit_all(world.t) == 2
    assert world.m.counts()["total_persons"] == 0


def test_zz_print_results_table():
    """Prints the scenario table (run with -s). 14 (low light) and 15 (crowded
    entrance) can't be meaningfully simulated: run scripts/replay_staff.py on
    recorded footage for those."""
    cols = ["scenario", "expected", "actual", "false_entry", "false_exit", "duplicate", "missed"]
    print("\n" + " | ".join(f"{c:<34}" if c == "scenario" else f"{c:>11}" for c in cols))
    for r in RESULTS:
        print(" | ".join(f"{r[c]:<34}" if c == "scenario" else f"{r[c]:>11}" for c in cols))
    assert all(r["expected"] == r["actual"] and not (r["false_entry"] or r["false_exit"] or r["duplicate"] or r["missed"]) for r in RESULTS)
