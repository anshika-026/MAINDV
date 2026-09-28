"""Desk analytics (app/desk_tracker.py + desk_db.py) against a throwaway
SQLite file. Run from backend/:  python -m pytest tests -v"""

import datetime

import pytest

from app import desk_db, desk_tracker
from app.desk_tracker import DeskTracker, in_polygon

W, H = 1000, 500
CAM = 11
DAY = "2026-09-28"
T0 = datetime.datetime(2026, 9, 28, 10, 0).timestamp()

# Two desks side by side: left half and right half of the frame.
LEFT = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
RIGHT = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.setattr(desk_db, "DB_PATH", tmp_path / "desks.db")
    monkeypatch.setattr(desk_tracker, "DESK_SESSION_GRACE_SECONDS", 60)
    desk_db.init_db()
    desk_db.create_zone(CAM, LEFT)
    desk_db.create_zone(CAM, RIGHT)
    return DeskTracker(now=T0)


def face(emp, side):
    x = 200 if side == "left" else 700
    return {"employee_id": emp, "bbox": [x, 200, x + 60, 260], "score": 0.9}


def row(emp):
    return next(r for r in desk_db.get_daily_report(DAY) if r["employee_id"] == emp)


def test_auto_labels_and_polygon_test(tracker):
    assert [z["zone_label"] for z in desk_db.list_zones(CAM)] == ["Desk 1", "Desk 2"]
    assert in_polygon(0.2, 0.5, LEFT) and not in_polygon(0.7, 0.5, LEFT)
    assert in_polygon(0.3, 0.3, [[0, 0], [1, 0], [0, 1]]) and not in_polygon(0.8, 0.8, [[0, 0], [1, 0], [0, 1]])


def test_desk_time_away_time_and_switch(tracker):
    # 10:00-10:10 at the left desk, seen every 30 s.
    for i in range(21):
        tracker.process_frame(CAM, [face("001", "left")], W, H, T0 + i * 30)
    assert tracker.live_status()["001"] == {"status": "at_desk", "zone_label": "Desk 1"}
    # Not seen for 5 minutes -> away from 10:10 (last sighting), after the 60 s grace.
    tracker.process_frame(CAM, [], W, H, T0 + 600 + 61)
    assert tracker.live_status()["001"]["status"] == "away"
    # Back at the other desk at 10:15 for 5 minutes.
    for i in range(11):
        tracker.process_frame(CAM, [face("001", "right")], W, H, T0 + 900 + i * 30)
    r = row("001")
    assert r["desk_seconds"] == 600 + 300
    assert r["away_seconds"] == 300
    assert tracker.live_status()["001"] == {"status": "at_desk", "zone_label": "Desk 2"}


def test_switching_desks_counts_a_movement(tracker):
    tracker.process_frame(CAM, [face("008", "left")], W, H, T0)
    tracker.process_frame(CAM, [face("008", "right")], W, H, T0 + 20)
    assert row("008")["movements"] == 1


def test_open_away_counts_until_now(tracker):
    tracker.process_frame(CAM, [face("013", "left")], W, H, T0)
    tracker.process_frame(CAM, [], W, H, T0 + 120)  # grace lapses: away from T0
    rows = desk_db.get_daily_report(DAY, open_away_until={"013": T0 + 1800})
    assert next(r for r in rows if r["employee_id"] == "013")["away_seconds"] == 1800


def test_two_people_recently_seen_at_one_desk_are_both_kept(tracker):
    tracker.process_frame(CAM, [face("001", "left"), face("006", "left")], W, H, T0)
    live = tracker.live_status()
    assert live["001"]["status"] == live["006"]["status"] == "at_desk"


def test_restart_picks_up_open_sessions(tracker):
    tracker.process_frame(CAM, [face("001", "left")], W, H, T0)
    again = DeskTracker(now=T0 + 30)
    assert again.live_status()["001"]["status"] == "at_desk"
