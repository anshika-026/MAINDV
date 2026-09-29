"""Attendance marked from face recognition (app/attendance.py), against a
throwaway SQLite file. Run from backend/:  python -m pytest tests -v"""

import datetime

import pytest

from app import alerts, attendance, camera_db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(attendance, "DB_PATH", tmp_path / "att.db")
    monkeypatch.setattr(alerts, "DB_PATH", tmp_path / "att.db")
    alerts.init_db()
    monkeypatch.setattr(attendance, "_last_write", {})
    monkeypatch.setattr(camera_db, "list_cameras", lambda: [{"id": 8, "name": "Entry"}])
    attendance.init_db()


DAY = "2026-09-28"


def at(hh, mm):
    return datetime.datetime(2026, 9, 28, hh, mm).timestamp()


def row(report, emp):
    return next(r for r in report["rows"] if r["employee_id"] == emp)


def test_low_confidence_or_unknown_ids_are_not_marked():
    assert not attendance.record("001", 8, 0.6, at(9, 0))
    assert not attendance.record("999", 8, 0.99, at(9, 0))
    assert row(attendance.day_report(DAY, now=at(10, 0)), "001")["status"] == "Absent"


def test_first_and_last_sighting_make_time_in_and_out():
    attendance.record("001", 8, 0.9, at(9, 10))
    attendance.record("001", 8, 0.95, at(9, 10) + 5)  # within the write throttle: ignored
    attendance.record("001", 8, 0.9, at(17, 45))
    r = row(attendance.day_report(DAY, now=at(19, 0)), "001")
    assert (r["status"], r["time_in"], r["time_out"], r["time_stay"], r["arrival"]) == \
        ("Present", "09:10 AM", "05:45 PM", "8h 35m", "On time")
    assert r["camera"] == "Entry" and r["confidence"] == 90


def test_late_arrival_and_on_site():
    attendance.record("008", 8, 0.9, at(10, 5))
    report = attendance.day_report(DAY, now=at(10, 12))
    r = row(report, "008")
    assert r["arrival"] == "Late arrival"
    # "On site" only makes sense for today; day_report's `now` is the 28th here.
    assert r["status"] in ("On site", "Present")
    # Time out is the last sighting even while they're still on site.
    assert r["time_out"] == "10:05 AM"
    assert report["stats"]["present"] == 1 and report["stats"]["late"] == 1


def test_leave_shows_instead_of_absent():
    attendance.add_leave("013", "2026-09-27", "2026-09-29", "Sick leave")
    report = attendance.day_report(DAY, now=at(12, 0))
    assert row(report, "013")["status"] == "On Leave"
    assert report["stats"]["on_leave"] == 1
    with pytest.raises(ValueError):
        attendance.add_leave("013", "2026-09-29", "2026-09-27", "backwards")


def test_late_first_sighting_raises_one_late_alert():
    attendance.record("008", 8, 0.9, at(10, 5))
    attendance.record("008", 8, 0.9, at(11, 0))  # later sighting same day: no second alert
    rows = alerts.list_alerts("all")
    assert [(r["event"], r["severity"], r["employee_id"]) for r in rows] == [("Late Arrival", "Low", "008")]
