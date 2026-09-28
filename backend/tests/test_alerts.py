"""Alerts (app/alerts.py) against a throwaway SQLite file."""

import pytest

from app import alerts, camera_db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "DB_PATH", tmp_path / "alerts.db")
    monkeypatch.setattr(alerts, "SNAPSHOT_DIR", tmp_path / "snaps")
    monkeypatch.setattr(alerts, "_last_unknown", {})
    cams = {8: {"id": 8, "name": "Entry", "purpose": "Entry/Exit", "site": "Noida Site"},
            11: {"id": 11, "name": "Main Hall 1", "purpose": "General", "site": "Noida Site"}}
    monkeypatch.setattr(camera_db, "get_camera", lambda cid: cams.get(cid))
    monkeypatch.setattr(camera_db, "list_cameras", lambda: list(cams.values()))
    alerts.init_db()


def test_open_alert_with_same_key_is_updated_not_duplicated():
    a = alerts.raise_alert("Camera Offline", "Critical", 8, dedupe_key="offline:8")
    b = alerts.raise_alert("Camera Offline", "Critical", 8, dedupe_key="offline:8")
    assert a == b
    [row] = alerts.list_alerts("all")
    assert row["occurrences"] == 2 and row["camera_name"] == "Entry" and row["location"] == "Noida Site"
    alerts.resolve_open("offline:8", "Video came back")
    assert alerts.list_alerts("all")[0]["status"] == "Resolved"
    # Offline again later: a new alert, since the old one is closed.
    assert alerts.raise_alert("Camera Offline", "Critical", 8, dedupe_key="offline:8") != a


def test_acknowledge_then_resolve_records_who():
    aid = alerts.raise_alert("Unknown Person", "High", 8)
    assert alerts.acknowledge(aid, "admin@test.com")
    assert alerts.resolve(aid, "admin@test.com", "False alarm")
    row = alerts.list_alerts("all")[0]
    assert (row["status"], row["acknowledged_by"], row["resolved_by"], row["resolution_reason"]) == \
        ("Resolved", "admin@test.com", "admin@test.com", "False alarm")
    s = alerts.summary()
    assert s["resolved_today"] == 1 and s["active"] == 0


def test_unknown_face_only_at_gates_only_when_clearly_unknown_and_rate_limited():
    alerts.unknown_face(11, 0.1, b"jpg")      # desk camera: ignored
    alerts.unknown_face(8, 0.55, b"jpg")      # probably a badly-seen employee: ignored
    alerts.unknown_face(8, 0.2, b"jpg")       # stranger at the gate: alert, with snapshot
    alerts.unknown_face(8, 0.1, b"jpg")       # within the cooldown: ignored
    rows = alerts.list_alerts("all")
    assert len(rows) == 1 and rows[0]["event"] == "Unknown Person" and rows[0]["has_snapshot"]
    assert alerts.snapshot_path(rows[0]["id"]).read_bytes() == b"jpg"
