"""Intrusion zones (app/intrusion.py), fed person boxes as the shared detector would."""

import datetime

import numpy as np
import pytest

from app import alerts, analytics_settings, camera_db, intrusion


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    for mod in (intrusion, alerts):
        monkeypatch.setattr(mod, "DB_PATH", tmp_path / "i.db")
    monkeypatch.setattr(alerts, "SNAPSHOT_DIR", tmp_path / "snaps")
    monkeypatch.setattr(analytics_settings, "_state", {k: True for k in analytics_settings.FEATURES})
    monkeypatch.setattr(camera_db, "list_cameras", lambda: [{"id": 5, "name": "Store room", "site": "Noida Site"}])
    monkeypatch.setattr(intrusion, "service", intrusion.IntrusionService())
    intrusion.init_db()
    alerts.init_db()


# Zone = right half of a 1000x500 frame.
RIGHT = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
FRAME = np.zeros((500, 1000, 3), dtype=np.uint8)


def at(h, m):
    return datetime.datetime(2026, 9, 28, h, m).timestamp()


def test_active_window_including_overnight():
    z = {"enabled": True, "active_from": "19:00", "active_to": "08:00"}
    assert intrusion.is_active(z, at(22, 0)) and intrusion.is_active(z, at(7, 30))
    assert not intrusion.is_active(z, at(12, 0))
    assert intrusion.is_active({"enabled": True, "active_from": None, "active_to": None}, at(12, 0))
    assert not intrusion.is_active({"enabled": False, "active_from": None, "active_to": None}, at(12, 0))


def test_person_whose_feet_are_in_the_zone_raises_one_alert():
    zone = intrusion.create_zone(5, "Store room", RIGHT, None, None)
    svc = intrusion.service
    # Person standing in the zone (feet at x=0.7), and one outside it (feet at x=0.2).
    boxes = [[650, 100, 750, 450], [150, 100, 250, 450]]
    svc._detect(5, FRAME, [zone], at(22, 0), boxes)
    svc._detect(5, FRAME, [zone], at(22, 0) + 5, boxes)  # still there: same alert, counted again
    rows = alerts.list_alerts("all")
    assert [(r["event"], r["occurrences"], r["has_snapshot"]) for r in rows] == [("Intrusion Detected", 2, True)]
    assert "1 person inside Store room" in rows[0]["message"]


def test_body_overlapping_zone_but_feet_outside_is_ignored():
    zone = intrusion.create_zone(5, "Store room", RIGHT, None, None)
    # Box straddles the edge; feet (bottom centre x=0.45) are outside the zone.
    intrusion.service._detect(5, FRAME, [zone], at(22, 0), [[350, 100, 550, 450]])
    assert alerts.list_alerts("all") == []
