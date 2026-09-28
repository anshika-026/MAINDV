"""Analytics on/off switches (app/analytics_settings.py)."""

from app import analytics_settings as a


def test_switches_persist_and_respect_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "DB_PATH", tmp_path / "s.db")
    monkeypatch.setattr(a, "_state", {k: True for k in a.FEATURES})
    a.init_db()
    assert all(a.enabled(k) for k in a.FEATURES)

    a.set_enabled("face_recognition", False)
    assert not a.enabled("face_recognition")
    assert not a.enabled("desk_analytics")        # needs face recognition
    assert a.snapshot()["desk_analytics"]["on"]    # its own switch is untouched
    assert a.enabled("footfall")

    # Survives a restart.
    monkeypatch.setattr(a, "_state", {k: True for k in a.FEATURES})
    a.init_db()
    assert not a.enabled("face_recognition")
