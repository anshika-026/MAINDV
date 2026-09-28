"""Staff Count thresholds, loaded from backend/config/staff_analytics.yaml with
built-in defaults for anything missing (see that file for what each means)."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "staff_analytics.yaml"

DEFAULTS = {
    "detection_confidence": 0.45,
    "tracker": "bytetrack.yaml",
    "imgsz": 640,
    "process_fps": 5,
    "track_lost_timeout": 8,
    "minimum_crossing_distance": 20,
    "line_margin": 0.15,
    "entry_exit_cooldown": 3,
    "crossing_confirm_seconds": 1.0,
    "recognition_confidence": 0.70,
    "recognition_min_votes": 2,
    "identify_attempts_per_cycle": 2,
    "late_identification_seconds": 10,
    "reid_match_threshold": 0.72,
    "reid_min_margin": 0.05,
    "end_of_day": "23:59",
}


def load(path: Path = CONFIG_PATH) -> dict:
    cfg = dict(DEFAULTS)
    if path.is_file():
        with open(path, encoding="utf8") as f:
            data = yaml.safe_load(f) or {}
        for key, value in (data.get("staff_analytics") or {}).items():
            if key in DEFAULTS and value is not None:
                cfg[key] = type(DEFAULTS[key])(value)
    return cfg


SETTINGS = load()
