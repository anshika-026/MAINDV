"""
storage.py

Portable image paths. The database stores every image location RELATIVE to
config.DATA_DIR in POSIX form ("face_training/018/cam3_track9_ab12.jpg"), and
absolute filesystem paths exist only at runtime. Moving or restoring the data
directory to another machine therefore needs no database rewrite.

Rows written before this existed hold absolute paths from whichever machine
wrote them (C:\\Users\\<someone>\\...\\backend\\data\\face_training\\...) or
cwd-relative ones ("backend/data/face_enroll\\018.png"). resolve() still finds
those files when they live under the current DATA_DIR, and
scripts/migrate_paths.py rewrites them to the relative form once.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

from . import config

# Top-level folders under DATA_DIR that hold images. Used to recognise the
# data-relative tail of a legacy absolute path written on another machine.
KNOWN_ROOTS = ("face_training", "face_captures", "face_enroll", "reid_snapshots", "alert_snapshots", "face_pending")


def _data_dir() -> Path:
    return Path(config.DATA_DIR).resolve()


def _parts(stored: str) -> tuple[str, ...]:
    # Accept either separator regardless of the OS that wrote the value.
    return PureWindowsPath(stored).parts if ("\\" in stored or ":" in stored[:3]) else PurePosixPath(stored).parts


def to_stored(path: str | Path | None) -> str | None:
    """Value to put in the DB for an on-disk file. Relative to DATA_DIR when
    the file is inside it (the normal case); otherwise the absolute path,
    unchanged, so nothing is ever lost."""
    if path is None:
        return None
    p = Path(path).resolve()
    try:
        return p.relative_to(_data_dir()).as_posix()
    except ValueError:
        return str(p)


def _legacy_tail(stored: str) -> str | None:
    """'face_training/018/x.jpg' from any path containing .../data/face_training/018/x.jpg."""
    parts = _parts(stored)
    for i in range(len(parts) - 1):
        if parts[i].lower() == "data" and parts[i + 1] in KNOWN_ROOTS:
            return "/".join(parts[i + 1:])
    if parts and parts[0] in KNOWN_ROOTS:
        return "/".join(parts)
    return None


def resolve(stored: str | None) -> Path | None:
    """Absolute path for a DB value, or None if there is no such value.
    Does not check existence (callers decide what a missing file means),
    except to prefer an existing legacy absolute path as-is."""
    if not stored:
        return None
    data = _data_dir()
    is_abs = PureWindowsPath(stored).is_absolute() or stored.startswith("/")
    if not is_abs:
        tail = _legacy_tail(stored) or stored.replace("\\", "/")
        candidate = (data / tail).resolve()
        # A relative value must never escape DATA_DIR.
        try:
            candidate.relative_to(data)
        except ValueError:
            return None
        return candidate
    p = Path(stored)
    if p.exists():
        return p
    tail = _legacy_tail(stored)
    return (data / tail).resolve() if tail else p


def exists(stored: str | None) -> bool:
    p = resolve(stored)
    return bool(p and p.is_file())


def normalize(stored: str | None) -> str | None:
    """The relative form of a (possibly legacy) DB value, if its file lives
    under DATA_DIR; else the value unchanged. Used by the path migration."""
    if not stored:
        return stored
    p = resolve(stored)
    if p is None:
        return stored
    try:
        return p.resolve().relative_to(_data_dir()).as_posix()
    except ValueError:
        return stored
