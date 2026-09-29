"""
classifier_io.py

Saving and loading the trained face classifier (a pickled scikit-learn
model) safely across library versions.

A classifier pickled by one scikit-learn version can load under another and
then crash on every prediction (seen in practice: trained with 1.9.1, loaded
under 1.5.2 -> AttributeError 'multi_class' inside predict_proba, which used
to switch face recognition off). So:

  - save() writes classifier.meta.json next to the model with the library
    versions and feature count it was trained with, and rotates backups;
  - load_verified() loads, compares versions, and runs one real
    predict_proba on a dummy input. If anything fails, it returns None with
    a clear log line (face recognition falls back to gallery matching) and
    the caller does not retry until the file changes (i.e. it is retrained).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger("classifier")


def meta_path(model_path: str | Path) -> Path:
    return Path(str(model_path)).with_suffix(".meta.json")


def _versions() -> dict:
    import joblib
    import sklearn

    return {"sklearn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__}


def rotate_backups(model_path: str | Path, keep: int | None = None) -> list[str]:
    """Keeps the newest `keep` classifier.joblib.bak-* files; returns removed paths.
    Never touches the active model."""
    keep = config.MODEL_BACKUP_RETENTION_COUNT if keep is None else keep
    p = Path(str(model_path))
    backups = sorted(p.parent.glob(p.name + ".bak-*"), key=lambda b: b.stat().st_mtime, reverse=True)
    removed = []
    for b in backups[max(0, keep):]:
        try:
            b.unlink()
            removed.append(str(b))
        except OSError:
            log.warning("could not remove old classifier backup %s", b.name)
    if removed:
        log.info("removed %d old classifier backup(s), kept %d", len(removed), min(keep, len(backups)))
    return removed


def save(clf, model_path: str | Path) -> str | None:
    """Atomic save with metadata; the previous model (if any) is kept as a
    timestamped backup. Returns the backup path."""
    import joblib

    model_path = str(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    backup_path = None
    if os.path.exists(model_path):
        backup_path = model_path + f".bak-{int(time.time() * 1000)}"  # ms: two runs in one second must not collide
        shutil.copy2(model_path, backup_path)
    tmp = model_path + ".tmp"
    try:
        joblib.dump(clf, tmp)
        os.replace(tmp, model_path)  # atomic on POSIX and NTFS
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    meta = {**_versions(), "trained_at": time.time(), "n_features": int(getattr(clf, "n_features_in_", 0)),
            "classes": [str(c) for c in getattr(clf, "classes_", [])]}
    tmp_meta = str(meta_path(model_path)) + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp_meta, meta_path(model_path))
    rotate_backups(model_path)
    return backup_path


def read_meta(model_path: str | Path) -> dict | None:
    try:
        with open(meta_path(model_path), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# Last load outcome, for /ready and the model-status endpoint.
last_status: dict = {"loaded": False, "reason": "not loaded yet"}


def load_verified(model_path: str | Path):
    """The classifier, or None if it is missing or unusable in this
    environment (reason in last_status)."""
    import joblib

    global last_status
    model_path = str(model_path)
    if not os.path.exists(model_path):
        last_status = {"loaded": False, "reason": "no trained classifier"}
        return None
    meta = read_meta(model_path)
    current = _versions()
    if meta and meta.get("sklearn") != current["sklearn"]:
        log.warning("classifier was trained with scikit-learn %s but %s is installed; verifying it still works "
                    "(retrain from the Face Training page to remove this warning)", meta.get("sklearn"), current["sklearn"])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # InconsistentVersionWarning: handled explicitly above/below
            clf = joblib.load(model_path)
        n = int(getattr(clf, "n_features_in_", 0) or (meta or {}).get("n_features") or 512)
        proba = clf.predict_proba(np.zeros((1, n), dtype=np.float32))
        if proba.shape != (1, len(clf.classes_)) or not np.all(np.isfinite(proba)):
            raise ValueError("predict_proba returned an unexpected result")
    except Exception as e:
        last_status = {"loaded": False, "reason": f"unusable with scikit-learn {current['sklearn']}: {type(e).__name__}: {e}"[:300],
                       "trained_with": (meta or {}).get("sklearn")}
        log.error("face classifier at %s cannot be used (%s). Face recognition falls back to gallery matching "
                  "until it is retrained (Face Training page, or python -m app.train_faces).",
                  os.path.basename(model_path), last_status["reason"])
        return None
    last_status = {"loaded": True, "reason": None, "trained_with": (meta or {}).get("sklearn"),
                   "running_with": current["sklearn"], "classes": len(clf.classes_)}
    log.info("face classifier loaded (%d classes, scikit-learn %s)", len(clf.classes_), current["sklearn"])
    return clf
