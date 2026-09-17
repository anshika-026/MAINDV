"""
face_training.py

The explicit, separate training operation — never runs automatically, only
via POST /api/faces/training/train (browser) or `python -m app.train_faces`
(terminal — see that module). Implements Option B (see FACE_TRAINING.md for
the full reasoning): ArcFace stays frozen as the feature extractor; this
trains a supervised classifier on top of its embeddings using real,
human-labeled camera captures (face_training_captures where
label_status = 'labeled').

Deliberately not in face_pipeline.py: that file is the real-time inference
path; this is an offline batch job that happens to write a file
face_pipeline.py's _match() then picks up automatically (see
CameraFacePipeline._get_classifier).

Validation split: captures are grouped by (camera_id, track_id) before
splitting — several captures from the same finished track are near-duplicate
crops of one appearance, so randomly shuffling individual samples between
train/validation would let the model "recognize" a specific pose/frame it
already saw rather than the person, inflating the reported accuracy. A
group only ends up in the held-out validation set if that leaves the class
still meeting MIN_SAMPLES_PER_CLASS in the training portion; classes that
can't be safely split are trained on but simply don't contribute to the
validation number. If too little data ends up validated, this reports that
plainly instead of inventing a percentage — see VALIDATION_MIN_SAMPLES /
VALIDATION_MIN_CLASSES below.
"""

import os
import shutil
import time
from collections import Counter, defaultdict

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from app import face_db
from app.face_pipeline import CLASSIFIER_PATH, TRAINING_CAPTURE_DIR

MIN_SAMPLES_PER_CLASS = 2  # a classifier can't learn a class boundary from a single example

# Below these, a computed "validation accuracy" would be a coin flip dressed
# up as a metric — report its absence honestly instead (see module docstring
# and step 14 of the spec this file implements).
VALIDATION_MIN_SAMPLES = 10
VALIDATION_MIN_CLASSES = 2


def _group_key(row: dict) -> tuple:
    # Rows can predate track_id being recorded, or have it null — fall back
    # to a per-capture key (no grouping, i.e. can't be safely split into
    # validation on its own) rather than crashing or silently merging
    # unrelated captures under one None key.
    if row["track_id"] is not None:
        return (row["camera_id"], row["track_id"])
    return ("capture", row["capture_id"])


def _build_split(usable: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Returns (train_rows, val_rows, split_info). val_rows may be empty —
    callers must check split_info['validation_possible'] before trusting any
    accuracy computed from it."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in usable:
        by_class[r["person_id"]].append(r)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    classes_with_val = 0

    for person_id, rows in by_class.items():
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            groups[_group_key(r)].append(r)
        group_keys = sorted(groups.keys(), key=lambda k: str(k))

        if len(group_keys) < 2:
            # Every sample for this person came from one track (or has no
            # track_id at all) — holding any of it out would either leak
            # (same track on both sides) or leave zero training data for
            # this class. Train on all of it, contribute nothing to val.
            train_rows.extend(rows)
            continue

        # Hold out ~25% of this person's distinct tracking sequences
        # (at least 1), but never so many that training drops below
        # MIN_SAMPLES_PER_CLASS for this class.
        n_hold_target = max(1, round(len(group_keys) * 0.25))
        hold_keys: list[tuple] = []
        for key in group_keys:
            if len(hold_keys) >= n_hold_target:
                break
            remaining_train = sum(
                len(groups[k]) for k in group_keys if k != key and k not in hold_keys
            )
            if remaining_train < MIN_SAMPLES_PER_CLASS:
                break  # would starve training for this class — stop holding out more
            hold_keys.append(key)

        for key in group_keys:
            (val_rows if key in hold_keys else train_rows).extend(groups[key])
        if hold_keys:
            classes_with_val += 1

    split_info = {
        "validation_possible": (
            len(val_rows) >= VALIDATION_MIN_SAMPLES and classes_with_val >= VALIDATION_MIN_CLASSES
        ),
        "val_samples": len(val_rows),
        "val_classes": classes_with_val,
        "reason": None,
    }
    if not split_info["validation_possible"]:
        split_info["reason"] = (
            f"only {len(val_rows)} held-out samples across {classes_with_val} employees "
            f"(need >= {VALIDATION_MIN_SAMPLES} samples across >= {VALIDATION_MIN_CLASSES} employees "
            f"with multiple distinct tracked appearances each) — training completed, but no reliable "
            f"validation accuracy was calculated."
        )
    return train_rows, val_rows, split_info


def _fit(rows: list[dict]) -> LogisticRegression:
    X = np.array([r["embedding"] for r in rows], dtype=np.float32)
    y = np.array([r["person_id"] for r in rows])
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)
    return clf


def train_classifier() -> dict:
    all_by_status = face_db.get_capture_counts_by_status()
    rows = face_db.get_labeled_training_embeddings()
    if not rows:
        raise ValueError("No labeled training captures yet — label some in the /face-training UI first.")

    counts = Counter(r["person_id"] for r in rows)
    too_few = {pid: n for pid, n in counts.items() if n < MIN_SAMPLES_PER_CLASS}
    usable = [r for r in rows if counts[r["person_id"]] >= MIN_SAMPLES_PER_CLASS]
    usable_classes = {r["person_id"] for r in usable}

    if len(usable_classes) < 2:
        raise ValueError(
            f"Need at least 2 employees with >= {MIN_SAMPLES_PER_CLASS} labeled captures each to train a "
            f"classifier. Current label counts: {dict(counts)}"
        )

    train_rows, val_rows, split_info = _build_split(usable)

    validation_result = None
    if split_info["validation_possible"]:
        train_classes = {r["person_id"] for r in train_rows}
        val_classes = {r["person_id"] for r in val_rows}
        if train_classes.issuperset(val_classes) and len(train_classes) >= 2:
            probe_clf = _fit(train_rows)
            X_val = np.array([r["embedding"] for r in val_rows], dtype=np.float32)
            y_val = np.array([r["person_id"] for r in val_rows])
            y_pred = probe_clf.predict(X_val)
            acc = float(accuracy_score(y_val, y_pred))
            validation_result = {
                "accuracy": acc,
                "val_samples": len(val_rows),
                "val_classes": len(val_classes),
                "measured_on_held_out_data": True,
            }
        else:
            split_info["reason"] = (
                "held-out set referenced an employee with no remaining training samples after the split — "
                "training completed, but no reliable validation accuracy was calculated."
            )

    # Final, deployed model is trained on ALL usable labeled data (train +
    # validation rows combined) — the split above exists only to measure
    # generalization, not to withhold real labeled data from the model that
    # actually ships.
    final_clf = _fit(usable)

    os.makedirs(TRAINING_CAPTURE_DIR, exist_ok=True)
    backup_path = None
    if os.path.exists(CLASSIFIER_PATH):
        backup_path = CLASSIFIER_PATH + f".bak-{int(time.time())}"
        shutil.copy2(CLASSIFIER_PATH, backup_path)

    tmp_path = CLASSIFIER_PATH + ".tmp"
    try:
        joblib.dump(final_clf, tmp_path)
        os.replace(tmp_path, CLASSIFIER_PATH)  # atomic on both POSIX and Windows (NTFS)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return {
        "trained_on_samples": len(usable),
        "employees": sorted(usable_classes),
        "excluded_too_few_samples": too_few,
        "excluded_no_embedding": all_by_status.get("no_embedding", 0),
        "excluded_rejected": all_by_status.get("rejected", 0),
        "model_path": CLASSIFIER_PATH,
        "backup_path": backup_path,
        "validation": validation_result,
        "validation_skipped_reason": split_info["reason"] if validation_result is None else None,
    }
