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
from datetime import datetime

import joblib
import numpy as np
import threadpoolctl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from app import face_db
from app.face_pipeline import CLASSIFIER_PATH, TRAINING_CAPTURE_DIR

MIN_SAMPLES_PER_CLASS = 2  # a classifier can't learn a class boundary from a single example

# CPU protection: measured on this box, LogisticRegression.fit() pulls in
# OpenBLAS + OpenMP thread pools that default to one thread PER LOGICAL
# CORE (8 here) — during the few seconds a fit takes, that's real
# contention against the live camera pipeline's own threads (RTSP read,
# YOLO inference, ArcFace embedding), which run in this same process. This
# is a background/scheduled operation (see face_training_scheduler.py),
# never inside the live frame loop, but still shares the process, so
# capping thread COUNT (not OS scheduling priority — Windows has no simple
# per-thread-priority hook from pure Python without pywin32, which this
# project doesn't otherwise depend on) is the concrete, low-risk lever
# available here. Leaves most cores free for live inference during a fit.
TRAINING_MAX_THREADS = int(os.environ.get("FACE_TRAINING_MAX_THREADS", "2"))

# Anti-domination cap: at most this many samples from a single (person,
# camera, track) group — one continuous tracked appearance — actually
# contribute to a training run. Real captures are already thinned a lot at
# CAPTURE time (15-minute per-camera dedup cooldown in face_pipeline.py),
# but nothing previously stopped one unusually long/frequent session (or
# one camera that simply sees an employee far more often than others) from
# quietly dominating that employee's embedding distribution — the model
# would end up biased toward that one appearance's pose/lighting/angle
# rather than the person in general. Deterministic, evenly-spaced
# subsampling (not random) so the kept samples still span the group's full
# time range rather than clustering at one end.
MAX_SAMPLES_PER_TRACK_GROUP = int(os.environ.get("FACE_MAX_SAMPLES_PER_TRACK_GROUP", "15"))


def _cap_group_domination(rows: list[dict]) -> list[dict]:
    """Same (camera_id, track_id) grouping _group_key/_build_split already
    use for validation, applied here to the training input itself —
    scoped per person_id too, since the same camera/track_id pair can't
    collide across different people anyway but this keeps the intent
    explicit. A row with no track_id is its own singleton group (via
    _group_key's "capture" fallback) and is never capped."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["person_id"], _group_key(r))].append(r)

    kept: list[dict] = []
    for _key, group_rows in groups.items():
        if len(group_rows) <= MAX_SAMPLES_PER_TRACK_GROUP:
            kept.extend(group_rows)
            continue
        group_rows.sort(key=lambda r: r["capture_id"])
        step = len(group_rows) / MAX_SAMPLES_PER_TRACK_GROUP
        indices = {int(i * step) for i in range(MAX_SAMPLES_PER_TRACK_GROUP)}
        kept.extend(group_rows[i] for i in sorted(indices))
    return kept

# Below these, a computed "validation accuracy" would be a coin flip dressed
# up as a metric — report its absence honestly instead (see module docstring
# and step 14 of the spec this file implements).
VALIDATION_MIN_SAMPLES = 10
VALIDATION_MIN_CLASSES = 2

# Per-class flags on top of the overall validation number (see
# _per_class_validation_report below) — deliberately separate thresholds
# from VALIDATION_MIN_SAMPLES/CLASSES above, which gate whether an overall
# number is reported at all; these gate which INDIVIDUAL employees get
# flagged once it is.
PER_CLASS_MIN_VAL_SAMPLES = 3    # below this, a per-class accuracy is too noisy to call "weak" or "strong"
PER_CLASS_WEAK_ACCURACY = 0.7    # below this (with enough samples), flag the employee explicitly


def _per_class_validation_report(y_val: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-employee accuracy/false-accept/false-reject on the SAME held-out
    split already used for the overall validation_result — never trained
    on. False rejection here means "this employee's own face, but the
    classifier picked someone else or nothing usable" (recall miss); false
    acceptance means "some OTHER employee's face was predicted as this one"
    — both counted straight from the confusion matrix, not estimated.
    Confusion matrix included as a labeled dict-of-dicts (true -> predicted
    -> count), which stays readable at the employee counts this system
    actually has, unlike a raw sklearn array with no labels attached."""
    labels = sorted(set(y_val) | set(y_pred))
    cm = confusion_matrix(y_val, y_pred, labels=labels)

    per_class = {}
    for i, label in enumerate(labels):
        actual_count = int(cm[i, :].sum())
        if actual_count == 0:
            continue  # this label only ever appears as a wrong prediction, never as ground truth in this split
        correct = int(cm[i, i])
        false_rejections = actual_count - correct
        false_acceptances = int(cm[:, i].sum()) - correct  # others wrongly predicted AS this label
        acc = correct / actual_count
        per_class[label] = {
            "val_samples": actual_count,
            "correct": correct,
            "accuracy": acc,
            "false_rejections": false_rejections,
            "false_acceptances": false_acceptances,
            "insufficient_samples": actual_count < PER_CLASS_MIN_VAL_SAMPLES,
            "weak": actual_count >= PER_CLASS_MIN_VAL_SAMPLES and acc < PER_CLASS_WEAK_ACCURACY,
        }

    return {
        "per_class": per_class,
        "confusion_matrix": {
            true_label: {str(labels[j]): int(cm[i, j]) for j in range(len(labels)) if cm[i, j] > 0}
            for i, true_label in enumerate(labels)
        },
        "weak_classes": sorted(l for l, v in per_class.items() if v["weak"]),
        "insufficient_sample_classes": sorted(l for l, v in per_class.items() if v["insufficient_samples"]),
    }


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
    # See TRAINING_MAX_THREADS above — caps OpenBLAS/OpenMP to a few
    # threads for just this fit call, leaving most cores free for the live
    # camera pipeline running in this same process. Restored automatically
    # on context exit regardless of how fit() returns.
    with threadpoolctl.threadpool_limits(limits=TRAINING_MAX_THREADS):
        clf.fit(X, y)
    return clf


def train_classifier() -> dict:
    # Captured BEFORE this run's own add_training_run() call below creates
    # a new row, so "since previous training" means exactly that — the
    # prior run, not this one.
    previous_runs = face_db.list_training_runs(limit=1)
    previous_trained_at = previous_runs[0]["trained_at"] if previous_runs else 0.0
    samples_added_since_previous = face_db.count_labeled_since(previous_trained_at)
    start_of_today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    samples_added_today = face_db.count_captures_since(start_of_today)

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

    # Anti-domination cap (see MAX_SAMPLES_PER_TRACK_GROUP) — applied AFTER
    # the raw label counts above (so per_employee_counts still reports true
    # "how much labeled data exists", not a capped figure), but BEFORE the
    # train/validation split and the actual fit, so neither the reported
    # validation number nor the deployed model is skewed by one dominant
    # tracking session.
    usable = _cap_group_domination(usable)
    capped_counts = Counter(r["person_id"] for r in usable)

    train_rows, val_rows, split_info = _build_split(usable)

    validation_result = None
    per_class_validation = None
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
            per_class_validation = _per_class_validation_report(y_val, y_pred)
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

    per_employee_counts = dict(sorted(counts.items()))
    per_employee_training_counts = dict(sorted(capped_counts.items()))
    capped_samples = {
        pid: n - capped_counts.get(pid, 0)
        for pid, n in counts.items()
        if capped_counts.get(pid, 0) < n
    }
    run_id = face_db.add_training_run(
        sample_count=len(usable),
        class_count=len(usable_classes),
        excluded_no_embedding=all_by_status.get("no_embedding", 0),
        excluded_rejected=all_by_status.get("rejected", 0),
        excluded_too_few_samples=sum(too_few.values()),
        model_path=CLASSIFIER_PATH,
        per_employee_counts=per_employee_counts,
        validation_samples=validation_result["val_samples"] if validation_result else None,
        validation_classes=validation_result["val_classes"] if validation_result else None,
        validation_accuracy=validation_result["accuracy"] if validation_result else None,
        per_class_validation=per_class_validation,
    )

    return {
        "run_id": run_id,
        "trained_on_samples": len(usable),
        "employees": sorted(usable_classes),
        "per_employee_counts": per_employee_counts,
        # What actually went into this run's fit, after the anti-domination
        # cap (MAX_SAMPLES_PER_TRACK_GROUP) — vs per_employee_counts above,
        # which is the raw total labeled count regardless of that cap.
        "per_employee_training_counts": per_employee_training_counts,
        "capped_samples": capped_samples,
        "samples_added_since_previous_training": samples_added_since_previous,
        "samples_added_today": samples_added_today,
        "excluded_too_few_samples": too_few,
        "excluded_no_embedding": all_by_status.get("no_embedding", 0),
        "excluded_rejected": all_by_status.get("rejected", 0),
        "model_path": CLASSIFIER_PATH,
        "backup_path": backup_path,
        "validation": validation_result,
        "validation_skipped_reason": split_info["reason"] if validation_result is None else None,
        # Per-employee accuracy/false-accept/false-reject + confusion matrix
        # on the same held-out split as `validation` above — None whenever
        # `validation` itself is None (nothing to break down per-class).
        "per_class_validation": per_class_validation,
    }
