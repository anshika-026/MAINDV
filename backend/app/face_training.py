"""
face_training.py

The explicit, separate training operation — never runs automatically, only
via POST /api/faces/training/train. Implements Option B (see
FACE_TRAINING.md for the full reasoning): ArcFace stays frozen as the
feature extractor; this trains a supervised classifier on top of its
embeddings using real, human-labeled camera captures
(face_training_captures where label_status = 'labeled').

Deliberately not in face_pipeline.py: that file is the real-time inference
path; this is an offline batch job that happens to write a file
face_pipeline.py's _match() then picks up automatically (see
CameraFacePipeline._get_classifier).
"""

import os
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from app import face_db
from app.face_pipeline import CLASSIFIER_PATH, TRAINING_CAPTURE_DIR

MIN_SAMPLES_PER_CLASS = 2  # a classifier can't learn a class boundary from a single example


def train_classifier() -> dict:
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

    X = np.array([r["embedding"] for r in usable], dtype=np.float32)
    y = np.array([r["person_id"] for r in usable])

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)

    os.makedirs(TRAINING_CAPTURE_DIR, exist_ok=True)
    joblib.dump(clf, CLASSIFIER_PATH)

    return {
        "trained_on_samples": len(usable),
        "employees": sorted(usable_classes),
        "excluded_too_few_samples": too_few,
        "model_path": CLASSIFIER_PATH,
    }
