"""Lightweight person Re-ID (appearance-based continuity) for the People
Identification module — used ONLY to bridge an already-face-CONFIRMED
track through a stretch where the face is temporarily unavailable (turned
away, occluded, too small/blurry to pass the quality gate). It must never
independently establish an identity: an HSV color-histogram signature has a
real false-accept rate against two people wearing similar clothing, which
is fine for "is this still probably the same track" but not for "who is
this" — see peopleid_fusion.py, the only caller, which enforces that
separation structurally rather than trusting this module's callers to.

A CNN-based re-id embedding (e.g. a small OSNet ONNX model) would be more
discriminative but costs a real model-load + inference on an already
CPU-constrained box; this HSV-histogram approach costs microseconds and is
the pragmatic default until benchmarking shows it's not good enough —
swappable behind this same signature/similarity pair if so.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_signature(frame_bgr: np.ndarray, person_bbox: list[float]) -> np.ndarray | None:
    """A concatenated H/S histogram (36 + 32 bins) over the person crop,
    L1-normalized — cheap, rotation/scale-invariant-ish, good enough to
    distinguish "same person, clothes unchanged, few seconds later" from "a
    different person," not much more than that."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in person_bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [36], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    sig = np.concatenate([hist_h.flatten(), hist_s.flatten()])
    total = sig.sum()
    if total <= 0:
        return None
    return (sig / total).astype(np.float32)


def similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Histogram intersection (sum of per-bin minimums) — bounded [0, 1]
    since both signatures are L1-normalized, cheaper and more robust to
    outlier bins than cosine similarity for histogram-shaped data."""
    return float(np.minimum(sig_a, sig_b).sum())
