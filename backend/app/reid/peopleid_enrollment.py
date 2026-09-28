"""Automatic reference-image curation for the People Identification
enrollment workflow (spec: an operator confirms "this is Anshika" once —
the system picks the 10-20 actual reference embeddings itself, not a
manually-labeled photo dump).

Pure aggregation logic only — no model inference here. Per-image detection
+ quality assessment + embedding happens in the detection worker process
(see the new "peopleid_embed" request type added to detection_worker.py,
mirroring the existing "embed" request the old People page already uses —
same synchronous embed_request_queue/embed_response_queue plumbing, richer
response). This module just takes the resulting per-image samples (however
many were captured — could be dozens to thousands of frames) and picks a
small, quality-gated, pose-diverse final set:

  raw samples -> quality filter (already applied per-sample by the worker)
              -> embedding-similarity dedup (drop near-identical repeats)
              -> pose-diversity bucket selection
              -> top PEOPLEID_ENROLLMENT_TARGET_EMBEDDINGS overall
"""

from __future__ import annotations

import numpy as np

from . import config


def curate(samples: list[dict], target_count: int | None = None) -> list[dict]:
    """samples: list of {"embedding": np.ndarray, "quality_score": float,
    "quality_passed": bool, "pose_label": str, "crop_jpeg": bytes | None}.
    Returns the selected subset (same dict shape), highest-value first."""
    target_count = target_count or config.PEOPLEID_ENROLLMENT_TARGET_EMBEDDINGS
    passed = [s for s in samples if s.get("quality_passed")]
    if not passed:
        return []

    passed.sort(key=lambda s: s["quality_score"], reverse=True)
    deduped: list[dict] = []
    for sample in passed:
        emb = sample["embedding"]
        emb_n = emb / (np.linalg.norm(emb) + 1e-8)
        is_duplicate = False
        for kept in deduped:
            kept_n = kept["embedding"] / (np.linalg.norm(kept["embedding"]) + 1e-8)
            if float(np.dot(emb_n, kept_n)) > config.PEOPLEID_DEDUP_SIMILARITY:
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(sample)

    buckets: dict[str, list[dict]] = {}
    for sample in deduped:
        buckets.setdefault(sample.get("pose_label", "unknown"), []).append(sample)
    for bucket in buckets.values():
        bucket.sort(key=lambda s: s["quality_score"], reverse=True)

    # Round-robin across pose buckets so the final set favors DIVERSITY
    # (front/left/right/up/down) over simply the N highest-quality frames,
    # which would likely all be near-duplicate frontal shots from the same
    # few seconds of good lighting.
    selected: list[dict] = []
    bucket_labels = list(buckets.keys())
    while len(selected) < target_count and any(buckets[label] for label in bucket_labels):
        progressed = False
        for label in bucket_labels:
            if buckets[label]:
                selected.append(buckets[label].pop(0))
                progressed = True
                if len(selected) >= target_count:
                    break
        if not progressed:
            break
    return selected
