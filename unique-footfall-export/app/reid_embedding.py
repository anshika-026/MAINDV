"""Body-appearance embedding for the unique-footfall Re-ID engine (reid_*.py)
— the one piece that's genuinely new rather than reused from the existing
FACE-based peopleid_* module (see config.py's "Body-appearance Re-ID unique
footfall engine" section for the full rationale).

Wraps torchreid's `FeatureExtractor` around a pretrained OSNet (small,
CPU-friendly — verified live this session on this machine's CPU: ~85ms for a
single crop, ~30ms/crop batched). One instance per camera worker process
(same lifetime/construction pattern as peopleid_tracker.PersonBodyTracker),
constructed once and reused for every tracked person in every frame.

REID_MODEL_PATH empty (the default) uses OSNet's auto-downloaded
ImageNet-pretrained backbone — a real pretrained CNN, but not fine-tuned on
a person-Re-ID task/dataset specifically. Setting REID_MODEL_PATH to a local
.pth from the deep-person-reid Model Zoo (Market1501/MSMT17-trained) swaps
in a materially more discriminative embedding with no code change — same
torchreid loading path either way.

IMPORTANT, confirmed live this session: with REID_MODEL_PATH unset (the
ImageNet-only fallback), this backbone's RAW pooled features are barely
discriminative at all for cosine similarity — 20 different synthetic
textures scored 0.90-0.96 similarity to EACH OTHER, which would force
-merge nearly every distinct person into one identity at any sane
threshold. This is a known property of classification-pretrained CNN
features (a large shared/"DC" component dominates cosine similarity,
since nothing during ImageNet training ever pushed different images'
features apart in COSINE terms specifically) — real Re-ID architectures
fix this with a metric-learning loss during training; lacking that here,
this module fixes it post-hoc with a standard image-retrieval technique:
subtract a fixed "average image" reference vector before normalizing.
Critically, that reference vector is computed ONCE at construction from
GENERIC synthetic calibration images (see _CALIBRATION_SEEDS below) —
deliberately NOT from the live gallery/enrolled people, which would create
a chicken-and-egg problem (confirmed live: centering derived from the
gallery is undefined with 0-1 enrolled people, and refusing to match
during that bootstrap window still doesn't help once a second person's
first observation also can't be centered yet). A fixed, gallery
-independent reference works immediately, from the very first person
enrolled. Measured live with this fixed calibration: same-image repeat
similarity 1.0, different-texture pairs 0.09-0.52 — a real gap, unlike the
uncentered 0.90-0.96 cluster. A REID_MODEL_PATH pointing at a real
Re-ID-task-trained checkpoint would not need this workaround (a properly
metric-learned embedding space is separated by construction), but this
keeps the zero-extra-download default usable in the meantime.
"""

from __future__ import annotations

import numpy as np

# Purely synthetic, generated once per process at model-load time — NOT
# real images, NOT gallery data. Their only job is to sample this backbone's
# feature space broadly enough that their mean approximates the same
# systematic bias any input (including a real person crop) carries, so
# subtracting it removes that shared component without needing any labeled
# or even real calibration data. Fixed seeds -> reproducible across runs.
_CALIBRATION_SEEDS = range(1000, 1032)
_CALIBRATION_CROP_SIZE = (256, 128)  # (h, w) — matches this model's own training input aspect ratio


def _clip_bbox(frame_bgr: np.ndarray, bbox: list[float]) -> tuple[int, int, int, int] | None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


class BodyReIdEmbedder:
    def __init__(self, model_name: str, model_path: str, device: str):
        import torch
        from torchreid.reid.utils import FeatureExtractor

        resolved_device = device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        # model_path='' makes torchreid fall back to its own auto-downloaded
        # ImageNet-pretrained checkpoint for model_name (verified live) —
        # only pass a real path when one's actually configured.
        self._extractor = FeatureExtractor(model_name=model_name, model_path=model_path or "", device=resolved_device)
        # A real Re-ID-task-trained checkpoint (model_path set) doesn't need
        # the calibration-mean workaround — its embedding space is already
        # separated by its own metric-learning training objective; computing
        # and subtracting a calibration mean in that case is unnecessary,
        # not harmful either way, but skipped to avoid the extra construction
        # cost when it buys nothing.
        self._calibration_mean = self._compute_calibration_mean() if not model_path else None

    def _compute_calibration_mean(self) -> np.ndarray:
        """The calibration set's own image statistics matter, not just its
        variety — confirmed live this session that literal per-pixel random
        noise (maximum-entropy, zero spatial correlation) produces a mean
        that does NOT generalize to real-ish crops: calibrating on pure
        noise left different-person similarity around 0.88-0.91, barely
        better than uncentered. A real person crop has smooth-ish regions
        (skin/fabric) and edges — spatially CORRELATED structure, not iid
        noise — so these calibration images use solid-colored blobs on a
        solid base color instead (same generator shape validated live:
        different-texture similarity 0.09-0.52 with this style, vs.
        0.88-0.91 with iid noise). Still no real photos involved."""
        import cv2

        h, w = _CALIBRATION_CROP_SIZE
        frames = []
        for seed in _CALIBRATION_SEEDS:
            rng = np.random.RandomState(seed)
            frame = np.full((h, w, 3), rng.randint(30, 220, size=3), dtype=np.uint8)
            for _ in range(8):
                cx, cy = rng.randint(0, w), rng.randint(0, h)
                r = rng.randint(h // 8, h // 3)
                color = rng.randint(0, 255, size=3).tolist()
                cv2.circle(frame, (cx, cy), r, color, -1)
            frames.append(frame)
        vectors = self._raw_embed_batch(frames)
        # Mean of each vector's OWN unit-normalized direction, not the mean
        # of raw (un-normalized) vectors — confirmed live this session these
        # give meaningfully different results: raw-feature averaging lets
        # larger-magnitude calibration samples dominate the mean
        # disproportionately, and the validated 0.09-0.52 separation result
        # was specifically produced by averaging unit vectors. embed_batch
        # below applies the identical unit-normalize-then-subtract recipe to
        # every query for consistency with how this mean was built.
        unit_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        return np.mean(unit_vectors, axis=0)

    def _raw_embed_batch(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        """Un-normalized, un-centered raw feature vectors straight from the
        model — the shared primitive both _compute_calibration_mean and
        embed_batch build on."""
        features = self._extractor(crops_rgb)
        return features.detach().cpu().numpy().astype(np.float32)

    def embed(self, frame_bgr: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        """L2-normalized appearance embedding for one person crop, or None
        for a degenerate crop (bbox clipped to nothing by the frame edge)."""
        result = self.embed_batch(frame_bgr, [bbox])
        return result[0]

    def embed_batch(self, frame_bgr: np.ndarray, bboxes: list[list[float]]) -> list[np.ndarray | None]:
        """Batched version — cheaper per-crop than calling embed() once per
        tracked person in the same frame (a single forward pass over a
        stacked batch instead of one per person)."""
        import cv2

        clipped = [_clip_bbox(frame_bgr, bbox) for bbox in bboxes]
        valid_indices = [i for i, c in enumerate(clipped) if c is not None]
        if not valid_indices:
            return [None] * len(bboxes)

        # torchreid's FeatureExtractor wraps a raw array straight into a PIL
        # Image with no channel-order conversion of its own — this
        # codebase's frames are BGR (OpenCV convention) throughout, so this
        # conversion is required for the ImageNet-trained backbone's
        # per-channel normalization to see genuine R/G/B, not swapped
        # channels (same reasoning as peopleid_reid.py's own cv2.cvtColor
        # call, that module's HSV case of the identical concern).
        crops_rgb = []
        for i in valid_indices:
            x1, y1, x2, y2 = clipped[i]
            crops_rgb.append(cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

        vectors = self._raw_embed_batch(crops_rgb)
        if self._calibration_mean is not None:
            # Unit-normalize EACH vector's own raw features first, THEN
            # subtract the calibration mean (itself a mean of unit vectors —
            # see _compute_calibration_mean) — subtracting from raw features
            # instead is a different, unvalidated statistic that measurably
            # under-performed in testing (different-person similarity
            # ~0.74-0.77 instead of the validated 0.09-0.52).
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
            vectors = (vectors / norms) - self._calibration_mean

        out: list[np.ndarray | None] = [None] * len(bboxes)
        for row, i in enumerate(valid_indices):
            vec = vectors[row]
            norm = np.linalg.norm(vec)
            out[i] = (vec / norm) if norm > 1e-8 else None
        return out
