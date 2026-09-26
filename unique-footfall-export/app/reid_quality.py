"""Body-crop quality gate for the unique-footfall Re-ID engine — same role
as peopleid_quality.py's face-quality gate ("a bad frame should not corrupt
an identity"), adapted for a full-body crop instead of a face crop:

- No 5-point facial landmarks exist for a body box, so there's no yaw/pitch
  pose estimate here (peopleid_quality._estimate_pose has no body-crop
  equivalent). Occlusion/pose is proxied instead by bbox aspect ratio — a
  standing, mostly-unoccluded person's box is reliably taller than it is
  wide (COCO/person-detection convention); a box that's nearly square or
  wider-than-tall usually means partial occlusion, a person bent over, or
  two people merged into one detection, all of which make for a poor
  reference crop.
- Otherwise the same composite-score pattern: size, Laplacian blur
  variance, brightness, each soft-normalized against a configurable floor
  and combined into one weighted score a track's observation must clear.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


class BodyQualityResult:
    __slots__ = ("passed", "composite_score", "size_score", "blur_score", "brightness_score", "aspect_score", "reject_reason")

    def __init__(self):
        self.passed = False
        self.composite_score = 0.0
        self.size_score = 0.0
        self.blur_score = 0.0
        self.brightness_score = 0.0
        self.aspect_score = 0.0
        self.reject_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "composite_score": round(self.composite_score, 3),
            "size_score": round(self.size_score, 3),
            "blur_score": round(self.blur_score, 3),
            "brightness_score": round(self.brightness_score, 3),
            "aspect_score": round(self.aspect_score, 3),
            "reject_reason": self.reject_reason,
        }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def assess(frame_bgr: np.ndarray, bbox: list[float], det_score: float, quality_min_score: float | None = None) -> BodyQualityResult:
    """quality_min_score overrides config.REID_QUALITY_MIN_SCORE — lets a
    per-camera calibration tighten/relax the pass bar, mirroring
    peopleid_quality.assess's same override parameter."""
    result = BodyQualityResult()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        result.reject_reason = "empty_crop"
        return result

    if min(box_w, box_h) < config.REID_MIN_BODY_SIZE_PX:
        result.reject_reason = "body_too_small"
        result.size_score = _clamp01(min(box_w, box_h) / config.REID_MIN_BODY_SIZE_PX)
        return result
    result.size_score = _clamp01(min(box_w, box_h) / (config.REID_MIN_BODY_SIZE_PX * 2))

    # A standing, unoccluded person's box height:width ratio is reliably
    # >1.0 (typically 1.8-3.5 depending on framing/crop tightness); a ratio
    # below ~1.0 usually means significant occlusion, a bent-over pose, or
    # two people merged into one detection box — score falls off smoothly
    # rather than hard-rejecting a legitimately close/cropped frame.
    aspect_ratio = box_h / box_w
    result.aspect_score = _clamp01((aspect_ratio - 0.6) / 1.4)

    crop = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    result.blur_score = _clamp01(blur_var / config.REID_BLUR_VARIANCE_FLOOR)
    if blur_var < config.REID_BLUR_VARIANCE_HARD_MIN:
        result.reject_reason = "too_blurry"
        return result

    mean_brightness = float(gray.mean())
    result.brightness_score = _clamp01(1.0 - abs(mean_brightness - 128.0) / 110.0)
    if mean_brightness < 15.0 or mean_brightness > 245.0:
        result.reject_reason = "bad_lighting"
        return result

    result.composite_score = (
        0.25 * result.size_score
        + 0.30 * result.blur_score
        + 0.15 * result.brightness_score
        + 0.15 * result.aspect_score
        + 0.15 * det_score
    )
    min_score = quality_min_score if quality_min_score is not None else config.REID_QUALITY_MIN_SCORE
    result.passed = result.composite_score >= min_score
    if not result.passed:
        result.reject_reason = "low_composite_quality"
    return result
