"""
expression.py

Facial-expression (mood) classification, layered on top of the EXISTING
face pipeline rather than beside it.

Architecture, and why:
  - No camera loop of its own. It never opens an RTSP stream, never runs a
    person detector, and never runs a face detector. It is handed the face
    crop that face_pipeline._update_person_identity() has ALREADY computed
    for the ArcFace embedding, on the same frame, for the same track_id —
    so adding expression costs one extra classifier call, not a second
    perception pipeline (the standalone facial-expression project drives
    its own cv2.VideoCapture; that shape is deliberately not reused here).
  - Runs on its own single-worker executor with drop-if-busy semantics.
    Identity recognition and footfall therefore can NEVER be blocked or
    slowed by expression inference: if the model is still busy with the
    previous crop, this cycle's crop is simply skipped. Expression is the
    feature that degrades under load, never recognition.
  - Results are advisory metadata keyed by (camera_id, track_id). A failed,
    slow, or unavailable expression model leaves a person's NAME completely
    untouched — identity never depends on this module in any direction.

Model: a plain ViT image classifier via Hugging Face `transformers`, the
same approach as the footfall-mood build (EMOTION_MODEL_NAME below).
Deliberately NOT the standalone branch's Keras model, which would have
required adding TensorFlow alongside the working onnxruntime/torch stack.

Everything here is optional at runtime: if `transformers` is missing or
the model can't be fetched, this degrades to "no expression" and the rest
of the application is unaffected.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger("expression")

# Swappable for a locally fine-tuned checkpoint directory with no code
# change, same as the footfall-mood build's EMOTION_MODEL_NAME.
EMOTION_MODEL_NAME = os.environ.get(
    "EMOTION_MODEL_NAME", "HardlyHumans/Facial-expression-detection"
)
EXPRESSION_ENABLED = os.environ.get("EXPRESSION_ENABLED", "true").strip().lower() not in ("0", "false", "no")

# Below this the expression is reported as None rather than guessed — an
# uncertain mood is worth less than no mood, and this value never affects
# identity either way.
EXPRESSION_MIN_CONFIDENCE = float(os.environ.get("EXPRESSION_MIN_CONFIDENCE", "0.45"))

# How long a result stays attached to a track after it was produced. Longer
# than the classify cadence so a face that briefly turns away keeps its last
# known mood instead of flickering to nothing.
EXPRESSION_TTL_SECONDS = float(os.environ.get("EXPRESSION_TTL_SECONDS", "20"))


class _ExpressionService:
    """Lazily loads the classifier on first use and runs it off-thread.

    Nothing is imported or downloaded at module import time — a backend
    that never sees a face never pays for this at all, and an environment
    without `transformers` still starts normally.
    """

    def __init__(self) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="expression"
        )
        self._future: concurrent.futures.Future | None = None
        self._lock = threading.Lock()
        self._results: dict[tuple[int, int], tuple[str, float, float]] = {}
        self._model = None
        self._processor = None
        self._torch = None
        self._load_failed = False

    # -- model ------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed or not EXPRESSION_ENABLED:
            return False
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            try:
                processor = AutoImageProcessor.from_pretrained(EMOTION_MODEL_NAME)
            except Exception:
                # Older checkpoints declare a preprocessor class name that
                # current transformers no longer exposes via Auto*; the ViT
                # processor reads the same preprocessor_config.json fine.
                from transformers import ViTImageProcessor

                processor = ViTImageProcessor.from_pretrained(EMOTION_MODEL_NAME)

            model = AutoModelForImageClassification.from_pretrained(EMOTION_MODEL_NAME)
            model.eval()
            self._torch, self._processor, self._model = torch, processor, model
            log.info("expression model loaded: %s", EMOTION_MODEL_NAME)
            return True
        except Exception:
            # Missing dependency, no network on first fetch, bad checkpoint —
            # all the same outcome: this feature stays off and everything
            # else keeps working.
            self._load_failed = True
            log.exception("expression model unavailable — continuing without expression")
            return False

    # -- inference --------------------------------------------------------

    def submit(self, camera_id: int, track_id: int, crop_bgr: np.ndarray) -> None:
        """Fire-and-forget. Returns immediately; drops this crop if the
        previous inference is still running, so the caller's thread (the
        face pipeline) is never made to wait on the expression model."""
        if not EXPRESSION_ENABLED or self._load_failed:
            return
        if self._future is not None and not self._future.done():
            return  # still busy — skip, never queue up behind it
        if crop_bgr is None or crop_bgr.size == 0:
            return
        crop = crop_bgr.copy()  # the caller reuses its frame buffer
        self._future = self._executor.submit(self._run, camera_id, track_id, crop)

    def _run(self, camera_id: int, track_id: int, crop_bgr: np.ndarray) -> None:
        try:
            if not self._ensure_loaded():
                return
            # cv2.cvtColor, not crop[:, :, ::-1]: the slice form produces a
            # negative-stride view that torch.from_numpy rejects outright
            # ("tensors with negative strides are not currently supported").
            import cv2

            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            inputs = self._processor(images=rgb, return_tensors="pt")
            with self._torch.no_grad():
                logits = self._model(**inputs).logits
            probs = self._torch.nn.functional.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax())
            score = float(probs[idx])
            label = str(self._model.config.id2label[idx])
            if score < EXPRESSION_MIN_CONFIDENCE:
                return  # not confident enough to claim a mood
            with self._lock:
                self._results[(camera_id, track_id)] = (label, score, time.time())
        except Exception:
            # Never allowed to propagate into the face pipeline.
            log.exception("expression inference failed for track %s", track_id)

    # -- readback ---------------------------------------------------------

    def get(self, camera_id: int, track_id: int) -> tuple[str | None, float]:
        """Last known expression for a track, or (None, 0.0). Expired
        results are treated as absent so a stale mood is never shown."""
        with self._lock:
            hit = self._results.get((camera_id, track_id))
        if hit is None:
            return None, 0.0
        label, score, at = hit
        if time.time() - at > EXPRESSION_TTL_SECONDS:
            return None, 0.0
        return label, score

    def forget(self, camera_id: int, track_id: int) -> None:
        """Called when a person track ages out, so results don't accumulate
        for tracks that no longer exist."""
        with self._lock:
            self._results.pop((camera_id, track_id), None)


service = _ExpressionService()
