"""
appearance.py

Identity hand-off by body appearance. Face recognition only works when a face
is visible, and most desk cameras see people from behind. So whenever face
recognition confidently names someone (the live overlay on any camera, or
Staff Count at an entrance), their body appearance (the Re-ID embedding the
footfall engine already uses) is kept for the day. Elsewhere, a person with
no visible face can then be named by matching their body against ONLY the
people face-recognised today: a small set, which is what makes appearance
matching workable.

Deliberately conservative, because appearance is a weaker signal than a face:
  - a match needs similarity >= APPEARANCE_MATCH_THRESHOLD and a clear lead
    (APPEARANCE_MIN_MARGIN) over the next-best employee;
  - the caller also requires consecutive agreeing matches before showing it;
  - only today's embeddings count (people change clothes day to day);
  - face identity always wins over appearance, and appearance never marks
    attendance or desk time (those stay face-only).
In memory only: after a restart it re-learns as people are recognised again.
"""

import datetime
import logging
import os
import threading
import time
from collections import deque

import numpy as np

log = logging.getLogger("appearance")

APPEARANCE_MATCH_THRESHOLD = float(os.environ.get("APPEARANCE_MATCH_THRESHOLD", "0.78"))
APPEARANCE_MIN_MARGIN = float(os.environ.get("APPEARANCE_MIN_MARGIN", "0.06"))
LEARN_EVERY_SECONDS = 10.0      # at most one new embedding per employee per this
MAX_PER_EMPLOYEE = 12           # keep this many recent views per employee
MIN_BODY_PX = 60                # skip tiny/partial boxes (either side)


class AppearanceGallery:
    def __init__(self):
        self._lock = threading.Lock()
        self._views: dict[str, deque] = {}       # employee_id -> deque[(ts, embedding)]
        self._last_learn: dict[str, float] = {}
        self._day = None
        self._embedder = None
        self._embed_lock = threading.Lock()

    # ---- embeddings -----------------------------------------------------------

    def _embed(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        if x2 - x1 < MIN_BODY_PX or y2 - y1 < MIN_BODY_PX:
            return None
        with self._embed_lock:
            if self._embedder is None:
                from app.reid import config as reid_config
                from app.reid.reid_embedding import BodyReIdEmbedder

                self._embedder = BodyReIdEmbedder(reid_config.REID_MODEL_NAME, reid_config.REID_MODEL_PATH, reid_config.REID_DEVICE)
            emb = self._embedder.embed(frame, [x1, y1, x2, y2])
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32)
        n = float(np.linalg.norm(emb))
        return emb / n if n else None

    def _roll_day(self, now: float) -> None:
        day = datetime.date.fromtimestamp(now)
        if day != self._day:
            self._views.clear()
            self._last_learn.clear()
            self._day = day

    # ---- learn (from a face-confirmed identity) -------------------------------

    def learn(self, employee_id: str, frame: np.ndarray, bbox, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        with self._lock:
            self._roll_day(now)
            if now - self._last_learn.get(employee_id, 0) < LEARN_EVERY_SECONDS:
                return False
            self._last_learn[employee_id] = now
        emb = self._embed(frame, bbox)
        if emb is None:
            return False
        with self._lock:
            self._views.setdefault(employee_id, deque(maxlen=MAX_PER_EMPLOYEE)).append((now, emb))
        return True

    def add_embedding(self, employee_id: str, emb, now: float | None = None) -> None:
        """For tests and callers that already have an embedding."""
        now = now if now is not None else time.time()
        emb = np.asarray(emb, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) or 1.0)
        with self._lock:
            self._roll_day(now)
            self._views.setdefault(employee_id, deque(maxlen=MAX_PER_EMPLOYEE)).append((now, emb))

    # ---- match (a person with no visible face) --------------------------------

    def match_embedding(self, emb, exclude: set | None = None, now: float | None = None) -> tuple[str | None, float]:
        now = now if now is not None else time.time()
        emb = np.asarray(emb, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) or 1.0)
        with self._lock:
            self._roll_day(now)
            scored = sorted(
                ((max(float(emb @ v) for _, v in views), emp) for emp, views in self._views.items()
                 if views and emp not in (exclude or set())),
                reverse=True,
            )
        if not scored or scored[0][0] < APPEARANCE_MATCH_THRESHOLD:
            return None, scored[0][0] if scored else 0.0
        if len(scored) > 1 and scored[0][0] - scored[1][0] < APPEARANCE_MIN_MARGIN:
            return None, scored[0][0]  # two employees look alike: don't guess
        return scored[0][1], scored[0][0]

    def match(self, frame: np.ndarray, bbox, exclude: set | None = None) -> tuple[str | None, float]:
        if not self._views:
            return None, 0.0
        emb = self._embed(frame, bbox)
        return self.match_embedding(emb, exclude) if emb is not None else (None, 0.0)

    def known_count(self) -> int:
        return sum(1 for v in self._views.values() if v)


gallery = AppearanceGallery()
