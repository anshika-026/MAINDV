"""In-process vector search for the People Identification gallery.

Deliberately a plain NumPy matrix, not FAISS: at this module's realistic
scale (100+ people x 10-20 embeddings each = a couple thousand vectors),
FAISS's IndexFlatIP performs the exact same brute-force matmul this does
internally — the existing production recognizer.py's brute-force Python
loop already proves this scale is fine even unvectorized (~36 people
today). Vectorizing it here (one matrix-vector product instead of a Python
loop) already costs low-single-digit milliseconds, comfortably cheaper than
the ~1-1.5s a single face embedding call itself costs — a new binary
dependency (with its own BLAS/threading that can contend with onnxruntime
on an already CPU-constrained box) bought nothing at this scale. Kept
behind this small class specifically so an ANN backend could be swapped in
later if a client's gallery ever genuinely outgrows this approach.

Each per-camera worker process builds/reloads its OWN copy from its
embedding store, mirroring exactly how recognizer.py's `_reload_enrolled`
rebuilds `self._enrolled` from face_db in-process — no cross-process
shared memory, no index file on disk, nothing new to keep in sync beyond
the existing "signal every worker to reload" queue-message pattern (see
peopleid_worker.py / pipeline.py's reload_people_gallery).

Deliberately parameterized over WHICH embedding store and WHICH
threshold/margin defaults to use (`embedding_loader`, `similarity_threshold`,
`min_margin` below) rather than importing peopleid_db/config.PEOPLEID_*
directly: reid_worker.py (the body-appearance Re-ID engine) reuses this
exact class over its own reid_db embeddings and REID_* thresholds — a
face embedding and a body embedding are different-dimension, different
-space vectors that must never share one gallery/threshold, but the
matching ALGORITHM (cosine search, best-per-person, margin-gated
decisiveness) is identical between the two, so only this construction
point needs to differ per caller. Embeddings are expected to already be
comparable on arrival (L2-normalizable, same feature space) — any
model-specific fix-up (e.g. reid_embedding.py's fixed-calibration mean
-centering, needed for its particular pretrained backbone's raw feature
bias) belongs in the embedding producer, not here, precisely so this class
can stay this simple and shared.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class VectorGallery:
    def __init__(
        self,
        embedding_loader: Callable[[], list[tuple[int, int, np.ndarray]]],
        similarity_threshold: float,
        min_margin: float,
        dim: int = 512,
        duplicate_identities_expected: bool = False,
    ):
        """duplicate_identities_expected changes what a near-tie between the
        top two people MEANS, which differs fundamentally between this
        class's two callers:

        Face gallery (peopleid_*): identities are enrolled by a human, so
        every person in it is a genuinely distinct individual. Two of them
        scoring within min_margin of each other is real ambiguity about WHO
        this is — picking the nominally-higher one risks labelling someone
        with another person's name, so it must resolve to Unknown. False.

        Re-ID gallery (reid_*): identities are created automatically, so the
        gallery routinely holds SEVERAL fragments of the same person. Two of
        them scoring within min_margin while BOTH clear the threshold is not
        ambiguity — it is evidence they are the same human. Rejecting there
        is actively harmful: it spawns yet another fragment of that person,
        which makes the next near-tie more likely, which spawns another. On
        this deployment's own data that loop rejected 94 of 186 crops and
        turned a handful of real people into 131 "unique" identities. True.

        In both cases a near-tie where the runner-up is BELOW the threshold
        is still rejected — that is the original 0.61/0.60/0.59 case this
        margin exists for, and it is unaffected by this flag.
        """
        self._embedding_loader = embedding_loader
        self._similarity_threshold = similarity_threshold
        self._min_margin = min_margin
        self._duplicate_identities_expected = duplicate_identities_expected
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._person_ids: np.ndarray = np.zeros((0,), dtype=np.int64)
        self.reload()

    def reload(self) -> None:
        rows = self._embedding_loader()
        if not rows:
            self._matrix = np.zeros((0, self._matrix.shape[1]), dtype=np.float32)
            self._person_ids = np.zeros((0,), dtype=np.int64)
            return
        _ids, person_ids, embeddings = zip(*rows)
        matrix = np.stack(embeddings).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        self._matrix = matrix / norms
        self._person_ids = np.array(person_ids, dtype=np.int64)

    @property
    def size(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def person_count(self) -> int:
        return int(len(set(self._person_ids.tolist())))

    def search(self, embedding: np.ndarray, top_k: int = 3) -> list[tuple[int, float]]:
        """Best cosine-similarity match PER PERSON (a gallery has many
        embeddings/person, so raw top-k rows would just return the same
        person repeatedly) — returns up to top_k (person_id, score) pairs,
        highest score first."""
        if self._matrix.shape[0] == 0:
            return []
        query = embedding.astype(np.float32) / (np.linalg.norm(embedding) + 1e-8)
        sims = self._matrix @ query
        best_per_person: dict[int, float] = {}
        for pid, sim in zip(self._person_ids.tolist(), sims.tolist()):
            if pid not in best_per_person or sim > best_per_person[pid]:
                best_per_person[pid] = sim
        ranked = sorted(best_per_person.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def best_match(self, embedding: np.ndarray, similarity_threshold: float | None = None) -> tuple[int | None, float, float]:
        """(person_id, score, margin_over_runner_up) for a query embedding,
        or (None, best_score, margin) if the top score doesn't clear the
        threshold (this gallery's own similarity_threshold, or a per-camera
        override passed here), or clears it by less than this gallery's
        min_margin over the second-best person (spec: three close scores
        like 0.61/0.60/0.59 must resolve to Unknown, never a forced pick of
        the nominally-highest one)."""
        threshold = similarity_threshold if similarity_threshold is not None else self._similarity_threshold
        ranked = self.search(embedding, top_k=2)
        if not ranked:
            return None, 0.0, 0.0
        best_pid, best_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - runner_up_score
        if best_score < threshold:
            return None, best_score, margin
        if margin < self._min_margin:
            # A near-tie where the runner-up ALSO clears the threshold means
            # something different in an auto-enrolled gallery than in a
            # human-curated one — see duplicate_identities_expected in
            # __init__ for why this is not simply "reject when unsure".
            both_confident = runner_up_score >= threshold
            if not (self._duplicate_identities_expected and both_confident):
                return None, best_score, margin
        return best_pid, best_score, margin
