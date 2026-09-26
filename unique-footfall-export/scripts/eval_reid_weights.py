"""A/B-evaluates Re-ID embedding weights against the real person crops this
deployment has already collected (data/reid_snapshots, grouped by the
person_id the engine assigned at the time).

Run from backend/:  python -m scripts.eval_reid_weights

Reports, per candidate model, the cosine-similarity distribution for
SAME-person pairs vs DIFFERENT-person pairs, plus the error rates a given
threshold would produce. Higher separation between the two distributions =
fewer false merges (two people counted as one) and fewer fragmentations
(one person counted repeatedly) in the unique-footfall count.

Label caveat, stated explicitly because it affects how to read the output:
the person_id grouping comes from the engine's OWN past decisions. Crops
sharing a person_id are genuinely the same person (they were matched and
enrolled together from the same track), so SAME-person numbers are
trustworthy. DIFFERENT-person pairs are contaminated in the pessimistic
direction — when the engine over-fragmented, two ids can be the same human
— so the measured different-person similarity is an UPPER bound, and any
model's real-world separation is at least as good as reported here.
"""

from __future__ import annotations

import itertools
import sqlite3
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, reid_db  # noqa: E402
from app.reid_embedding import BodyReIdEmbedder  # noqa: E402


def load_crops() -> dict[int, list[np.ndarray]]:
    with sqlite3.connect(reid_db.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT person_id, file_path FROM reid_snapshots")]

    by_person: dict[int, list[np.ndarray]] = {}
    for row in rows:
        path = Path(row["file_path"])
        if not path.is_absolute():
            path = reid_db.SNAPSHOTS_DIR / path.name
        img = cv2.imread(str(path))
        if img is None:
            continue
        by_person.setdefault(row["person_id"], []).append(img)
    return by_person


def embed_all(embedder: BodyReIdEmbedder, by_person: dict[int, list[np.ndarray]]) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {}
    for pid, crops in by_person.items():
        vecs = []
        for crop in crops:
            h, w = crop.shape[:2]
            vec = embedder.embed(crop, [0, 0, w, h])
            if vec is not None:
                vecs.append(vec)
        if vecs:
            out[pid] = vecs
    return out


def similarity_stats(embeddings: dict[int, list[np.ndarray]]) -> tuple[list[float], list[float]]:
    same, diff = [], []
    for vecs in embeddings.values():
        for a, b in itertools.combinations(vecs, 2):
            same.append(float(a @ b))
    pids = list(embeddings)
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            for a in embeddings[pids[i]]:
                for b in embeddings[pids[j]]:
                    diff.append(float(a @ b))
    return same, diff


def describe(name: str, values: list[float]) -> None:
    if not values:
        print(f"  {name}: (none)")
        return
    ordered = sorted(values)
    print(
        f"  {name}: n={len(ordered)} min={ordered[0]:.3f} "
        f"p5={ordered[int(0.05 * len(ordered))]:.3f} median={st.median(ordered):.3f} "
        f"p95={ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]:.3f} max={ordered[-1]:.3f}"
    )


def best_threshold(same: list[float], diff: list[float]) -> tuple[float, float, float, float]:
    """Threshold minimizing total error rate, plus the rates it produces."""
    best = (0.0, 1.0, 1.0, 1.0)
    for step in range(20, 100):
        th = step / 100
        frag = sum(1 for v in same if v < th) / max(1, len(same))
        merge = sum(1 for v in diff if v >= th) / max(1, len(diff))
        if frag + merge < best[1] + best[2]:
            best = (th, frag, merge, frag + merge)
    return best


def evaluate(label: str, model_name: str, model_path: str, by_person: dict[int, list[np.ndarray]]) -> None:
    print(f"\n=== {label} ===")
    print(f"  model_name={model_name!r} model_path={model_path or '(imagenet fallback + calibration centering)'!r}")
    embedder = BodyReIdEmbedder(model_name, model_path, config.REID_DEVICE)
    embeddings = embed_all(embedder, by_person)
    same, diff = similarity_stats(embeddings)
    describe("SAME-person     ", same)
    describe("DIFFERENT-person", diff)
    if same and diff:
        gap = st.median(same) - st.median(diff)
        print(f"  separation (median same - median diff): {gap:+.3f}")
        th, frag, merge, total = best_threshold(same, diff)
        print(f"  best threshold={th:.2f} -> fragmentation {frag * 100:.1f}%, false-merge {merge * 100:.1f}% (total {total * 100:.1f}%)")
        cur = config.REID_SIMILARITY_THRESHOLD
        frag_c = sum(1 for v in same if v < cur) / len(same)
        merge_c = sum(1 for v in diff if v >= cur) / len(diff)
        print(f"  at CURRENT threshold={cur:.2f} -> fragmentation {frag_c * 100:.1f}%, false-merge {merge_c * 100:.1f}%")


def main() -> None:
    by_person = load_crops()
    total = sum(len(v) for v in by_person.values())
    print(f"Loaded {total} real crops across {len(by_person)} recorded identities from {reid_db.SNAPSHOTS_DIR}")
    usable = {k: v for k, v in by_person.items() if len(v) >= 2}
    print(f"{len(usable)} identities have >=2 crops (usable for same-person pairs)")

    evaluate("CURRENT (ImageNet features)", config.REID_MODEL_NAME, "", by_person)

    candidate = Path(__file__).resolve().parent.parent / "models" / "osnet_x0_25_msmt17.pth"
    if candidate.exists():
        evaluate("CANDIDATE (MSMT17 Re-ID-trained)", "osnet_x0_25", str(candidate), by_person)
    else:
        print(f"\n(candidate weights not found at {candidate})")


if __name__ == "__main__":
    main()
