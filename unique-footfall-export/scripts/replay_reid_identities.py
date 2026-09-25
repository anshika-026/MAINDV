"""Replays real person crops through the full identity pipeline and reports
how many distinct identities come out the other side.

    python -m scripts.replay_reid_identities --snapshots <dir> [--db <old.db>]

This is the end-to-end accuracy check that the unit tests can't be: it uses
the real embedder, the real vector gallery, the real auto-enrolment rule and
the real similarity/margin thresholds from config — just driven from crops on
disk instead of a live RTSP feed, so it is deterministic and repeatable.

Read the output as: how many real humans are in these crops vs. how many
identities the engine created. Those two numbers matching is what "unique
footfall is accurate" actually means. It writes a montage per resulting
identity so the grouping can be checked by eye rather than taken on trust.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.reid_embedding import BodyReIdEmbedder  # noqa: E402


class InMemoryGallery:
    """Same matching rules as peopleid_gallery.VectorGallery (best score per
    PERSON, threshold + margin over the runner-up person), without touching
    the live database."""

    def __init__(self, threshold: float, min_margin: float):
        self.threshold = threshold
        self.min_margin = min_margin
        self.by_person: dict[int, list[np.ndarray]] = defaultdict(list)

    def best_match(self, vec: np.ndarray) -> tuple[int | None, float, str]:
        """Returns (person_id|None, best_score, reason) — reason explains a
        rejection so the replay can report WHY identities were created."""
        if not self.by_person:
            return None, 0.0, "empty_gallery"
        scores = sorted(
            ((pid, max(float(v @ vec) for v in vecs)) for pid, vecs in self.by_person.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        best_pid, best = scores[0]
        runner_up = scores[1][1] if len(scores) > 1 else 0.0
        if best < self.threshold:
            return None, best, "below_threshold"
        if (best - runner_up) < self.min_margin:
            if runner_up >= self.threshold:
                # Both clear the threshold: same person, split across two
                # auto-created identities. Matching the best one is correct;
                # rejecting would mint a third fragment of that same person.
                # Mirrors VectorGallery(duplicate_identities_expected=True).
                return best_pid, best, "matched_duplicate_identity"
            return None, best, "margin_runner_up_below"
        return best_pid, best, "matched"

    def add(self, person_id: int, vec: np.ndarray) -> None:
        self.by_person[person_id].append(vec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="directory of person crop .jpg files")
    parser.add_argument("--db", help="sqlite file with a reid_snapshots table, to group crops by their ORIGINAL id")
    parser.add_argument("--threshold", type=float, default=config.REID_SIMILARITY_THRESHOLD)
    parser.add_argument("--margin", type=float, default=config.REID_MIN_MARGIN)
    parser.add_argument("--out", default="replay_identities", help="directory for per-identity montages")
    parser.add_argument(
        "--sweep",
        help="comma-separated thresholds to compare, e.g. 0.55,0.60,0.65,0.70,0.75 "
        "(embeds once, re-runs matching for each — montages are written for --threshold only)",
    )
    args = parser.parse_args()

    snap_dir = Path(args.snapshots)
    files = sorted(snap_dir.glob("*.jpg"))
    if not files:
        print(f"No .jpg crops in {snap_dir}")
        return 1

    original_of: dict[str, int] = {}
    if args.db:
        with sqlite3.connect(args.db) as conn:
            for pid, path in conn.execute("SELECT person_id, file_path FROM reid_snapshots"):
                original_of[Path(path).name] = pid

    print(f"model     : {config.REID_MODEL_PATH or '(ImageNet fallback)'}")
    print(f"threshold : {args.threshold}   margin: {args.margin}")
    print(f"crops     : {len(files)} from {snap_dir}")
    if original_of:
        print(f"original identities in that data: {len(set(original_of.values()))}")

    embedder = BodyReIdEmbedder(config.REID_MODEL_NAME, config.REID_MODEL_PATH, config.REID_DEVICE)

    # Embed once; matching is cheap to redo per threshold.
    embedded: list[tuple[Path, np.ndarray]] = []
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        vec = embedder.embed(img, [0, 0, w, h])
        if vec is not None:
            embedded.append((path, vec))

    def run(threshold: float) -> tuple[dict[int, list[Path]], dict[str, int], int]:
        gal = InMemoryGallery(threshold, args.margin)
        out: dict[int, list[Path]] = defaultdict(list)
        why: dict[str, int] = defaultdict(int)
        nid, matched = 1, 0
        for path, vec in embedded:
            pid, _score, reason = gal.best_match(vec)
            why[reason] += 1
            if pid is None:
                pid = nid
                nid += 1
            else:
                matched += 1
            gal.add(pid, vec)
            out[pid].append(path)
        return out, why, matched

    if args.sweep:
        print("\nthreshold sweep (identities produced from the same crops):")
        print(f"  {'threshold':<10} {'identities':<11} {'matched':<9} below_threshold")
        for th in [float(t) for t in args.sweep.split(",")]:
            out, why, matched = run(th)
            print(f"  {th:<10.2f} {len(out):<11} {matched:<9} {why.get('below_threshold', 0)}")
        print("\nFewer identities is only better if they are still the RIGHT groupings —")
        print("check the montages before adopting a lower threshold.")

    gallery = InMemoryGallery(args.threshold, args.margin)

    next_id = 1
    assigned: dict[int, list[Path]] = defaultdict(list)
    matched_count = 0
    reasons: dict[str, int] = defaultdict(int)

    for path, vec in embedded:
        pid, score, reason = gallery.best_match(vec)
        reasons[reason] += 1
        if pid is None:
            pid = next_id
            next_id += 1
        else:
            matched_count += 1
        gallery.add(pid, vec)
        assigned[pid].append(path)

    print(f"\nidentities created: {len(assigned)}")
    print(f"crops matched to an existing identity: {matched_count}/{len(files)}")
    print("why each crop did NOT match (i.e. why an identity was created):")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<24} {n}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pid, paths in sorted(assigned.items()):
        originals = sorted({original_of.get(p.name) for p in paths if p.name in original_of})
        print(f"  PERSON_{pid:03d}: {len(paths):>3} crop(s)" + (f"  (was ids {originals})" if originals else ""))
        thumbs = []
        for p in paths[:12]:
            img = cv2.imread(str(p))
            if img is not None:
                thumbs.append(cv2.resize(img, (96, 192)))
        if thumbs:
            cv2.imwrite(str(out_dir / f"person_{pid:03d}.jpg"), np.hstack(thumbs))

    print(f"\nmontages written to {out_dir.resolve()} — check the grouping by eye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
