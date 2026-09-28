"""Downloads the person-Re-ID checkpoint the unique-footfall engine needs.

Run from backend/:  python -m scripts.fetch_reid_model

Why this is a separate step rather than a committed file: the checkpoint is
~9MB of binary weights, which doesn't belong in git, and torchreid's own
auto-download only ever fetches the ImageNet backbone — which is NOT
fine-tuned for person Re-ID and cannot separate people (see the measured
comparison in config.py's REID_MODEL_PATH comment, and reproduce it with
scripts/eval_reid_weights.py).

Source: kaiyangzhou/osnet on HuggingFace — OSNet's own author's mirror of
the deep-person-reid Model Zoo. Same osnet_x0_25 architecture as the
ImageNet default, trained on MSMT17 (combineall), so swapping it in changes
only the weights: no code change, no inference-speed change, no extra
memory. Falls back gracefully (config.REID_MODEL_PATH stays "") if this is
never run, so the app still boots without it.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TARGET = MODELS_DIR / "osnet_x0_25_msmt17.pth"

_REPO_FILE = (
    "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015"
    "_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
URL = f"https://huggingface.co/kaiyangzhou/osnet/resolve/main/{_REPO_FILE}"

# Sanity floor only — guards against a truncated download or an HTML error
# page landing on disk as a .pth, not a cryptographic integrity check.
MIN_BYTES = 5_000_000


def main() -> int:
    if TARGET.exists():
        print(f"Already present: {TARGET} ({TARGET.stat().st_size:,} bytes)")
        return 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}\n  -> {TARGET}")
    tmp = TARGET.with_suffix(".partial")
    try:
        urllib.request.urlretrieve(URL, tmp)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"FAILED: {e}", file=sys.stderr)
        print(
            "The engine will fall back to the ImageNet backbone, which counts the "
            "same person repeatedly — re-run this before trusting footfall numbers.",
            file=sys.stderr,
        )
        return 1

    size = tmp.stat().st_size
    if size < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        print(f"FAILED: downloaded file is only {size:,} bytes (expected >{MIN_BYTES:,})", file=sys.stderr)
        return 1

    tmp.replace(TARGET)
    print(f"OK: {TARGET} ({size:,} bytes)")
    print("Restart the backend to load it (config.REID_MODEL_PATH picks it up automatically).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
