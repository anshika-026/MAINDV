"""
Provision every model the backend needs, so production runs fully offline
(MODEL_OFFLINE_MODE=true never downloads anything at runtime).

Run from backend/ on a machine WITH internet access, then copy MODEL_DIR,
INSIGHTFACE_ROOT and the Hugging Face cache (HF_HOME) to the target machine,
or run it on the target during provisioning:

    python -m scripts.fetch_models            # download whatever is missing
    python -m scripts.fetch_models --check    # report only; exit 1 if a required model is missing
    python -m scripts.fetch_models --skip-optional

Model locations come from app/models.py (and the same environment variables
the app uses), so this script and the app always agree on paths.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MODEL_OFFLINE_MODE", "false")  # this script is the one place downloads are allowed

from app import models  # noqa: E402

DIRECT_URLS = {
    "yolo_person": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "yolo_person_rescue": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
    "insightface_buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        print(f"  downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp_path, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        os.replace(tmp_path, dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(spec: models.ModelSpec) -> None:
    if spec.key == "insightface_buffalo_l":
        zip_path = spec.path.parent / "buffalo_l.zip"
        _download(DIRECT_URLS[spec.key], zip_path)
        spec.path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():
                name = Path(member).name
                if name.endswith(".onnx"):
                    with z.open(member) as src, open(spec.path / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        zip_path.unlink()
    elif spec.key in DIRECT_URLS:
        _download(DIRECT_URLS[spec.key], spec.path)
    elif spec.key == "reid_osnet":
        from scripts import fetch_reid_model

        _download(fetch_reid_model.URL, spec.path)
        if spec.path.stat().st_size < fetch_reid_model.MIN_BYTES:
            spec.path.unlink()
            raise RuntimeError("Re-ID download truncated")
    elif spec.source.startswith("git: "):
        # Weights published on a branch of this repository (e.g. the
        # object-detection branch's models/). Needs a clone with that ref.
        import subprocess

        ref_path = spec.source[len("git: "):]
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        with open(spec.path, "wb") as f:
            done = subprocess.run(["git", "show", ref_path], stdout=f, stderr=subprocess.PIPE)
        if done.returncode != 0:
            spec.path.unlink(missing_ok=True)
            raise RuntimeError(f"git show {ref_path} failed: {done.stderr.decode(errors='replace').strip()} "
                               f"(run `git fetch origin object-detection` first)")
    elif spec.key == "expression_hf":
        from huggingface_hub import snapshot_download

        snapshot_download(models.EMOTION_HF_REPO)
    else:
        raise RuntimeError(f"no automatic source for {spec.key}; get it manually: {spec.source}")
    if spec.sha256 and not spec.is_dir and _sha256(spec.path) != spec.sha256:
        spec.path.unlink()
        raise RuntimeError(f"checksum mismatch for {spec.key}; file removed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report; exit 1 if a required model is missing")
    ap.add_argument("--skip-optional", action="store_true")
    args = ap.parse_args(argv)

    missing_required = []
    for spec in models.specs():
        ok = models.present(spec)
        tag = "required" if spec.required else "optional"
        if ok and spec.sha256 and not spec.is_dir and _sha256(spec.path) != spec.sha256:
            print(f"[BADSUM ] {spec.key:22} {tag:8} {spec.path}  (checksum differs from the verified weights)")
            if spec.required:
                missing_required.append(spec.key)
            continue
        print(f"[{'ok' if ok else 'MISSING':7}] {spec.key:22} {tag:8} {spec.path}")
        if ok or args.check or (args.skip_optional and not spec.required):
            if not ok and spec.required:
                missing_required.append(spec.key)
            continue
        try:
            fetch(spec)
            print(f"          -> fetched {spec.key}")
        except Exception as e:
            print(f"          -> FAILED: {e}\n             manual source: {spec.source}")
            if spec.required:
                missing_required.append(spec.key)
    if missing_required:
        print(f"\nMissing required models: {', '.join(missing_required)}")
        return 1
    print("\nAll required models present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
