"""
models.py

Every ML model the backend uses, where it lives, and which features need it.
Used by:
  - /ready and startup, to report missing models clearly instead of failing
    later inside a camera thread;
  - every loader (face_pipeline, person_detection, reid, expression), via
    require(), so that in MODEL_OFFLINE_MODE a missing file is a clear,
    actionable error instead of a silent internet download (ultralytics,
    InsightFace, torchreid and transformers all try to download missing
    weights by default);
  - scripts/fetch_models.py, to provision a machine before deployment.

Paths default to MODEL_DIR and can be overridden per model with the same
environment variables the loaders always honoured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass(frozen=True)
class ModelSpec:
    key: str
    description: str
    path: Path
    features: tuple[str, ...]
    required: bool = True          # False = optional feature, app runs without it
    is_dir: bool = False
    source: str = ""               # where scripts/fetch_models.py gets it
    sha256: str | None = None      # pinned checksum, if known
    notes: str = field(default="", compare=False)


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var, "").strip()
    return Path(raw) if raw else default


def _hf_cache_dir(repo: str) -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub" / ("models--" + repo.replace("/", "--"))


EMOTION_HF_REPO = os.environ.get("EMOTION_MODEL_NAME", "HardlyHumans/Facial-expression-detection")


def specs() -> list[ModelSpec]:
    md = Path(config.MODEL_DIR)
    return [
        ModelSpec("yolo_person", "YOLOv8n COCO person detector + tracker", _env_path("YOLO_PERSON_WEIGHTS", md / "yolov8n.pt"),
                  ("person_detection", "footfall", "staff_count", "intrusion", "live_overlay", "face_recognition"),
                  source="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
                  sha256="f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"),
        ModelSpec("yolo_face", "YOLOv8n-face detector", _env_path("YOLO_FACE_WEIGHTS", md / "yolov8n-face.pt"),
                  ("face_recognition",), source="https://github.com/akanametov/yolo-face/releases (yolov8n-face.pt)",
                  notes="See FACE_RECOGNITION_WIRING.md",
                  sha256="d545bf1add5aa736a4febac4f4f9245a6d596cd0fe70d5d57989fe0cb9e626ca"),
        ModelSpec("yolo_person_rescue", "YOLOv8s person detector (occasional rescue sweep)",
                  _env_path("PERSON_RESCUE_WEIGHTS", md / "yolov8s.pt"), ("face_recognition", "live_overlay"),
                  source="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
                  sha256="1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"),
        ModelSpec("reid_osnet", "OSNet x0.25 MSMT17 person Re-ID weights",
                  _env_path("REID_MODEL_PATH", md / "osnet_x0_25_msmt17.pth"), ("footfall", "appearance_handoff"),
                  source="python -m scripts.fetch_models (huggingface kaiyangzhou/osnet)",
                  sha256="cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18"),
        ModelSpec("insightface_buffalo_l", "InsightFace buffalo_l (face alignment + ArcFace embedding)",
                  Path(config.INSIGHTFACE_ROOT) / "models" / "buffalo_l", ("face_recognition", "face_enrollment"),
                  is_dir=True, source="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"),
        ModelSpec("expression_hf", f"Facial expression classifier ({EMOTION_HF_REPO})", _hf_cache_dir(EMOTION_HF_REPO),
                  ("expression",), required=False, is_dir=True, source=f"huggingface: {EMOTION_HF_REPO}"),
        ModelSpec("emotion_keras", "Keras emotion model (webcam Behavior Analytics demo)",
                  _env_path("EMOTION_KERAS_MODEL", md / "emotion_model_v3.keras"), ("webcam_behavior",), required=False,
                  source="tracked in git (backend/models/emotion_model_v3.keras)",
                  sha256="bff91f4e68dae4dd1b9a93870526047cabaeadaa340a17e07e0e20b98431efed"),
    ]


def get(key: str) -> ModelSpec:
    for s in specs():
        if s.key == key:
            return s
    raise KeyError(key)


def present(spec: ModelSpec) -> bool:
    p = spec.path
    if spec.is_dir:
        return p.is_dir() and any(p.iterdir())
    return p.is_file() and p.stat().st_size > 0


class ModelMissingError(FileNotFoundError):
    pass


def require(key: str) -> Path:
    """Path to a model that must exist before loading it. In offline mode a
    missing model raises ModelMissingError with the fix; outside offline
    mode it is returned anyway (the library may download it, as before)."""
    spec = get(key)
    if present(spec) or not config.MODEL_OFFLINE_MODE:
        return spec.path
    raise ModelMissingError(
        f"Model '{spec.key}' ({spec.description}) not found at {spec.path}. MODEL_OFFLINE_MODE is on, so it will "
        f"not be downloaded. Provision it with: python -m scripts.fetch_models  (source: {spec.source})"
    )


def status() -> list[dict]:
    return [{"key": s.key, "present": present(s), "required": s.required, "features": list(s.features),
             "path": str(s.path)} for s in specs()]


def missing_required() -> list[str]:
    return [s.key for s in specs() if s.required and not present(s)]


def apply_offline_environment() -> None:
    """Tell every library not to reach the network for weights."""
    if not config.MODEL_OFFLINE_MODE:
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("ULTRALYTICS_SYNC", "False")
