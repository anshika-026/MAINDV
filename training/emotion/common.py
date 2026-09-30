"""Shared settings for training and evaluating the emotion model.

The backend (backend/app/behavior_webcam.py, EMOTION_LABELS) maps the
model's output index i to LABELS[i]. Keras assigns indices from the
dataset's folder names in alphabetical order, so the folders MUST be
exactly these seven names; load_split() refuses anything else rather than
silently training a model whose outputs mean something different.
"""

from __future__ import annotations

from pathlib import Path

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_SIZE = (48, 48)
DEFAULT_BATCH_SIZE = 32


def check_class_names(found: list[str]) -> None:
    if [c.lower() for c in found] != LABELS:
        raise SystemExit(
            f"Dataset folders must be exactly {LABELS} (alphabetical; one folder per emotion). "
            f"Found {found}. The backend maps output index -> label in this order."
        )


def load_split(path: str | Path, batch_size: int = DEFAULT_BATCH_SIZE, shuffle: bool = False):
    """A normalised (0..1), 48x48 grayscale tf.data split from
    <path>/<emotion>/*.png, plus its class names (checked)."""
    import tensorflow as tf

    path = Path(path)
    if not path.is_dir():
        raise SystemExit(f"Dataset folder not found: {path}")
    ds = tf.keras.utils.image_dataset_from_directory(
        path, image_size=IMAGE_SIZE, batch_size=batch_size, color_mode="grayscale", shuffle=shuffle
    )
    check_class_names(ds.class_names)
    rescale = tf.keras.layers.Rescaling(1.0 / 255)
    return ds.map(lambda images, labels: (rescale(images), labels)), ds.class_names
