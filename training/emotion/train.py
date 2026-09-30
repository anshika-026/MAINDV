"""Train the emotion CNN (model v3, the one the backend ships as
backend/models/emotion_model_v3.keras).

    python train.py --data dataset --out output/emotion_model.keras

Architecture, augmentation, optimiser, callbacks and class weights are the
v3 settings from the mood-detection branch, unchanged. It writes to --out,
never over the deployed model; see README.md for how to deploy a new one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_BATCH_SIZE, IMAGE_SIZE, LABELS, load_split

# Disgust is heavily under-represented in FER-style datasets.
CLASS_WEIGHTS = {0: 1.0, 1: 2.5, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}


def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    augmentation = models.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        layers.RandomFlip("horizontal"),
    ])

    def block(filters):
        return [
            layers.Conv2D(filters, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(filters, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
        ]

    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 1)),
        augmentation,
        *block(32),
        *block(64),
        *block(128),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(LABELS), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset", help="folder with train/ and val/ subfolders (default: dataset)")
    ap.add_argument("--out", default="output/emotion_model.keras", help="where to save the trained model")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = ap.parse_args(argv)

    import tensorflow as tf

    data = Path(args.data)
    train, class_names = load_split(data / "train", args.batch_size, shuffle=True)
    val, _ = load_split(data / "val", args.batch_size, shuffle=False)
    print("Classes:", class_names)
    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001),
    ]
    model.fit(train, validation_data=val, epochs=args.epochs, class_weight=CLASS_WEIGHTS, callbacks=callbacks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"Saved {out}. Evaluate it with: python evaluate.py --model {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
