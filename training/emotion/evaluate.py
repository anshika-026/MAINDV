"""Evaluate an emotion model on the test split: accuracy, per-class
precision/recall (classification report) and the confusion matrix.

    python evaluate.py --data dataset/test --model output/emotion_model.keras
    python evaluate.py --model ../../backend/models/emotion_model_v3.keras   # the deployed model
"""

from __future__ import annotations

import argparse

from common import DEFAULT_BATCH_SIZE, load_split


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset/test")
    ap.add_argument("--model", default="output/emotion_model.keras")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = ap.parse_args(argv)

    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    test, class_names = load_split(args.data, args.batch_size, shuffle=False)
    model = tf.keras.models.load_model(args.model)
    loss, accuracy = model.evaluate(test)
    print(f"\nTest loss {loss:.4f}  test accuracy {accuracy:.4f}\n")

    y_true, y_pred = [], []
    for images, labels in test:
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(model.predict(images, verbose=0), axis=1))
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_true, y_pred))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
