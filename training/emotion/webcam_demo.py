"""Local webcam demo of an emotion model (desktop window, press Q to quit).
Same pipeline as the backend's Behavior Analytics page: Haar face
detection, 48x48 grayscale, predictions averaged over the last 10 frames.

    python webcam_demo.py --model output/emotion_model.keras [--camera 0]
"""

from __future__ import annotations

import argparse
from collections import deque

from common import IMAGE_SIZE, LABELS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="../../backend/models/emotion_model_v3.keras")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args(argv)

    import cv2
    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.load_model(args.model)
    history = deque(maxlen=10)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from the camera")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)):
                face = cv2.resize(gray[y:y + h, x:x + w], IMAGE_SIZE) / 255.0
                history.append(model.predict(face.reshape(1, *IMAGE_SIZE, 1), verbose=0)[0])
                avg = np.mean(history, axis=0)
                i = int(np.argmax(avg))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{LABELS[i].title()}: {avg[i] * 100:.1f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Emotion recognition (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
