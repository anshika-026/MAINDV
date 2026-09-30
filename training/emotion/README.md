# Emotion (mood) model training

Training and evaluation for the facial-expression CNN the backend ships as
`backend/models/emotion_model_v3.keras`. The Behavior Analytics page uses it
through `backend/app/behavior_webcam.py`. It's based on the `mood-detection`
branch (commit `f1faf21`, by Anshika). These are the v3 training settings,
which produced the deployed model.

This is an offline tool. Nothing here runs inside the backend, and the
backend doesn't need TensorFlow unless you use the webcam demo page (see
`backend/requirements-optional.txt`).

## Model

- **Input:** 48x48 grayscale face crop, scaled to 0..1.
- **Output:** 7 classes, in this order:

  | index | emotion |
  |---|---|
  | 0 | angry |
  | 1 | disgust |
  | 2 | fear |
  | 3 | happy |
  | 4 | neutral |
  | 5 | sad |
  | 6 | surprise |

  The backend relies on this order, and `common.py` refuses a dataset
  whose folders don't match it.
- **Architecture:** 3 conv blocks (32, 64, 128 filters; batch norm; dropout
  0.25), global average pooling, dense 128 with dropout 0.5, then softmax.
  Training adds rotation, zoom, shift and horizontal-flip augmentation.
- **Training:** Adam 1e-3, early stopping (patience 7), learning rate halved
  on plateau (patience 3), and class weight 2.5 for the rare "disgust"
  class.

## Setup

```bash
cd training/emotion
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

The layout is FER-2013 style. It's never committed (git-ignored); it's
face images of people.

```
dataset/
  train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
  val/  {same 7 folders}
  test/ {same 7 folders}
```

```bash
python explore_dataset.py --data dataset/train     # counts per class + one sample each
```

## Train, evaluate, try it

```bash
python train.py --data dataset --out output/emotion_model.keras
python evaluate.py --data dataset/test --model output/emotion_model.keras
python webcam_demo.py --model output/emotion_model.keras

# Compare with the model in production:
python evaluate.py --data dataset/test --model ../../backend/models/emotion_model_v3.keras
```

`train.py` never overwrites the deployed model.

## Deploying a new model

Only deploy if it beats the current model on the same test split, both
overall and per class (see the classification report).

1. Copy it to `backend/models/emotion_model_v3.keras`, keeping the name, or
   set `EMOTION_KERAS_MODEL` to its path.
2. Update the `emotion_keras` checksum in `backend/app/models.py`
   (`sha256sum backend/models/emotion_model_v3.keras`), or
   `python -m scripts.fetch_models --check` will report it as `BADSUM`.
3. Restart the backend.

Earlier iterations (v1 and v2 training and evaluation scripts) are in the
`mood-detection` branch history. v3 supersedes them.
