"""Object detection (backpack, handbag, bottle, laptop) on live cameras.

detector.py, object_tracker.py, ppe_detector.py, fixtures_detector.py and
color_detector.py are the object-detection module from the
`object-detection` branch (specialist model + COCO model, box fusion, size
sanity filters, cross-frame tracking), vendored with only two mechanical
changes: model files are read from the app's MODEL_DIR, and diagnostic
print() output goes to logging. service.py adapts it to this app: one
background worker, per-camera rate limit, 4 target classes, offline models,
failure backoff, latest results for the API.
"""
