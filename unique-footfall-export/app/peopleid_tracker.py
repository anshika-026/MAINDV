"""Person-body multi-object tracking for the People Identification module.

Wraps ultralytics' own built-in ByteTrack integration (`YOLO.track(...,
tracker="bytetrack.yaml", persist=True)`) rather than hand-rolling a
tracker — `ultralytics` (hence `torch`) is ALREADY an installed dependency
here (see pose_detector.py, which uses the same library's pose variant for
footfall/fall detection), so this adds zero new dependencies.

One PersonBodyTracker instance per camera (each holds its own YOLO model +
ByteTrack state) — mirrors how detection_worker.py's per-camera worker
process already loads its own copy of the face recognition models; track
IDs are meaningless mixed across cameras anyway, and Ultralytics' `persist`
tracking state is not designed to be safely shared across independently
-called, unrelated frame sources.

Cost/cadence is intentionally NOT decided here — see peopleid_worker.py for
the PEOPLEID_MOT_INTERVAL_SECONDS cadence gate. This class just answers
"given a frame, who (bodies) is in it and which track are they" when asked.
"""

from __future__ import annotations

import numpy as np


class PersonBodyTracker:
    def __init__(self, model_path: str, imgsz: int, confidence: float):
        from ultralytics import YOLO  # local import: keeps torch's load cost out of any process that doesn't need it

        self._model = YOLO(model_path)
        self._imgsz = imgsz
        self._confidence = confidence

    def update(self, frame: np.ndarray) -> list[dict]:
        """Returns a list of {"track_id": int, "bbox": [x1,y1,x2,y2], "confidence": float}
        for every person ByteTrack currently associates a stable ID with.
        Detections ByteTrack couldn't yet confirm into a track (very first
        frame of a brand-new person) are dropped — this module only ever
        deals in track-identified people, matching the rest of the People
        Identification pipeline's track-centric design."""
        results = self._model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0],
            imgsz=self._imgsz, conf=self._confidence, verbose=False,
        )
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        return [
            {"track_id": int(tid), "bbox": [float(v) for v in box], "confidence": float(conf)}
            for tid, box, conf in zip(track_ids, xyxy, confs)
        ]
