"""
rtsp_reader.py

Reads one RTSP camera in its own OS process and publishes only the newest
frame through shared memory. camera_stream.CameraStream starts one per
streaming camera.

Why a separate process: reading RTSP means pulling every frame the camera
sends (25 fps here) — skip reads and the frames queue up and everything
downstream falls steadily behind real time. Inside the backend process that
read loop shares one Python interpreter with face recognition and footfall,
and was measured managing only 10-16 of the camera's 25 fps while they ran,
so video drifted tens of seconds behind. In its own process nothing can
hold it up.

Deliberately imports only cv2/numpy (not the rest of `app`), so each child
starts in about a second without loading any of the ML models.

Protocol with the parent:
  - meta_q gets ("shm", name, shape) whenever a new shared-memory block is
    created (first frame, or a resolution change), and ("log", text) lines.
  - The block holds 2 frame slots. The child writes the newest frame into
    slot (seq + 1) % 2 and only then increments seq, so the parent always
    reads a complete frame from slot seq % 2.
"""

import time
from multiprocessing import shared_memory

import numpy as np

# How far behind the camera the reader may fall before it reconnects to jump
# back to live. With the reader in its own process this should only happen
# when the whole machine is overloaded or the network stalls.
MAX_STREAM_LAG_SECONDS = 3.0
STATS_EVERY_SECONDS = 60


def run(url: str, fps: float, stop, seq, meta_q) -> None:
    import cv2

    interval = 1.0 / fps
    shm = None
    shape = None
    cap = None
    lag_anchor = None  # (wall clock, stream time) at a moment we were live
    last_emit = 0.0
    stats_at, grabs, emits = time.monotonic(), 0, 0

    def log(text: str) -> None:
        try:
            meta_q.put_nowait(("log", text))
        except Exception:
            pass

    try:
        while not stop.is_set():
            if cap is None:
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    log("failed to open stream, retrying in 3s")
                    cap.release()
                    cap = None
                    time.sleep(3)
                    continue
                lag_anchor = None

            # grab() blocks until the camera's next frame, so this loop runs
            # at the source rate; only one frame per `interval` is decoded to
            # pixels (retrieve) and published.
            if not cap.grab():
                log("stream read failed, reconnecting")
                cap.release()
                cap = None
                time.sleep(2)
                continue
            now = time.monotonic()
            grabs += 1

            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos > 0:
                if lag_anchor is None or (now - lag_anchor[0]) < (pos - lag_anchor[1]):
                    lag_anchor = (now, pos)
                lag = (now - lag_anchor[0]) - (pos - lag_anchor[1])
                if lag > MAX_STREAM_LAG_SECONDS:
                    log(f"{lag:.1f}s behind live, reconnecting to catch up")
                    cap.release()
                    cap = None
                    continue

            if now - stats_at >= STATS_EVERY_SECONDS:
                log(f"reading {grabs / (now - stats_at):.1f} fps from camera, publishing {emits / (now - stats_at):.1f} fps")
                stats_at, grabs, emits = now, 0, 0

            if now - last_emit < interval:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            # Keep a steady schedule: frames arrive every ~40 ms, so
            # "interval since the last emit" alone rounds 125 ms up to 160 ms
            # (6 fps instead of 8). Never schedule into the past, though.
            last_emit = max(last_emit + interval, now - interval)
            emits += 1

            if shm is None or frame.shape != shape:
                if shm is not None:
                    del slots  # a live numpy view blocks shm.close()
                    shm.close()
                    shm.unlink()
                shape = frame.shape
                shm = shared_memory.SharedMemory(create=True, size=2 * frame.nbytes)
                slots = np.ndarray((2, *shape), dtype=np.uint8, buffer=shm.buf)
                meta_q.put(("shm", shm.name, shape))

            slots[(seq.value + 1) % 2] = frame
            seq.value += 1
    finally:
        if cap is not None:
            cap.release()
        if shm is not None:
            del slots
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
