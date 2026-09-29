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
starts in about a second without loading any of the ML models. Settings
arrive as a plain dict from the parent for the same reason.

Resilience:
  - Every open and read has a timeout (FFmpeg open/read timeouts), so an
    unreachable camera or a stalled network can't block forever.
  - Failed connects/reads back off exponentially with jitter (2 s doubling
    to 60 s by default). A CP Plus / Hikvision NVR locks the account after
    repeated failed logins, so a tight retry loop can take a camera offline
    for everyone; a slow, jittered one can't.
  - While waiting between retries the child sends heartbeats, so the
    parent's watchdog can tell "waiting to retry" (fine) from "hung inside
    FFmpeg" (kill and restart the process).

Protocol with the parent (meta_q):
  ("shm", name, shape)      new shared-memory block (first frame / resolution change)
  ("log", text)             human-readable line for the parent's log (never the URL)
  ("status", state, info)   state in: connecting | connected | reconnecting
  ("heartbeat",)            alive, waiting to retry
The block holds 2 frame slots. The child writes the newest frame into slot
(seq + 1) % 2 and only then increments seq, so the parent always reads a
complete frame from slot seq % 2.
"""

import random
import time
from multiprocessing import shared_memory

import numpy as np

# How far behind the camera the reader may fall before it reconnects to jump
# back to live. With the reader in its own process this should only happen
# when the whole machine is overloaded or the network stalls.
MAX_STREAM_LAG_SECONDS = 3.0
STATS_EVERY_SECONDS = 60
HEARTBEAT_SECONDS = 2.0

DEFAULTS = {
    "connect_timeout": 10.0,
    "read_timeout": 10.0,
    "reconnect_delay": 2.0,
    "reconnect_max_delay": 60.0,
}


def reconnect_delay(attempt: int, base: float, maximum: float, rand=random.random) -> float:
    """attempt >= 1 -> seconds to wait, exponential with +/-20% jitter."""
    delay = min(maximum, base * (2 ** max(0, attempt - 1)))
    return delay * (0.8 + 0.4 * rand())


def open_capture(url: str, connect_timeout: float, read_timeout: float):
    import cv2

    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(connect_timeout * 1000),
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(read_timeout * 1000),
    ]
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)


def run(url: str, fps: float, stop, seq, meta_q, settings: dict | None = None) -> None:
    import cv2

    s = {**DEFAULTS, **(settings or {})}
    interval = 1.0 / fps
    shm = None
    slots = None
    shape = None
    cap = None
    lag_anchor = None  # (wall clock, stream time) at a moment we were live
    last_emit = 0.0
    stats_at, grabs, emits = time.monotonic(), 0, 0
    attempt = 0  # consecutive failed connects/reads

    def send(msg) -> None:
        try:
            meta_q.put_nowait(msg)
        except Exception:  # noqa: S110
            pass  # parent gone or queue full; nothing useful to do here

    def log(text: str) -> None:
        send(("log", text))

    def wait_before_retry(reason: str) -> None:
        nonlocal attempt
        attempt += 1
        delay = reconnect_delay(attempt, s["reconnect_delay"], s["reconnect_max_delay"])
        send(("status", "reconnecting", {"attempt": attempt, "delay": round(delay, 2), "reason": reason}))
        log(f"{reason}; retry {attempt} in {delay:.0f}s")
        deadline = time.monotonic() + delay
        while not stop.is_set():
            left = deadline - time.monotonic()
            if left <= 0:
                break
            stop.wait(min(HEARTBEAT_SECONDS, left))
            send(("heartbeat",))

    try:
        while not stop.is_set():
            if cap is None:
                send(("status", "connecting", {"attempt": attempt + 1}))
                cap = open_capture(url, s["connect_timeout"], s["read_timeout"])
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    wait_before_retry("failed to open stream (check address, credentials, network)")
                    continue
                lag_anchor = None

            # grab() blocks until the camera's next frame (bounded by the
            # read timeout), so this loop runs at the source rate; only one
            # frame per `interval` is decoded to pixels (retrieve) and published.
            if not cap.grab():
                cap.release()
                cap = None
                wait_before_retry("stream read failed or timed out")
                continue
            now = time.monotonic()
            grabs += 1
            if attempt:
                log(f"connected after {attempt} retr{'y' if attempt == 1 else 'ies'}")
                attempt = 0
            if grabs == 1 or (grabs % 250 == 0):
                send(("status", "connected", {}))

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
            if not ok or frame is None or frame.ndim != 3 or frame.dtype != np.uint8:
                continue  # corrupt/partial frame: skip it, keep the stream
            # Keep a steady schedule: frames arrive every ~40 ms, so
            # "interval since the last emit" alone rounds 125 ms up to 160 ms
            # (6 fps instead of 8). Never schedule into the past, though.
            last_emit = max(last_emit + interval, now - interval)
            emits += 1

            if shm is None or frame.shape != shape:
                if shm is not None:
                    slots = None  # a live numpy view blocks shm.close()
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
            slots = None
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
