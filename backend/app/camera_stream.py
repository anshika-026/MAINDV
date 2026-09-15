"""Pulls JPEG frames off a camera's RTSP stream in a background thread and
fans them out to however many websocket viewers are currently watching that
camera, so N browser tabs on the same camera share one RTSP connection
instead of each opening their own to the NVR."""

import logging
import threading
import time

import cv2

from . import camera_db, config

log = logging.getLogger("camera_stream")


def build_rtsp_url(cam: dict) -> str | None:
    host = cam.get("host")
    if not host:
        return None
    port = cam.get("port") or 554
    user = cam.get("user") or ""
    password = cam.get("password") or ""
    path = cam.get("stream_path") or ""
    if not path.startswith("/"):
        path = "/" + path
    auth = f"{user}:{password}@" if user or password else ""
    return f"rtsp://{auth}{host}:{port}{path}"


class CameraStream:
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._subscribers: set = set()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def subscribe(self, queue) -> None:
        with self._lock:
            self._subscribers.add(queue)
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def unsubscribe(self, queue) -> None:
        with self._lock:
            self._subscribers.discard(queue)
            if not self._subscribers:
                self._stop.set()

    def _run(self) -> None:
        cam = camera_db.get_camera_connection(self.camera_id)
        url = build_rtsp_url(cam) if cam else None
        if not url:
            log.warning("camera %s: no host configured, nothing to stream", self.camera_id)
            return

        interval = 1.0 / config.LIVE_STREAM_FPS
        cap = None
        try:
            while not self._stop.is_set():
                if cap is None:
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        log.warning("camera %s: failed to open %s, retrying in 3s", self.camera_id, url)
                        cap.release()
                        cap = None
                        time.sleep(3)
                        continue

                ok, frame = cap.read()
                if not ok:
                    log.info("camera %s: stream read failed, reconnecting", self.camera_id)
                    cap.release()
                    cap = None
                    time.sleep(2)
                    continue

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                data = buf.tobytes()

                with self._lock:
                    subs = list(self._subscribers)
                for q in subs:
                    try:
                        q.put_nowait(data)
                    except Exception:
                        pass

                time.sleep(interval)
        finally:
            if cap is not None:
                cap.release()

_streams: dict[int, CameraStream] = {}
_streams_lock = threading.Lock()


def get_stream(camera_id: int) -> CameraStream:
    with _streams_lock:
        stream = _streams.get(camera_id)
        if stream is None:
            stream = CameraStream(camera_id)
            _streams[camera_id] = stream
        return stream
