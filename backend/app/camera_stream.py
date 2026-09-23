"""Pulls JPEG frames off a camera's RTSP stream in a background thread and
fans them out to however many websocket viewers are currently watching that
camera, so N browser tabs on the same camera share one RTSP connection
instead of each opening their own to the NVR."""

import concurrent.futures
import logging
import threading
import time

import cv2

from . import camera_db, config
from .face_pipeline import get_pipeline

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
        # Subset of _subscribers that are background collectors (see
        # face_collection.py's _CollectorSink), not a real person watching
        # the live feed — used by has_real_viewer() below so the (expensive)
        # person-detection overlay only runs while someone can actually see
        # it, not 24/7 just because background collection keeps this
        # thread alive. Never touches whether frames are broadcast/collected
        # — only whether the EXTRA person-overlay work happens.
        self._collector_subscribers: set = set()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._face_pipeline = None
        self._face_pipeline_failed = False
        # feed_frame() runs YOLO/InsightFace inference — easily slower than
        # the video loop's own frame interval under load. Runs on this
        # single-worker executor instead of inline so a slow detection pass
        # only ever delays detection, never the raw video frame this same
        # loop iteration already decoded and is about to broadcast. One
        # worker + the busy check below means at most one feed_frame() call
        # in flight per camera; a frame arriving while it's still running is
        # simply not sent to detection this cycle (SAMPLE_FPS-style
        # throttling already assumes/allows that), not queued up behind it.
        self._pipeline_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"face-pipeline-{camera_id}"
        )
        self._pipeline_future: concurrent.futures.Future | None = None

    def subscribe(self, queue, is_collector: bool = False) -> None:
        with self._lock:
            self._subscribers.add(queue)
            if is_collector:
                self._collector_subscribers.add(queue)
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def unsubscribe(self, queue) -> None:
        with self._lock:
            self._subscribers.discard(queue)
            self._collector_subscribers.discard(queue)
            if not self._subscribers:
                self._stop.set()

    def has_real_viewer(self) -> bool:
        with self._lock:
            return len(self._subscribers) > len(self._collector_subscribers)

    def _feed_pipeline(self, frame, has_viewer: bool) -> None:
        """Runs on _pipeline_executor's worker thread, never on the video
        loop's own thread — see the submit() call in _run(). Same
        never-take-down-the-video-broadcast contract as before: a missing
        model file or any other pipeline error logs once and disables face
        recognition for the rest of this stream's lifetime rather than
        retrying every frame."""
        try:
            self._face_pipeline.feed_frame(frame, has_viewer=has_viewer)
        except Exception:
            log.exception(
                "camera %s: face pipeline failed, disabling face recognition for this stream",
                self.camera_id,
            )
            self._face_pipeline_failed = True

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

                # Face detection/tracking/recognition, fed off the same
                # frame the live view already reads — no second RTSP
                # connection. Submitted to _pipeline_executor (see __init__)
                # rather than called inline: this keeps a slow/backed-up
                # detection pass from ever delaying the video encode+
                # broadcast below. Skipped (not queued) if the previous call
                # is still running.
                if not self._face_pipeline_failed and (self._pipeline_future is None or self._pipeline_future.done()):
                    if self._face_pipeline is None:
                        self._face_pipeline = get_pipeline(self.camera_id)
                    self._pipeline_future = self._pipeline_executor.submit(
                        self._feed_pipeline, frame, self.has_real_viewer()
                    )

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
