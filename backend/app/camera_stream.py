"""Pulls JPEG frames off a camera's RTSP stream in a background thread and
fans them out to however many websocket viewers are currently watching that
camera, so N browser tabs on the same camera share one RTSP connection
instead of each opening their own to the NVR."""

import concurrent.futures
import logging
import multiprocessing
import queue
import threading
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

from . import analytics_settings, camera_db, config, footfall, intrusion, rtsp_reader
from .face_pipeline import get_pipeline
from .staff.service import service as staff_service

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
        # Most recent encoded frame — lets the footfall UAT panel show a still
        # of a gate to draw its counting zone on, without a second RTSP pull.
        self.last_jpeg: bytes | None = None
        # When the last frame arrived from the camera (time.time()), for real
        # online/offline status on the dashboard; None = never streamed.
        self.last_frame_at: float | None = None
        self._frameless: set = set()
        self._widths: dict = {}

    def subscribe(self, queue, is_collector: bool = False, width: int | None = None, wants_frames: bool = True) -> None:
        """is_collector: doesn't need the person-overlay detection (background
        work, or a plain Live Feed viewer). width: send JPEGs scaled down to
        this many pixels wide (None = full resolution). wants_frames=False:
        a keep-alive that only needs the read loop running — no JPEG is
        encoded for it at all."""
        with self._lock:
            self._subscribers.add(queue)
            if is_collector:
                self._collector_subscribers.add(queue)
            if not wants_frames:
                self._frameless.add(queue)
            elif width:
                self._widths[queue] = int(width)
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def unsubscribe(self, queue) -> None:
        with self._lock:
            self._subscribers.discard(queue)
            self._collector_subscribers.discard(queue)
            self._frameless.discard(queue)
            self._widths.pop(queue, None)
            if not self._subscribers:
                self._stop.set()

    def stop(self) -> None:
        """Feed switched off: end the read loop (and its reader process) now,
        even though subscribers are still attached."""
        self._stop.set()

    def resume(self) -> None:
        """Feed switched back on: restart the read loop if anything is still
        subscribed (keep-alives, open viewers)."""
        with self._lock:
            if self._subscribers and (self._thread is None or not self._thread.is_alive()):
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

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
            # Created here, inside the try, rather than in _run(): the
            # constructor loads the models, and a missing checkpoint must
            # disable face recognition, not kill the video thread.
            if self._face_pipeline is None:
                self._face_pipeline = get_pipeline(self.camera_id)
            self._face_pipeline.feed_frame(frame, has_viewer=has_viewer)
        except Exception:
            log.exception(
                "camera %s: face pipeline failed, disabling face recognition for this stream",
                self.camera_id,
            )
            self._face_pipeline_failed = True

    def _run(self) -> None:
        cam = camera_db.get_camera_connection(self.camera_id)
        if cam and not camera_db.is_streamable(cam):
            log.info("camera %s: feed switched off, not connecting", self.camera_id)
            return
        url = build_rtsp_url(cam) if cam else None
        if not url:
            log.warning("camera %s: no host configured, nothing to stream", self.camera_id)
            return
        # A camera with attendance_tracking off is live view only: frames are
        # broadcast to viewers but never sent to face recognition. (Footfall
        # is separately limited to "Entry/Exit" cameras — see footfall.py.)
        analytics = bool(cam.get("attendance_tracking", 1))
        if not analytics:
            log.info("camera %s: live view only, analytics off", self.camera_id)

        # RTSP is read in a separate process (rtsp_reader.py) that always
        # keeps just the newest frame, so the video never falls behind real
        # time however busy face recognition and footfall keep this process.
        ctx = multiprocessing.get_context("spawn")
        proc = stop = seq = meta_q = None
        shm = slots = None
        last_seq = 0

        def start_reader():
            nonlocal proc, stop, seq, meta_q, last_seq
            stop = ctx.Event()
            seq = ctx.Value("Q", 0, lock=False)
            meta_q = ctx.Queue()
            last_seq = 0
            proc = ctx.Process(
                target=rtsp_reader.run, args=(url, config.LIVE_STREAM_FPS, stop, seq, meta_q),
                daemon=True, name=f"rtsp-reader-{self.camera_id}",
            )
            proc.start()

        try:
            start_reader()
            while not self._stop.is_set():
                while True:
                    try:
                        msg = meta_q.get_nowait()
                    except queue.Empty:
                        break
                    if msg[0] == "shm":
                        if shm is not None:
                            slots = None
                            shm.close()
                        shm = shared_memory.SharedMemory(name=msg[1])
                        slots = np.ndarray((2, *msg[2]), dtype=np.uint8, buffer=shm.buf)
                    elif msg[0] == "log":
                        log.info("camera %s: %s", self.camera_id, msg[1])

                if not proc.is_alive():
                    log.warning("camera %s: reader process exited (code %s), restarting", self.camera_id, proc.exitcode)
                    if shm is not None:
                        slots = None
                        shm.close()
                        shm = None
                    time.sleep(2)
                    start_reader()
                    continue

                current = seq.value
                if slots is None or current == last_seq:
                    time.sleep(0.01)
                    continue
                last_seq = current
                frame = slots[current % 2].copy()
                self.last_frame_at = time.time()

                # Face detection/tracking/recognition, fed off the same
                # frame the live view already reads — no second RTSP
                # connection. Submitted to _pipeline_executor (see __init__)
                # rather than called inline: this keeps a slow/backed-up
                # detection pass from ever delaying the video encode+
                # broadcast below. Skipped (not queued) if the previous call
                # is still running.
                # Switched off on the Analytics switches (analytics_settings.py):
                # skipped entirely, so no CPU goes to face recognition or boxes.
                if (analytics and analytics_settings.enabled("face_recognition") and not self._face_pipeline_failed
                        and (self._pipeline_future is None or self._pipeline_future.done())):
                    self._pipeline_future = self._pipeline_executor.submit(
                        self._feed_pipeline, frame, self.has_real_viewer() and analytics_settings.enabled("live_overlay")
                    )

                # Unique footfall (gate cameras only — no-op otherwise).
                # Hands off to its own executor, same reasoning as above.
                footfall.service.feed(self.camera_id, frame)
                # Restricted-zone intrusion (cameras with zones only).
                intrusion.service.feed(self.camera_id, frame)
                # Staff Count: entry-line crossings at entrance cameras only.
                staff_service.feed(self.camera_id, frame)

                # JPEG-encode once per distinct width actually being watched.
                # Full-HD JPEGs at 8 fps are ~17 Mbps per camera to each
                # browser (measured); a 960 px grid tile is ~4.4 Mbps and
                # looks the same at tile size. Nothing is encoded when only
                # keep-alives (analytics, footfall) are subscribed.
                with self._lock:
                    receivers = [(q, self._widths.get(q)) for q in self._subscribers if q not in self._frameless]
                encoded: dict = {}
                fh, fw = frame.shape[:2]
                for q, width in receivers:
                    key = width if width and width < fw else None
                    if key not in encoded:
                        img = frame if key is None else cv2.resize(frame, (key, round(fh * key / fw)), interpolation=cv2.INTER_AREA)
                        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80 if key is None else 72])
                        encoded[key] = buf.tobytes() if ok else None
                        if ok and key is None:
                            self.last_jpeg = encoded[key]
                    data = encoded[key]
                    if data is None:
                        continue
                    try:
                        q.put_nowait(data)
                    except Exception:
                        pass
        finally:
            if stop is not None:
                stop.set()
            if proc is not None:
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.terminate()
            slots = None
            if shm is not None:
                shm.close()


_streams: dict[int, CameraStream] = {}
_streams_lock = threading.Lock()


def get_stream(camera_id: int) -> CameraStream:
    with _streams_lock:
        stream = _streams.get(camera_id)
        if stream is None:
            stream = CameraStream(camera_id)
            _streams[camera_id] = stream
        return stream


class _StillSink:
    def __init__(self):
        self.data: bytes | None = None
        self.got = threading.Event()

    def put_nowait(self, item):
        self.data = item
        self.got.set()


def grab_still(camera_id: int, timeout: float = 12.0) -> bytes | None:
    """A fresh full-resolution JPEG from one camera (starting its stream
    briefly if it isn't running), for drawing desks and counting zones on.
    None if the camera sends nothing within `timeout`."""
    sink = _StillSink()
    stream = get_stream(camera_id)
    stream.subscribe(sink, is_collector=True)
    try:
        sink.got.wait(timeout=timeout)
    finally:
        stream.unsubscribe(sink)
    return sink.data
