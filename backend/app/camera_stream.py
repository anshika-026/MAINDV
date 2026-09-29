"""Pulls JPEG frames off a camera's RTSP stream in a background thread and
fans them out to however many websocket viewers are currently watching that
camera, so N browser tabs on the same camera share one RTSP connection
instead of each opening their own to the NVR.

Resilience (see rtsp_reader.py and resilience.py):
  - The RTSP reader child process has open/read timeouts and backs off
    between reconnects; this parent runs a watchdog that kills and restarts
    a reader that has stopped producing frames AND heartbeats for
    CAMERA_FRAME_TIMEOUT (hung inside FFmpeg), with backoff between restarts.
  - A face-recognition failure never switches recognition off for good: it
    is retried on a backoff, and the camera's pipeline is rebuilt after
    repeated failures. Video, footfall, staff count and intrusion keep
    running meanwhile — they don't depend on face recognition.
  - Stream state (RUNNING / CONNECTING / RECONNECTING / STALLED / FAILED /
    STOPPED) and last_frame_at are exposed for /ready and camera health.
"""

import concurrent.futures
import logging
import multiprocessing
import queue
import threading
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

from . import analytics_settings, camera_db, config, footfall, intrusion, person_detection, resilience, rtsp_reader  # noqa: F401 (footfall/intrusion register as consumers on import)
from .face_pipeline import discard_pipeline, get_pipeline
from .staff import service as _staff_service  # noqa: F401 (registers with person_detection on import)

log = logging.getLogger("camera_stream")

# Stream states reported by CameraStream.status().
S_IDLE = "IDLE"                  # no subscribers, not streaming
S_CONNECTING = "CONNECTING"
S_RUNNING = "RUNNING"
S_RECONNECTING = "RECONNECTING"  # reader is backing off after a failed open/read
S_STALLED = "STALLED"            # watchdog found a hung reader; restarting it
S_FAILED = "FAILED"              # RTSP_MAX_RETRIES consecutive failures (still retrying)
S_STOPPED = "STOPPED"            # feed switched off / not configured


def _reader_settings() -> dict:
    return {
        "connect_timeout": config.RTSP_CONNECT_TIMEOUT,
        "read_timeout": config.RTSP_READ_TIMEOUT,
        "reconnect_delay": config.RTSP_RECONNECT_DELAY,
        "reconnect_max_delay": config.RTSP_RECONNECT_MAX_DELAY,
    }


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
        self.face_health = resilience.health_for("face_recognition", camera_id)
        self.state = S_IDLE
        self.state_detail: str | None = None
        self.connect_failures = 0
        self.watchdog_restarts = 0
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

    def status(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "state": self.state,
            "detail": self.state_detail,
            "last_frame_at": self.last_frame_at,
            "connect_failures": self.connect_failures,
            "watchdog_restarts": self.watchdog_restarts,
            "subscribers": len(self._subscribers),
            "face_recognition": self.face_health.snapshot(),
        }

    def _feed_pipeline(self, frame, has_viewer: bool) -> None:
        """Runs on _pipeline_executor's worker thread, never on the video
        loop's own thread — see the submit() call in _run(). A failure here
        never takes down the video broadcast, and never switches face
        recognition off for good: face_health pauses it on a backoff and the
        pipeline is rebuilt after repeated failures."""
        try:
            # Created here, inside the try, rather than in _run(): the
            # constructor loads the models, and a missing checkpoint must
            # pause face recognition, not kill the video thread.
            if self._face_pipeline is None:
                self._face_pipeline = get_pipeline(self.camera_id)
            self._face_pipeline.feed_frame(frame, has_viewer=has_viewer)
        except Exception as e:
            if self.face_health.failure(e):
                # Fresh tracker/temporal state next time; shared models that
                # failed to load are retried by the new pipeline.
                discard_pipeline(self.camera_id)
                self._face_pipeline = None
        else:
            self.face_health.success()

    def _handle_reader_message(self, msg) -> None:
        kind = msg[0]
        if kind == "log":
            log.info("camera %s: %s", self.camera_id, msg[1])
        elif kind == "status":
            state, info = msg[1], (msg[2] if len(msg) > 2 else {})
            if state == "connecting":
                if self.state not in (S_RECONNECTING, S_FAILED):
                    self.state = S_CONNECTING
            elif state == "connected":
                self.connect_failures = 0
                self.state, self.state_detail = S_RUNNING, None
            elif state == "reconnecting":
                self.connect_failures = int(info.get("attempt", self.connect_failures + 1))
                self.state_detail = info.get("reason")
                self.state = S_FAILED if self.connect_failures >= config.RTSP_MAX_RETRIES else S_RECONNECTING

    def _run(self) -> None:
        try:
            self._run_inner()
        except Exception:
            # Should never happen (everything inside is guarded), but a crash
            # here must be visible, and must not leave a stale RUNNING state.
            log.exception("camera %s: stream loop crashed", self.camera_id)
            self.state = S_FAILED
        finally:
            if self.state != S_FAILED:
                self.state = S_STOPPED if self._subscribers else S_IDLE

    def _run_inner(self) -> None:
        cam = camera_db.get_camera_connection(self.camera_id)
        if cam and not camera_db.is_streamable(cam):
            log.info("camera %s: feed switched off, not connecting", self.camera_id)
            self.state, self.state_detail = S_STOPPED, "feed switched off"
            return
        url = build_rtsp_url(cam) if cam else None
        if not url:
            log.warning("camera %s: no host configured, nothing to stream", self.camera_id)
            self.state, self.state_detail = S_STOPPED, "no host configured"
            return
        self.state, self.state_detail = S_CONNECTING, None
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
        # Watchdog: last time the reader showed any sign of life (a new
        # frame, a status message or a heartbeat). Monotonic clock.
        last_progress = time.monotonic()
        last_watchdog_check = 0.0
        restarts = 0  # consecutive reader restarts without a frame in between

        def start_reader():
            nonlocal proc, stop, seq, meta_q, last_seq, last_progress
            stop = ctx.Event()
            seq = ctx.Value("Q", 0, lock=False)
            meta_q = ctx.Queue(maxsize=256)
            last_seq = 0
            proc = ctx.Process(
                target=rtsp_reader.run, args=(url, config.LIVE_STREAM_FPS, stop, seq, meta_q, _reader_settings()),
                daemon=True, name=f"rtsp-reader-{self.camera_id}",
            )
            proc.start()
            # Opening can legitimately take up to the connect timeout.
            last_progress = time.monotonic() + config.RTSP_CONNECT_TIMEOUT

        def kill_reader(reason: str):
            nonlocal shm, slots
            if stop is not None:
                stop.set()
            if proc is not None:
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=3)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
            if shm is not None:
                slots = None
                shm.close()
                shm = None
            (log.info if reason == "stopped" else log.warning)("camera %s: reader %s", self.camera_id, reason)

        def restart_reader(reason: str) -> bool:
            """Backs off, then starts a new reader. False if asked to stop meanwhile."""
            nonlocal restarts
            restarts += 1
            kill_reader(reason)
            delay = resilience.backoff_delay(restarts, config.RTSP_RECONNECT_DELAY, config.RTSP_RECONNECT_MAX_DELAY)
            if self._stop.wait(delay):
                return False
            start_reader()
            return True

        try:
            start_reader()
            while not self._stop.is_set():
                while True:
                    try:
                        msg = meta_q.get_nowait()
                    except queue.Empty:
                        break
                    except (EOFError, OSError):
                        break
                    last_progress = max(last_progress, time.monotonic())
                    if msg[0] == "shm":
                        if shm is not None:
                            slots = None
                            shm.close()
                        shm = shared_memory.SharedMemory(name=msg[1])
                        slots = np.ndarray((2, *msg[2]), dtype=np.uint8, buffer=shm.buf)
                    else:
                        self._handle_reader_message(msg)

                if not proc.is_alive():
                    if not restart_reader(f"process exited (code {proc.exitcode}), restarting"):
                        break
                    continue

                now = time.monotonic()
                if now - last_watchdog_check >= config.CAMERA_WATCHDOG_INTERVAL:
                    last_watchdog_check = now
                    if now - last_progress > config.CAMERA_FRAME_TIMEOUT:
                        self.state, self.state_detail = S_STALLED, "no frames or heartbeat from reader"
                        self.watchdog_restarts += 1
                        if not restart_reader(f"stalled for {now - last_progress:.0f}s, restarting"):
                            break
                        continue

                current = seq.value
                if slots is None or current == last_seq:
                    time.sleep(0.01)
                    continue
                last_seq = current
                frame = slots[current % 2].copy()
                last_progress = time.monotonic()
                restarts = 0
                self.last_frame_at = time.time()
                if self.state != S_RUNNING:
                    self.state, self.state_detail = S_RUNNING, None

                # Face detection/tracking/recognition, fed off the same
                # frame the live view already reads — no second RTSP
                # connection. Submitted to _pipeline_executor (see __init__)
                # rather than called inline: this keeps a slow/backed-up
                # detection pass from ever delaying the video encode+
                # broadcast below. Skipped (not queued) if the previous call
                # is still running.
                # Switched off on the Analytics switches (analytics_settings.py):
                # skipped entirely, so no CPU goes to face recognition or boxes.
                if (analytics and analytics_settings.enabled("face_recognition")
                        and (self._pipeline_future is None or self._pipeline_future.done())
                        and self.face_health.allow()):
                    self._pipeline_future = self._pipeline_executor.submit(
                        self._feed_pipeline, frame, self.has_real_viewer() and analytics_settings.enabled("live_overlay")
                    )

                # Footfall, Staff Count and intrusion share ONE person detector
                # per camera (person_detection.py), run at the rate the most
                # demanding of them needs and only while one of them wants
                # this camera. Hands off to its own executor, same reasoning
                # as above. Guarded so a consumer bug can't stop the video.
                try:
                    person_detection.service.feed(self.camera_id, frame)
                except Exception:
                    log.exception("camera %s: person detection dispatch failed", self.camera_id)

                # JPEG-encode once per distinct width actually being watched.
                # Full-HD JPEGs at 8 fps are ~17 Mbps per camera to each
                # browser (measured); a 960 px grid tile is ~4.4 Mbps and
                # looks the same at tile size. Nothing is encoded when only
                # keep-alives (analytics, footfall) are subscribed. Full-size
                # frames use quality 70 rather than 80: at 1920x1080 the encode
                # and the ~350 KB/frame were real costs on a CPU-bound box, with
                # no visible difference on a live view.
                with self._lock:
                    receivers = [(q, self._widths.get(q)) for q in self._subscribers if q not in self._frameless]
                encoded: dict = {}
                fh, fw = frame.shape[:2]
                for q, width in receivers:
                    key = width if width and width < fw else None
                    if key not in encoded:
                        img = frame if key is None else cv2.resize(frame, (key, round(fh * key / fw)), interpolation=cv2.INTER_AREA)
                        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70 if key is None else 72])
                        encoded[key] = buf.tobytes() if ok else None
                        if ok and key is None:
                            self.last_jpeg = encoded[key]
                    data = encoded[key]
                    if data is None:
                        continue
                    try:
                        q.put_nowait(data)
                    except Exception:
                        # A full/closed viewer queue drops this frame for
                        # that viewer only (it gets the next one).
                        pass
        finally:
            kill_reader("stopped")


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


def all_status() -> list[dict]:
    with _streams_lock:
        streams = list(_streams.values())
    return [s.status() for s in streams]


def stop_all(timeout: float = 10.0) -> None:
    """Graceful shutdown: stop every read loop, which stops (then, if it
    doesn't exit, terminates/kills) its RTSP reader child process, and shut
    down the per-camera inference executors. Leaves no child processes."""
    with _streams_lock:
        streams = list(_streams.values())
    for s in streams:
        s._stop.set()
    deadline = time.monotonic() + timeout
    for s in streams:
        t = s._thread
        if t is not None and t.is_alive():
            t.join(timeout=max(0.1, deadline - time.monotonic()))
            if t.is_alive():
                log.warning("camera %s: stream thread did not stop within %.0fs", s.camera_id, timeout)
        s._pipeline_executor.shutdown(wait=False, cancel_futures=True)
    # Belt and braces: any reader child still alive is terminated.
    for child in multiprocessing.active_children():
        if child.name.startswith("rtsp-reader-"):
            child.terminate()
            child.join(timeout=2)
            if child.is_alive():
                child.kill()
