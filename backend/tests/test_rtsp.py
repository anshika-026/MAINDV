"""RTSP reader timeouts/backoff (app/rtsp_reader.py) and the camera watchdog
(app/camera_stream.py). No real camera: the reader runs against a scripted
fake capture, and the watchdog against a real child process that hangs."""

import multiprocessing
import queue
import threading
import time

import numpy as np
import pytest

from app import rtsp_reader

FAST = {"connect_timeout": 1, "read_timeout": 1, "reconnect_delay": 0.01, "reconnect_max_delay": 0.05}


class FakeCap:
    """One VideoCapture session: `frames` successful grabs, then a read
    failure (network drop). corrupt: indices whose retrieve() is garbage."""

    def __init__(self, opened=True, frames=10**9, corrupt=()):
        self.opened, self.frames, self.corrupt = opened, frames, set(corrupt)
        self.grabs = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def grab(self):
        self.grabs += 1
        return self.grabs <= self.frames

    def retrieve(self):
        if self.grabs in self.corrupt:
            return True, None
        return True, np.full((24, 32, 3), self.grabs % 255, np.uint8)

    def get(self, prop):
        return 0.0

    def release(self):
        self.released = True


class Seq:
    value = 0


def run_reader(sessions, until, timeout=10, fps=1000):
    """Runs rtsp_reader.run in a thread with `sessions` as successive
    open_capture() results; stops once until(seq, messages) is true."""
    caps = iter(sessions)
    opened = []

    def fake_open(url, ct, rt):
        opened.append((ct, rt))
        try:
            return next(caps)
        except StopIteration:
            return FakeCap(opened=False)

    stop, seq, q = threading.Event(), Seq(), queue.Queue()
    msgs = []
    orig = rtsp_reader.open_capture
    rtsp_reader.open_capture = fake_open
    t = threading.Thread(target=rtsp_reader.run, args=("rtsp://u:p@cam/x", fps, stop, seq, q, FAST), daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                msgs.append(q.get(timeout=0.02))
            except queue.Empty:
                pass
            if until(seq, msgs):
                break
        else:
            raise AssertionError(f"condition not reached; seq={seq.value} msgs={msgs[-10:]}")
    finally:
        stop.set()
        t.join(timeout=5)
        rtsp_reader.open_capture = orig
        while True:
            try:
                msgs.append(q.get_nowait())
            except queue.Empty:
                break
    assert not t.is_alive(), "reader did not stop when asked"
    return seq, msgs, opened


def statuses(msgs):
    return [m[1] for m in msgs if m[0] == "status"]


def test_camera_available_publishes_frames_with_timeouts_set():
    seq, msgs, opened = run_reader([FakeCap()], until=lambda s, m: s.value >= 5)
    assert seq.value >= 5
    assert opened[0] == (1, 1)                 # connect/read timeouts passed to FFmpeg
    assert "connected" in statuses(msgs)
    assert any(m[0] == "shm" for m in msgs)


def test_camera_unavailable_or_wrong_credentials_backs_off_and_keeps_trying():
    seq, msgs, opened = run_reader([FakeCap(opened=False)] * 6 + [FakeCap()], until=lambda s, m: s.value >= 1)
    recon = [m[2] for m in msgs if m[0] == "status" and m[1] == "reconnecting"]
    assert [r["attempt"] for r in recon[:6]] == [1, 2, 3, 4, 5, 6]
    assert all(r["delay"] <= FAST["reconnect_max_delay"] * 1.2 + 0.01 for r in recon)
    assert len(opened) == 7 and seq.value >= 1


def test_network_interruption_reconnects_and_resumes():
    first, second = FakeCap(frames=3), FakeCap()
    seq, msgs, _ = run_reader([first, second], until=lambda s, m: s.value >= 6)
    assert first.released and second.grabs > 0
    assert "reconnecting" in statuses(msgs)
    assert any(m[0] == "log" and "connected after" in m[1] for m in msgs)


def test_corrupted_frames_are_skipped_without_crashing():
    seq, msgs, _ = run_reader([FakeCap(corrupt=range(1, 20))], until=lambda s, m: s.value >= 3)
    assert seq.value >= 3


def test_reader_never_logs_the_camera_url():
    _, msgs, _ = run_reader([FakeCap(opened=False), FakeCap()], until=lambda s, m: s.value >= 1)
    assert not any("rtsp://" in str(m) or "u:p@" in str(m) for m in msgs)


def test_reconnect_delay_is_exponential_capped_and_jittered():
    assert rtsp_reader.reconnect_delay(1, 2, 60, rand=lambda: 0.5) == pytest.approx(2)
    assert rtsp_reader.reconnect_delay(4, 2, 60, rand=lambda: 0.5) == pytest.approx(16)
    assert rtsp_reader.reconnect_delay(20, 2, 60, rand=lambda: 0.5) == pytest.approx(60)
    assert rtsp_reader.reconnect_delay(20, 2, 60, rand=lambda: 1.0) == pytest.approx(72)
    assert rtsp_reader.reconnect_delay(20, 2, 60, rand=lambda: 0.0) == pytest.approx(48)


def test_heartbeats_are_sent_while_waiting_to_retry(monkeypatch):
    monkeypatch.setattr(rtsp_reader, "HEARTBEAT_SECONDS", 0.02)
    slow = {**FAST, "reconnect_delay": 0.2, "reconnect_max_delay": 0.2}
    stop, seq, q = threading.Event(), Seq(), queue.Queue()
    monkeypatch.setattr(rtsp_reader, "open_capture", lambda *a: FakeCap(opened=False))
    t = threading.Thread(target=rtsp_reader.run, args=("rtsp://x", 10, stop, seq, q, slow), daemon=True)
    t.start()
    time.sleep(0.5)
    stop.set()
    t.join(5)
    items = []
    while not q.empty():
        items.append(q.get())
    assert sum(1 for m in items if m[0] == "heartbeat") >= 3


# --- watchdog (parent side) ----------------------------------------------------

@pytest.fixture
def stream_env(monkeypatch):
    from app import camera_db, camera_stream, config

    monkeypatch.setattr(config, "CAMERA_FRAME_TIMEOUT", 1.0)
    monkeypatch.setattr(config, "CAMERA_WATCHDOG_INTERVAL", 0.2)
    monkeypatch.setattr(config, "RTSP_CONNECT_TIMEOUT", 0.5)
    monkeypatch.setattr(config, "RTSP_RECONNECT_DELAY", 0.1)
    monkeypatch.setattr(config, "RTSP_RECONNECT_MAX_DELAY", 0.2)
    cam = {"id": 901, "host": "10.9.9.9", "port": 554, "user": "u", "password": "p", "stream_path": "/x",
           "attendance_tracking": 0, "live_feed_enabled": 1, "status": "active"}
    monkeypatch.setattr(camera_db, "get_camera_connection", lambda cid: cam)
    monkeypatch.setattr(camera_db, "is_streamable", lambda c: True)
    yield camera_stream
    camera_stream.stop_all(timeout=5)
    camera_stream._streams.pop(901, None)


def _reader_children():
    return [p for p in multiprocessing.active_children() if p.name.startswith("rtsp-reader-") and p.is_alive()]


class Sink:
    def put_nowait(self, item):
        pass


def _wait(cond, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_watchdog_restarts_a_stalled_reader_and_shutdown_leaves_no_children(stream_env, monkeypatch):
    from tests import fake_readers

    monkeypatch.setattr(rtsp_reader, "run", fake_readers.hang)
    s = stream_env.get_stream(901)
    s.subscribe(Sink(), is_collector=True, wants_frames=False)
    assert _wait(lambda: s.watchdog_restarts >= 1, 30), s.status()
    assert s.state in ("STALLED", "CONNECTING", "RECONNECTING")
    stream_env.stop_all(timeout=10)
    assert not s._thread.is_alive()
    assert _wait(lambda: not _reader_children(), 10), _reader_children()


def test_a_crashing_reader_is_restarted_with_backoff(stream_env, monkeypatch):
    from tests import fake_readers

    monkeypatch.setattr(rtsp_reader, "run", fake_readers.exit_immediately)
    s = stream_env.get_stream(901)
    s.subscribe(Sink(), is_collector=True, wants_frames=False)
    # It keeps being restarted (not given up on), and the loop stays alive.
    time.sleep(3)
    assert s._thread.is_alive()
    stream_env.stop_all(timeout=10)
    assert _wait(lambda: not _reader_children(), 10)
