"""Stand-ins for rtsp_reader.run, importable by a spawned child process."""

import time


def hang(url, fps, stop, seq, meta_q, settings=None):
    """A reader stuck inside FFmpeg: never produces a frame or a heartbeat,
    and ignores the stop event (only terminate/kill can end it)."""
    while True:
        time.sleep(0.5)


def exit_immediately(url, fps, stop, seq, meta_q, settings=None):
    """A reader that crashes on start."""
    raise SystemExit(3)
