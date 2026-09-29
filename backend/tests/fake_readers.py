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


def start_real_reader_then_hang(pid_file):
    """Middle process: starts a REAL rtsp_reader against an unreachable
    camera, records the reader's pid, then waits to be killed abruptly."""
    import multiprocessing
    import pathlib

    from app import rtsp_reader

    ctx = multiprocessing.get_context("spawn")
    stop, seq, q = ctx.Event(), ctx.Value("Q", 0, lock=False), ctx.Queue()
    settings = {"connect_timeout": 1, "read_timeout": 1, "reconnect_delay": 0.5, "reconnect_max_delay": 1}
    p = ctx.Process(target=rtsp_reader.run, args=("rtsp://127.0.0.1:9/none", 5, stop, seq, q, settings), daemon=False)
    p.start()
    pathlib.Path(pid_file).write_text(str(p.pid))
    while True:
        time.sleep(1)
