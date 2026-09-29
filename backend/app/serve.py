"""
serve.py — the supported way to run the backend (development and production):

    python -m app.serve

Why not a bare `uvicorn app.main:app` command line:
  - exactly ONE worker, never --reload: camera readers, the capture limit
    lock, login rate limits and the in-memory staff/footfall state all
    assume a single process (DEPLOYMENT.md "Single process");
  - a bounded graceful shutdown: uvicorn otherwise waits FOREVER for open
    connections to close before running the app's shutdown, and a browser
    with a live-video tab keeps its websocket open indefinitely — so a
    restart hung until systemd killed the process and orphaned the camera
    readers. After GRACEFUL_SHUTDOWN_SECONDS open connections are cancelled
    and the lifespan shutdown (stop cameras, readers, workers) runs;
  - proxy headers trusted only from the local reverse proxy.
"""

from __future__ import annotations

import os

GRACEFUL_SHUTDOWN_SECONDS = int(os.environ.get("GRACEFUL_SHUTDOWN_SECONDS", "10"))


def build_config():
    import uvicorn

    from app import config

    return uvicorn.Config(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,
        reload=False,
        lifespan="on",
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
        proxy_headers=config.TRUST_PROXY_HEADERS,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        log_config=None,  # app.logging_setup owns logging (rotation, redaction)
        access_log=True,
    )


def main() -> None:
    import multiprocessing

    import uvicorn

    multiprocessing.freeze_support()
    uvicorn.Server(build_config()).run()


if __name__ == "__main__":
    main()
