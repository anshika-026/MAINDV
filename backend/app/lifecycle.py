"""
lifecycle.py

Process-wide shutdown signal for background loops. Every periodic loop
(alerts monitor, keep-alive syncs, retraining scheduler, collection expiry)
waits with `lifecycle.wait(seconds)` instead of time.sleep, so on SIGTERM /
SIGINT (uvicorn runs the FastAPI lifespan shutdown) they all return promptly
instead of being killed mid-iteration.
"""

import threading

_stopping = threading.Event()


def wait(seconds: float) -> bool:
    """Sleep up to `seconds`. True = shutting down, the caller should return."""
    return _stopping.wait(seconds)


def stopping() -> bool:
    return _stopping.is_set()


def begin_shutdown() -> None:
    _stopping.set()


def reset() -> None:
    """Tests only."""
    _stopping.clear()
