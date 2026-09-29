"""
logging_setup.py

Process-wide logging: timestamped lines on stderr, and (LOG_TO_FILE, on by
default in production) a size-rotated file under LOG_DIR so logs can't fill
the disk. A redaction filter scrubs credentials that could otherwise reach a
log line through an exception message or a URL: RTSP user:password@,
password= / token= query or form values, and Bearer tokens.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys

from . import config

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

_REDACTIONS = (
    (re.compile(r"(rtsps?://)([^:/@\s]+):([^@\s]+)@", re.I), r"\1\2:***@"),
    (re.compile(r"((?:password|passwd|pwd|token|secret|api_key)=)[^&\s'\"]+", re.I), r"\1***"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.I), r"\1***"),
)


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        clean = redact(msg)
        if clean != msg:
            record.msg, record.args = clean, None
        if record.exc_info and not record.exc_text:
            # Render the traceback now so it can be scrubbed too.
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


_configured = False


def configure() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if config.LOG_TO_FILE:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(
            config.LOG_DIR / "backend.log", maxBytes=config.LOG_FILE_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT, encoding="utf-8"))
    fmt = logging.Formatter(_FORMAT)
    redactor = RedactingFilter()
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(redactor)
        root.addHandler(h)
    # uvicorn's access log would otherwise print every websocket URL, which
    # carries the session token as ?token=.
    for name in ("uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.addFilter(redactor)
    # Chatty third-party loggers.
    for name in ("httpx", "urllib3", "PIL", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)
