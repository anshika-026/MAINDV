"""
uploads.py

Safe handling of user-supplied images (face enrollment, gallery sync).

Rules, all enforced here so no route builds a filesystem path itself:
  - Identifiers that end up in a filename are validated against a strict
    allowlist (letters, digits, '-' and '_'), never sanitised-and-hoped.
  - The client's filename and extension are ignored entirely. The bytes are
    decoded with OpenCV and must be a real JPEG/PNG/WebP image; what is
    written to disk is OpenCV's own re-encode as .jpg, so nothing the
    client sent (polyglot payloads, EXIF, scripts) is stored verbatim.
  - The final path is resolved and must stay inside the intended directory.
  - Files are created with O_EXCL under a server-generated UUID name, so an
    upload can never overwrite an existing file.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import numpy as np

from . import config

# Employee / person ids as used across the app ("018", "EMP-2201", "sonam_k").
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# Magic numbers of the formats we accept. Checked before decoding so an
# obviously wrong file (PDF, script, zip) is rejected without handing it to
# the image decoder at all.
_SIGNATURES = (
    b"\xff\xd8\xff",          # JPEG
    b"\x89PNG\r\n\x1a\n",     # PNG
)
_WEBP = (b"RIFF", b"WEBP")    # RIFF....WEBP

MAX_PIXELS = 40_000_000       # decompression-bomb guard (~8K x 5K)


class UploadError(ValueError):
    """Carries an HTTP status for the route to return."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def validate_identifier(value: str, what: str = "person_id") -> str:
    if not isinstance(value, str):
        raise UploadError(f"Invalid {what}", 422)
    if value != value.strip() or not _ID_RE.match(value):
        raise UploadError(f"Invalid {what}: use 1-64 letters, digits, '-' or '_'", 422)
    return value


def _looks_like_image(data: bytes) -> bool:
    if any(data.startswith(sig) for sig in _SIGNATURES):
        return True
    return len(data) >= 12 and data[:4] == _WEBP[0] and data[8:12] == _WEBP[1]


def decode_image(data: bytes, max_bytes: int | None = None):
    """Validated BGR ndarray from raw upload bytes, or UploadError."""
    import cv2

    max_bytes = max_bytes or config.MAX_UPLOAD_BYTES
    if not data:
        raise UploadError("The uploaded file is empty", 400)
    if len(data) > max_bytes:
        raise UploadError(f"The image is larger than {max_bytes // (1024 * 1024)} MB", 413)
    if not _looks_like_image(data):
        raise UploadError("Only JPEG, PNG or WebP images are accepted", 415)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.shape[0] < 16 or img.shape[1] < 16:
        raise UploadError("Could not decode image", 400)
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise UploadError("The image dimensions are too large", 413)
    return img


def safe_child(directory: str | Path, filename: str) -> Path:
    """`directory/filename`, guaranteed to resolve inside `directory`."""
    base = Path(directory).resolve()
    if not filename or "/" in filename or "\\" in filename or "\x00" in filename or filename in (".", ".."):
        raise UploadError("Invalid file name", 400)
    target = (base / filename).resolve()
    if target.parent != base:
        raise UploadError("Invalid file name", 400)
    return target


def save_image(img, directory: str | Path, prefix: str) -> Path:
    """Re-encodes `img` as JPEG to a new, never-existing file under
    `directory` named <prefix>_<uuid>.jpg. `prefix` must already be a
    validated identifier."""
    import cv2

    validate_identifier(prefix, "prefix")
    Path(directory).mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise UploadError("Could not store image", 500)
    target = safe_child(directory, f"{prefix}_{uuid.uuid4().hex}.jpg")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(target, flags, 0o640)
    try:
        view = memoryview(buf.tobytes())
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)
    return target
