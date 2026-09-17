import os
import secrets
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8821"))

# Frontend dev server origins allowed to call this API / open websockets
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:5180,http://127.0.0.1:5180").split(",")
    if o.strip()
]

# How often the live-view websocket pushes a JPEG frame, independent of the
# camera's own frame rate — keeps bandwidth/CPU bounded regardless of source FPS.
LIVE_STREAM_FPS = float(os.environ.get("LIVE_STREAM_FPS", "8"))

# Used only to sign the license QR payload (license_qr.py) so a
# photographed/edited QR image can't silently claim a different license key
# — NOT used for user login/session tokens, since this app has no real
# auth/JWT session system yet (see BACKEND_HANDOFF.md). Generated once per
# process if not set explicitly, which is fine for this narrow use (existing
# QR images just need re-issuing if the process restarts without a fixed
# secret — set JWT_SECRET in backend/.env for a stable one across restarts).
JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
