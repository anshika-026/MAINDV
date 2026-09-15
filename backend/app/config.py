import os
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
