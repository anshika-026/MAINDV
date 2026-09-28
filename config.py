import os

from dotenv import load_dotenv

load_dotenv()

CAMERA_HOST = os.getenv("CAMERA_HOST", "")
CAMERA_RTSP_PORT = int(os.getenv("CAMERA_RTSP_PORT", "554"))
CAMERA_USER = os.getenv("CAMERA_USER", "")
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "")
CAMERA_STREAM_PATH = os.getenv("CAMERA_STREAM_PATH", "/h264/ch1/sub/av_stream")
CAMERA_ADMIN_PORT = int(os.getenv("CAMERA_ADMIN_PORT", "443"))

SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")

# =====================================================================
# SERVER PORT - MUST MATCH api.js
# =====================================================================
#
# This was 8811 while frontend/src/api.js had:
#
#     API_BASE = "http://127.0.0.1:18000"
#
# Two different numbers. Start the server the normal way and it listened
# on 8811 while the dashboard talked to 18000, so nothing connected -
# detections showed in the terminal and never on screen. The old
# ObjectDetection.jsx made it worse by hardcoding 8811 for websockets
# while REST went to 18000, so the two halves of the same page
# disagreed with each other.
#
# 18000 is now the default here, matching api.js.
#
# To use a different port, change it in BOTH places:
#
#   1. backend/.env          SERVER_PORT=9000
#   2. frontend/.env         VITE_API_BASE=http://127.0.0.1:9000
#
# Changing only one breaks the dashboard silently - the backend keeps
# working perfectly and the page just stays empty.
SERVER_PORT = int(os.getenv("SERVER_PORT", "18000"))

# Many RTSP cameras/NVRs never reply over UDP from behind NAT/firewalls,
# which makes OpenCV's ffmpeg backend fail to open the stream with no
# useful error - forcing TCP fixes that for the large majority of devices.
RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{RTSP_TRANSPORT}"


def actual_server_port():
    """
    The port this process is REALLY listening on.

    SERVER_PORT is only the default. Starting with

        uvicorn app.main:app --port 8811

    overrides it, and nothing in this file would know. An earlier version
    of the check compared api.js against SERVER_PORT and cheerfully
    reported "both on 18000" while uvicorn was actually on 8811 - a false
    all-clear on the exact problem it existed to catch.

    So read the real thing: the command line first, then SERVER_PORT.
    """
    import sys

    argv = sys.argv

    for index, item in enumerate(argv):
        if item == "--port" and index + 1 < len(argv):
            try:
                return int(argv[index + 1])
            except (TypeError, ValueError):
                pass
        elif item.startswith("--port="):
            try:
                return int(item.split("=", 1)[1])
            except (TypeError, ValueError):
                pass

    return SERVER_PORT


def check_port_matches_frontend():
    """
    Warn at startup if api.js is pointing somewhere else.

    Compares the port this process is ACTUALLY listening on against the
    one in frontend/src/api.js. Purely a warning - it never changes
    anything or stops the server.
    """
    from pathlib import Path
    import re

    server_port = actual_server_port()

    candidates = [
        Path(__file__).resolve().parent.parent.parent / "frontend" / "src" / "api.js",
        Path(__file__).resolve().parent.parent / "frontend" / "src" / "api.js",
    ]

    for path in candidates:
        if not path.is_file():
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        match = re.search(r"API_BASE\s*=.*?:(\d{2,5})", text)
        if not match:
            continue

        frontend_port = int(match.group(1))

        return {
            "match": frontend_port == server_port,
            "server_port": server_port,
            "frontend_port": frontend_port,
            "api_js": str(path),
            "from_command_line": server_port != SERVER_PORT,
        }

    return {"match": None, "server_port": server_port, "api_js": None}