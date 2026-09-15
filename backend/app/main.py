import asyncio
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import camera_db, camera_stream, config, face_db
from . import face_routes, face_training_routes

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Deco Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(face_routes.router)
app.include_router(face_training_routes.router)


@app.on_event("startup")
def on_startup():
    camera_db.init_db()
    face_db.init_face_tables()


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

class CameraIn(BaseModel):
    name: str
    site: str
    cam_code: str = ""
    purpose: str = "GENERAL"
    host: str = ""
    port: int = 554
    user: str = ""
    password: str | None = None
    stream_path: str = "/h264/ch1/sub/av_stream"
    vendor: str = ""
    live_feed_enabled: bool = True


class CameraUpdate(BaseModel):
    name: str | None = None
    site: str | None = None
    cam_code: str | None = None
    purpose: str | None = None
    host: str | None = None
    port: int | None = None
    user: str | None = None
    password: str | None = None
    stream_path: str | None = None
    vendor: str | None = None
    live_feed_enabled: bool | None = None


@app.get("/api/cameras")
def list_cameras():
    return camera_db.list_cameras()


@app.post("/api/cameras")
def create_camera(payload: CameraIn):
    camera_id = camera_db.add_camera(payload.name, payload.site, **payload.model_dump(exclude={"name", "site"}))
    return camera_db.get_camera(camera_id)


@app.put("/api/cameras/{camera_id}")
def update_camera(camera_id: int, payload: CameraUpdate):
    if camera_db.get_camera(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    fields = payload.model_dump(exclude_unset=True)
    camera_db.update_camera(camera_id, **fields)
    return camera_db.get_camera(camera_id)


@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: int):
    camera_db.delete_camera(camera_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------

class SiteIn(BaseModel):
    name: str
    description: str = ""


class SiteUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


@app.get("/api/sites")
def list_sites():
    return camera_db.list_sites()


@app.post("/api/sites")
def create_site(payload: SiteIn):
    camera_db.add_site(payload.name, payload.description)
    return {"ok": True}


@app.put("/api/sites/{site_id}")
def update_site(site_id: int, payload: SiteUpdate):
    camera_db.update_site(site_id, name=payload.name, description=payload.description)
    return {"ok": True}


@app.delete("/api/sites/{site_id}")
def delete_site(site_id: int):
    camera_db.delete_site(site_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Dashboard support: stats / alerts / settings / auth
# ---------------------------------------------------------------------------

_settings = {"detection_fps": 1.0}


@app.get("/api/stats")
def get_stats():
    cameras = camera_db.list_cameras()
    return {
        "total_cameras": len(cameras),
        "active_cameras": sum(1 for c in cameras if c["status"] == "active"),
        "faces_enrolled": 0,
        "active_alerts": 0,
        "detections_today": 0,
    }


@app.get("/api/alerts")
def list_alerts(resolved: bool | None = None):
    return []


@app.post("/api/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int):
    return {"ok": True}


@app.get("/api/settings")
def get_settings():
    return _settings


@app.put("/api/settings")
def update_settings(payload: dict):
    _settings.update(payload)
    return _settings


class LoginIn(BaseModel):
    email: str


@app.post("/api/auth/login")
def login(payload: LoginIn):
    return {"email": payload.email, "name": payload.email.split("@")[0]}


# ---------------------------------------------------------------------------
# Live video: one RTSP-reading thread per camera (camera_stream.py), fanned
# out to every connected viewer over its own websocket.
# ---------------------------------------------------------------------------

@app.websocket("/ws/live/{camera_id}")
async def ws_live(websocket: WebSocket, camera_id: int):
    await websocket.accept()
    cam = camera_db.get_camera(camera_id)
    if not cam or not cam.get("host"):
        await websocket.close()
        return

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)

    class ThreadSafePut:
        @staticmethod
        def put_nowait(item):
            loop.call_soon_threadsafe(_drop_oldest_and_put, queue, item)

    stream = camera_stream.get_stream(camera_id)
    stream.subscribe(ThreadSafePut)
    try:
        while True:
            frame = await queue.get()
            await websocket.send_bytes(frame)
    except WebSocketDisconnect:
        pass
    finally:
        stream.unsubscribe(ThreadSafePut)


def _drop_oldest_and_put(queue: asyncio.Queue, item: bytes) -> None:
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        pass


@app.websocket("/ws/detections/{camera_id}")
async def ws_detections(websocket: WebSocket, camera_id: int):
    """No detection pipeline wired up yet — send empty frames on a steady
    interval so CameraTile's overlay socket connects cleanly instead of
    erroring, without claiming to detect anything."""
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({"faces": [], "fire_smoke": []})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
