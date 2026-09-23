import asyncio
import logging
import os

# Must be set before numpy/torch/onnxruntime get imported (transitively, by
# face_pipeline below) to take effect. Each of the 3 concurrent camera
# threads runs its own YOLO/InsightFace inference calls against the same
# shared model instances (see face_pipeline.CameraFacePipeline's
# class-level _yolo/_yolo_person/_arcface); left at their library default,
# each call tries to use every CPU core, so 3 cameras running "at once"
# means 3x oversubscription fighting itself for the same 8 cores rather
# than 3 cameras actually running in parallel. Capping per-call threads
# lets the OS scheduler give each camera's thread a fair, non-thrashing
# share instead. Measured: this machine's backend process was pinned at
# ~540% CPU (of 8 cores) with live video down to ~0.4 fps against an 8 fps
# target before this change.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import cv2

from . import auth, camera_db, camera_stream, config, employee_directory, face_collection, face_db, face_pipeline, face_training_scheduler, license_db
from . import face_routes, face_training_routes, license_routes

logging.basicConfig(level=logging.INFO)

# Same oversubscription fix as the OMP/BLAS env vars above, for OpenCV's
# own internal parallelism (JPEG decode/encode, resize) — otherwise each of
# the 3 camera threads' cv2 calls also each try to claim every core.
cv2.setNumThreads(2)
try:
    import torch

    torch.set_num_threads(2)
except ImportError:
    pass

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
app.include_router(license_routes.router)


@app.on_event("startup")
def on_startup():
    camera_db.init_db()
    face_db.init_face_tables()
    license_db.init_db()
    auth.init_db()
    # Always start the expiry watcher (cheap, idempotent) so a session
    # started later still gets watched, then resume whatever collection
    # session was running before a restart (if its planned end time hasn't
    # already passed) — see face_collection.py for why this can never
    # create a duplicate worker per camera.
    face_collection.ensure_expiry_watcher_started()
    face_collection.resume_if_needed()
    # Periodic background retraining trigger (new labeled samples
    # accumulate -> auto-retrain) — separate daemon thread, never blocks
    # the camera pipelines or request handling. See
    # face_training_scheduler.py.
    face_training_scheduler.ensure_scheduler_started()


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
    attendance_tracking: bool = True


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
    attendance_tracking: bool | None = None


@app.get("/api/cameras")
def list_cameras(principal: dict = Depends(auth.get_principal)):
    """Tenant isolation lives here, not in the React UI: an admin
    principal sees every camera (unchanged), but a client principal only
    ever sees the cameras their own license has been assigned — enforced
    fresh against the DB on every call, not from anything the browser
    claims about itself. Previously this endpoint had no authentication
    at all and returned every camera to anyone who called it."""
    cameras = camera_db.list_cameras()
    if principal["principal_type"] == auth.CLIENT:
        lic = auth.load_active_client_license(principal)
        allowed_ids = {c["id"] for c in license_db.list_cameras_for_license(lic["id"])}
        cameras = [c for c in cameras if c["id"] in allowed_ids]
    return cameras


@app.post("/api/cameras")
def create_camera(payload: CameraIn, _: dict = Depends(auth.require_admin)):
    camera_id = camera_db.add_camera(payload.name, payload.site, **payload.model_dump(exclude={"name", "site"}))
    return camera_db.get_camera(camera_id)


@app.put("/api/cameras/{camera_id}")
def update_camera(camera_id: int, payload: CameraUpdate, _: dict = Depends(auth.require_admin)):
    if camera_db.get_camera(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    fields = payload.model_dump(exclude_unset=True)
    camera_db.update_camera(camera_id, **fields)
    return camera_db.get_camera(camera_id)


@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: int, _: dict = Depends(auth.require_admin)):
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
def list_sites(_: dict = Depends(auth.get_principal)):
    # Sites aren't part of the per-license assignment model (unlike
    # cameras) — any authenticated principal can list them, but only an
    # admin can create/edit/delete one (below). No per-tenant filtering
    # exists here because a "site" isn't owned by a client in this app's
    # data model, only its cameras are.
    return camera_db.list_sites()


@app.post("/api/sites")
def create_site(payload: SiteIn, _: dict = Depends(auth.require_admin)):
    camera_db.add_site(payload.name, payload.description)
    return {"ok": True}


@app.put("/api/sites/{site_id}")
def update_site(site_id: int, payload: SiteUpdate, _: dict = Depends(auth.require_admin)):
    camera_db.update_site(site_id, name=payload.name, description=payload.description)
    return {"ok": True}


@app.delete("/api/sites/{site_id}")
def delete_site(site_id: int, _: dict = Depends(auth.require_admin)):
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
    """LIMITATION, called out explicitly rather than left silent: this
    still does not check a password — it never has (see
    BACKEND_HANDOFF.md). Submitting any email logs in as an admin. What
    changed here is that a real, server-side session token is now issued
    and required on every subsequent admin-only request (auth.py) —
    previously NO token was checked at all, so this is a real
    improvement (no token -> no access), just not a substitute for actual
    admin authentication, which would need a real user/password table.
    That's a separate, larger piece of work than this change covers."""
    token = auth.create_admin_session(payload.email)
    return {"email": payload.email, "name": payload.email.split("@")[0], "token": token}


@app.post("/api/auth/logout")
def logout(principal: dict = Depends(auth.get_principal)):
    auth.revoke_session(principal["token"])
    return {"ok": True}


# ---------------------------------------------------------------------------
# Live video: one RTSP-reading thread per camera (camera_stream.py), fanned
# out to every connected viewer over its own websocket.
# ---------------------------------------------------------------------------

def _authorize_camera_ws(token: str | None, camera_id: int) -> bool:
    """Shared by both camera websockets below. Browsers can't attach an
    Authorization header to a WebSocket handshake, so the frontend passes
    the same bearer token as a `?token=` query param instead (see
    useLiveCameraFeed.js) — auth.get_session() is the same lookup
    get_principal() uses for HTTP requests, just invoked directly since
    there's no Header-based Depends() for this transport. Returns False
    (caller closes the socket) for: missing/invalid/expired token, a
    client whose license is no longer active, or a client whose license
    doesn't include this specific camera. An admin session may access any
    camera, matching the HTTP camera endpoints' behavior."""
    session = auth.get_session(token) if token else None
    if session is None:
        return False
    if session["principal_type"] == auth.ADMIN:
        return True
    try:
        lic = auth.load_active_client_license(session)
    except HTTPException:
        return False
    return license_db.is_camera_assigned(lic["id"], camera_id)


@app.websocket("/ws/live/{camera_id}")
async def ws_live(websocket: WebSocket, camera_id: int, token: str | None = None):
    await websocket.accept()
    if not _authorize_camera_ws(token, camera_id):
        await websocket.close(code=4401)
        return
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
async def ws_detections(websocket: WebSocket, camera_id: int, token: str | None = None):
    """Live PERSON overlay for LiveCameraTile/CameraViewerModal — reads
    face_pipeline.CameraFacePipeline.get_live_detections(), which is now
    person-track-based (full-body box, person-first — see
    _update_person_overlay): a person is listed here regardless of whether
    their face is currently visible/recognized. `employee_id`/`name` are
    only set once face recognition inside that person's box is confident
    AND temporally stable; otherwise the frontend shows "Person", never
    "Unknown". Populated by the same pipeline that already processes this
    camera's frames for /ws/live — no second inference path. fire_smoke
    stays [] — no such detector exists in this codebase. Authorized the
    same way as /ws/live."""
    await websocket.accept()
    if not _authorize_camera_ws(token, camera_id):
        await websocket.close(code=4401)
        return
    try:
        while True:
            pipeline = face_pipeline.get_existing_pipeline(camera_id)
            people = []
            if pipeline is not None:
                for det in pipeline.get_live_detections():
                    emp_id = det["employee_id"]
                    # Color is resolved from employee_id -> company via
                    # employee_directory.py, never guessed from the display
                    # name. This dict is rebuilt field-by-field from
                    # get_live_detections() rather than forwarded as-is, so
                    # "color" (and face_pipeline.py's own "name") were
                    # previously dropped here even though face_pipeline.py
                    # already computed them — that's what made every box
                    # render in the frontend's gray fallback regardless of
                    # employee_id.
                    _, color = employee_directory.get_display(emp_id)
                    people.append({
                        "track_id": det["track_id"],
                        "bbox": det["bbox"],
                        "employee_id": emp_id,
                        "name": _employee_display_name(emp_id) if emp_id else None,
                        "color": color,
                        "confidence": det["confidence"],
                    })
            await websocket.send_json({"people": people, "fire_smoke": []})
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass


def _employee_display_name(employee_id: str) -> str:
    """Resolves a classifier's predicted employee_id to a real name from
    the SAME roster face-training validates labels against (face_db's
    employees table) — never invents a name; falls back to the bare ID if
    that roster doesn't (yet) have an entry for it."""
    for e in face_db.list_employees():
        if e["employee_id"] == employee_id:
            return e["name"]
    return employee_id
