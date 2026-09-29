import asyncio
import contextlib
import logging
import os
import time

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

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import cv2

from . import config, logging_setup, models

logging_setup.configure()
# Before any ML library is imported below: no runtime model downloads in
# offline mode (see models.py).
models.apply_offline_environment()

from . import audit, auth, camera_db, camera_stream, employee_directory, face_collection, face_db, face_pipeline, face_training_scheduler, license_db, ratelimit  # noqa: E402
from .staff import routes as staff_routes  # noqa: E402
from .staff.service import service as staff_service  # noqa: E402
from . import alerts, alerts_routes, analytics_routes, analytics_settings, intrusion, intrusion_routes, attendance, attendance_routes, desk_db, desk_routes, desks, face_routes, insights_routes, face_training_routes, footfall, footfall_routes, license_routes  # noqa: E402
from . import health_routes, lifecycle, person_detection, retention  # noqa: E402

log = logging.getLogger("main")

# Same oversubscription fix as the OMP/BLAS env vars above, for OpenCV's
# own internal parallelism (JPEG decode/encode, resize) — otherwise each of
# the 3 camera threads' cv2 calls also each try to claim every core.
cv2.setNumThreads(2)
try:
    import torch

    torch.set_num_threads(2)
except ImportError:
    pass

@contextlib.asynccontextmanager
async def lifespan(_app):
    # Startup is synchronous and ordered: refuse an unsafe production
    # configuration, create/migrate tables, then start background services.
    check_configuration()
    init_databases()
    start_services()
    log.info("startup complete (APP_ENV=%s)", config.APP_ENV)
    try:
        yield
    finally:
        shutdown()


app = FastAPI(
    lifespan=lifespan,
    title="Deco Vision API",
    docs_url="/docs" if config.ENABLE_API_DOCS else None,
    redoc_url=None,
    openapi_url="/openapi.json" if config.ENABLE_API_DOCS else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Largest body any endpoint legitimately takes: one image upload plus
# multipart overhead. Checked against Content-Length before the body is
# read, so an oversized upload is refused without being buffered.
_MAX_BODY_BYTES = config.MAX_UPLOAD_BYTES + 256 * 1024


@app.middleware("http")
async def _limits_and_headers(request: Request, call_next):
    length = request.headers.get("content-length")
    if length is not None:
        try:
            too_big = int(length) > _MAX_BODY_BYTES
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
        if too_big:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    # Full traceback to the log, nothing internal to the caller.
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse({"detail": "Internal server error"}, status_code=500)

app.include_router(face_routes.router)
app.include_router(face_training_routes.router)
app.include_router(license_routes.router)
app.include_router(footfall_routes.router)
app.include_router(attendance_routes.router)
app.include_router(desk_routes.router)
app.include_router(insights_routes.router)
app.include_router(alerts_routes.router)
app.include_router(analytics_routes.router)
app.include_router(intrusion_routes.router)
app.include_router(staff_routes.router)
app.include_router(health_routes.router)
staff_routes.register_websockets(app)


def _enable_wal() -> None:
    """Attendance, desk analytics, footfall and face-training capture all
    write to data/app.db through the day, and in SQLite's default rollback
    mode a reader blocks writers (and vice versa) — which surfaced as
    "database is locked" on login. WAL lets reads proceed during a write.
    It's a persistent property of the database file, so this is a no-op
    after the first run."""
    from app import db

    conn = db.open_connection(camera_db.DB_PATH)
    try:
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        logging.getLogger("main").info("database journal mode: %s", mode)
    finally:
        conn.close()


def check_configuration() -> None:
    """Fatal in production, warnings elsewhere — see config.validate()."""
    problems = config.validate()
    for p in problems:
        (log.error if config.IS_PRODUCTION else log.warning)("configuration: %s", p)
    if problems and config.IS_PRODUCTION:
        raise RuntimeError("Refusing to start with an unsafe production configuration: " + "; ".join(problems))
    if config.JWT_SECRET_IS_EPHEMERAL:
        log.warning("configuration: JWT_SECRET not set; license QR codes will stop verifying after a restart")


def init_databases() -> None:
    _enable_wal()
    camera_db.init_db()
    analytics_settings.init_db()
    face_db.init_face_tables()
    license_db.init_db()
    auth.init_db()
    audit.init_db()
    auth.bootstrap_admin_from_env()
    if auth.count_admin_users() == 0:
        msg = "no admin account exists; create one with: python -m app.manage create-admin --email you@example.com"
        if config.IS_PRODUCTION:
            raise RuntimeError(msg)
        log.warning(msg)
    attendance.init_db()
    alerts.init_db()
    intrusion.init_db()
    desk_db.init_db()


def start_services() -> None:
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
    # Unique footfall across entry gates — keeps every "Entry/Exit" camera
    # streaming and counting, viewer or not. See footfall.py.
    footfall.service.start()
    # Desk analytics + keeping face-recognition cameras streaming all day.
    desks.service.start()
    # Camera-offline / footfall-stopped alert monitor.
    alerts.start()
    intrusion.service.start()
    # Staff Count at entrance cameras (see app/staff/).
    staff_service.start()
    # Hourly cleanup of expired sessions, old captures/snapshots, backups (retention.py).
    retention.start()


def shutdown() -> None:
    """SIGTERM/SIGINT (uvicorn runs the lifespan exit): stop taking new
    work, stop every camera read loop and its RTSP reader child process,
    stop inference executors and background loops. Nothing is left running
    and no orphan reader processes survive the backend."""
    log.info("shutdown: stopping background services")
    lifecycle.begin_shutdown()
    try:
        camera_stream.stop_all(timeout=10)
    except Exception:
        log.exception("shutdown: stopping camera streams failed")
    try:
        person_detection.service.shutdown()
    except Exception:
        log.exception("shutdown: stopping person detection failed")
    log.info("shutdown complete")


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
def create_camera(payload: CameraIn, principal: dict = Depends(auth.require_admin)):
    camera_id = camera_db.add_camera(payload.name, payload.site, **payload.model_dump(exclude={"name", "site"}))
    audit.record("camera.create", principal, target=f"camera:{camera_id}", name=payload.name)
    return camera_db.get_camera(camera_id)


class StreamTestIn(BaseModel):
    rtsp_url: str


@app.post("/api/cameras/test-stream")
def test_camera_stream(payload: StreamTestIn, _: dict = Depends(auth.require_admin)):
    """Connects to an RTSP link once and returns a single still (JPEG), so the
    Add Camera form can show the link works before it's saved. Runs in
    FastAPI's threadpool (plain def), so a slow camera never blocks others."""
    url = payload.rtsp_url.strip()
    if not url.lower().startswith("rtsp://"):
        raise HTTPException(status_code=422, detail="The link must start with rtsp://")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000])
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Couldn't connect. Check the address, port, username and password.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(status_code=400, detail="Connected, but the camera sent no video. Check the stream path.")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    finally:
        cap.release()
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.put("/api/cameras/{camera_id}")
def update_camera(camera_id: int, payload: CameraUpdate, principal: dict = Depends(auth.require_admin)):
    if camera_db.get_camera(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    fields = payload.model_dump(exclude_unset=True)
    camera_db.update_camera(camera_id, **fields)
    audit.record("camera.update", principal, target=f"camera:{camera_id}",
                 fields=sorted(k for k in fields if k != "password"), password_changed="password" in fields)
    if "live_feed_enabled" in fields:
        stream = camera_stream.get_stream(camera_id)
        stream.resume() if fields["live_feed_enabled"] else stream.stop()
        # Background keep-alives (footfall, desks, intrusion) pick it up now,
        # not in 15 s. Each is independent; one failing is logged and the
        # others still run (it will also catch up on its own periodic sync).
        for name, sync in (("footfall", footfall.service._sync_gates), ("desks", desks.service._sync),
                           ("intrusion", intrusion.service.sync), ("staff", staff_service.sync)):
            try:
                sync()
            except Exception:
                log.exception("camera %s: %s resync after feed change failed", camera_id, name)
    return camera_db.get_camera(camera_id)


@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: int, principal: dict = Depends(auth.require_admin)):
    camera_db.delete_camera(camera_id)
    audit.record("camera.delete", principal, target=f"camera:{camera_id}")
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
def list_sites(principal: dict = Depends(auth.get_principal)):
    # Sites aren't owned by a client, but each site lists its cameras, so a
    # client only sees their own licensed cameras inside it (and only the
    # sites that contain one) — never another tenant's camera names.
    sites = camera_db.list_sites()
    allowed = auth.allowed_camera_ids(principal)
    if allowed is None:
        return sites
    scoped = []
    for s in sites:
        cams = [c for c in s.get("cameras", []) if c.get("id") in allowed]
        if cams:
            scoped.append({**s, "cameras": cams, "active_count": sum(1 for c in cams if c.get("status") == "active")})
    return scoped


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


class SettingsIn(BaseModel):
    # Only known keys, each bounded: this used to accept (and store) any
    # JSON object from anyone, unauthenticated.
    model_config = ConfigDict(extra="forbid")
    detection_fps: float = Field(ge=0.1, le=30)


@app.get("/api/stats")
def get_stats(_: dict = Depends(auth.require_admin)):
    cameras = camera_db.list_cameras()
    return {
        "total_cameras": len(cameras),
        "active_cameras": sum(1 for c in cameras if c["status"] == "active"),
        "faces_enrolled": 0,
        "active_alerts": 0,
        "detections_today": 0,
    }


@app.get("/api/settings")
def get_settings(_: dict = Depends(auth.require_admin)):
    return _settings


@app.put("/api/settings")
def update_settings(payload: SettingsIn, principal: dict = Depends(auth.require_admin)):
    changes = payload.model_dump()
    _settings.update(changes)
    audit.record("settings.update", principal, target="settings", **changes)
    return _settings


class LoginIn(BaseModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=1024)


@app.post("/api/auth/login")
def login(payload: LoginIn, request: Request):
    """Admin login: email + password checked against admin_users
    (auth.authenticate_admin). Failures are throttled per IP and per email
    (ratelimit.py) and always get the same generic message, whether the
    email exists or not."""
    ip = ratelimit.client_ip(request)
    guard = ratelimit.admin_login_guard
    guard.check(ip, payload.email)
    admin = auth.authenticate_admin(payload.email, payload.password)
    if admin is None:
        guard.failed(ip, payload.email)
        raise HTTPException(status_code=401, detail="Invalid email or password")
    guard.succeeded(ip, payload.email)
    token = auth.create_admin_session(admin)
    audit.record("admin.login", {"email": admin["email"]}, ip=ip)
    return {"email": admin["email"], "name": admin["name"], "token": token}


@app.get("/api/auth/me")
def me(principal: dict = Depends(auth.require_admin)):
    admin = auth.get_admin_user(principal["admin_user_id"])
    if admin is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"email": admin["email"], "name": admin["name"]}


@app.post("/api/auth/logout")
def logout(principal: dict = Depends(auth.get_principal)):
    auth.revoke_session(principal["token"])
    return {"ok": True}


@app.get("/api/audit")
def audit_log(limit: int = 200, _: dict = Depends(auth.require_admin)):
    return audit.recent(limit)


# ---------------------------------------------------------------------------
# Live video: one RTSP-reading thread per camera (camera_stream.py), fanned
# out to every connected viewer over its own websocket.
# ---------------------------------------------------------------------------

# Websockets are long-lived: the session/license is re-validated this often,
# so a logout, password change or license suspension ends an open live view.
_WS_REAUTH_SECONDS = 30.0


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
    return auth.can_access_camera(auth.get_session(token), camera_id)


@app.websocket("/ws/live/{camera_id}")
async def ws_live(websocket: WebSocket, camera_id: int, token: str | None = None, plain: bool = False, w: int | None = None):
    await websocket.accept()
    if not _authorize_camera_ws(token, camera_id):
        await websocket.close(code=4401)
        return
    cam = camera_db.get_camera(camera_id)
    if not cam or not cam.get("host"):
        await websocket.close()
        return
    if not cam.get("live"):
        await websocket.close(code=4403, reason="Feed switched off")
        return

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)

    class ThreadSafePut:
        @staticmethod
        def put_nowait(item):
            loop.call_soon_threadsafe(_drop_oldest_and_put, queue, item)

    stream = camera_stream.get_stream(camera_id)
    # plain=1 (the plain Live Feed page, no boxes drawn) subscribes like a
    # background collector, so it doesn't make face_pipeline run the extra
    # person-overlay detection that only the AI Analytics view displays.
    # w: send frames scaled to this width (grid tiles ask for 960 px; see
    # camera_stream's encoding comment for the bandwidth this saves).
    stream.subscribe(ThreadSafePut, is_collector=plain, width=max(320, min(w, 3840)) if w else None)
    next_auth_check = loop.time() + _WS_REAUTH_SECONDS
    try:
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=_WS_REAUTH_SECONDS)
            except asyncio.TimeoutError:
                frame = None  # no video right now; still re-check the session below
            if loop.time() >= next_auth_check:
                next_auth_check = loop.time() + _WS_REAUTH_SECONDS
                if not await asyncio.to_thread(_authorize_camera_ws, token, camera_id):
                    await websocket.close(code=4401)
                    return
            if frame is not None:
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
    loop = asyncio.get_event_loop()
    next_auth_check = loop.time() + _WS_REAUTH_SECONDS
    try:
        while True:
            if loop.time() >= next_auth_check:
                next_auth_check = loop.time() + _WS_REAUTH_SECONDS
                if not await asyncio.to_thread(_authorize_camera_ws, token, camera_id):
                    await websocket.close(code=4401)
                    return
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
                        # face | appearance (named by body, appearance.py) | None
                        "identity_source": det.get("identity_source"),
                        # Additive fields — every existing consumer of this
                        # payload keeps working unchanged. Both are null/0
                        # when the expression model has nothing confident
                        # for this track, which never affects the name.
                        "expression": det.get("expression"),
                        "expression_confidence": det.get("expression_confidence", 0.0),
                    })
            # Boxes are in the camera's full-resolution pixels; the frame
            # size lets a viewer showing a scaled-down picture place them.
            fw, fh = pipeline.frame_size if pipeline is not None and pipeline.frame_size else (None, None)
            await websocket.send_json({"people": people, "fire_smoke": [], "frame_w": fw, "frame_h": fh})
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass


def _employee_display_name(employee_id: str) -> str:
    """Resolves a classifier's predicted employee_id to a real name from
    the SAME roster face-training validates labels against (face_db's
    employees table) — never invents a name; falls back to the bare ID if
    that roster doesn't (yet) have an entry for it.

    Cached for 30 s: this runs for every person in every overlay push
    (~3/s per open viewer), which used to be a full-table query each time."""
    global _names_cache, _names_cached_at
    now = time.monotonic()
    if now - _names_cached_at > 30:
        _names_cache = {e["employee_id"]: e["name"] for e in face_db.list_employees()}
        _names_cached_at = now
    return _names_cache.get(employee_id, employee_id)


_names_cache: dict[str, str] = {}
_names_cached_at = -1e9
