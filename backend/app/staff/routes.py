"""
routes.py

Staff Count API (see service.py / occupancy.py).

Mount with: app.include_router(staff.routes.router) in main.py; the two
websockets are registered on the app itself (register_websockets).
"""

import asyncio
import datetime

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app import auth, camera_db, employee_directory
from app.staff import service as svc_mod

router = APIRouter(prefix="/api/staff", tags=["staff"])
service = svc_mod.service

# Tenant scoping: an admin sees every camera; a client only sees Staff Count
# if their license includes attendance (the same feature the sidebar gates
# the page on) and only for visits/events on cameras assigned to them.
_staff_viewer = auth.require_feature("attendance")


def _scope(principal: dict) -> set[int] | None:
    return auth.allowed_camera_ids(principal)


def _in_scope(rows: list[dict], cams: set[int] | None) -> list[dict]:
    return rows if cams is None else [r for r in rows if r.get("camera_id") in cams]


def _today_start() -> float:
    return datetime.datetime.combine(datetime.date.today(), datetime.time()).timestamp()


def _with_names(rows: list[dict]) -> list[dict]:
    cams = {c["id"]: c["name"] for c in camera_db.list_cameras()}
    for r in rows:
        emp = employee_directory.get_employee(r.get("employee_id")) if r.get("employee_id") else None
        r["employee_name"] = emp["name"] if emp else None
        r["camera_name"] = cams.get(r.get("camera_id"))
        r["timestamp"] = datetime.datetime.fromtimestamp(r["ts"]).isoformat(timespec="seconds") if r.get("ts") else None
    return rows


@router.get("/count")
def staff_count(principal: dict = Depends(_staff_viewer)):
    return service.count(camera_ids=_scope(principal))


@router.get("/present")
def staff_present(principal: dict = Depends(_staff_viewer)):
    """Everyone inside now (named employees and anonymous people), plus
    employees who were in earlier today and have left."""
    cams = _scope(principal)
    m = service.manager
    rows = [{"visit_id": v["visit_id"], "employee_id": v.get("employee_id"), "status": "PRESENT",
             "entry_time": v["entry_time"], "last_seen": v["last_seen"], "camera_id": v.get("camera_id"),
             "confidence": v.get("confidence")} for v in list(m._present.values())]
    present_emps = {r["employee_id"] for r in rows if r["employee_id"]}
    rows += [dict(r, status="EXITED") for r in m.employees_today() if r["status"] == "EXITED" and r["employee_id"] not in present_emps]
    rows = _in_scope(rows, cams)
    for r in rows:
        r["ts"] = r["last_seen"]
    return _with_names(rows)


@router.get("/events")
def staff_events(event_type: str | None = None, limit: int = 200, principal: dict = Depends(_staff_viewer)):
    return _with_names(service.manager.events(event_type, max(1, min(limit, 1000)), camera_ids=_scope(principal)))


@router.get("/entries")
def staff_entries(principal: dict = Depends(_staff_viewer)):
    return _with_names(service.manager.events("ENTRY", 500, since=_today_start(), camera_ids=_scope(principal)))


@router.get("/exits")
def staff_exits(principal: dict = Depends(_staff_viewer)):
    return _with_names(service.manager.events("EXIT", 500, since=_today_start(), camera_ids=_scope(principal)))


@router.get("/status")
def staff_status(_: dict = Depends(auth.require_admin)):
    return service.status()


@router.get("/cameras")
def staff_cameras(_: dict = Depends(auth.require_admin)):
    configs = {c["camera_id"]: c for c in svc_mod.list_camera_configs()}
    return [{"camera_id": c["id"], "name": c["name"], "config": configs.get(c["id"])} for c in camera_db.list_cameras()]


class CameraConfigIn(BaseModel):
    enabled: bool = True
    line: list[list[float]] | None = None      # [[x1, y1], [x2, y2]] fractions of the frame
    inside_sign: int = 1                        # which side of the line is the office
    roi: list[list[float]] | None = None        # polygon or null = whole frame


def _valid_points(points, n_min, n_max=None):
    return (points is not None and len(points) >= n_min and (n_max is None or len(points) <= n_max)
            and all(len(p) == 2 and 0 <= p[0] <= 1 and 0 <= p[1] <= 1 for p in points))


@router.put("/cameras/{camera_id}/config")
def set_camera_config(camera_id: int, payload: CameraConfigIn, _: dict = Depends(auth.require_admin)):
    if camera_db.get_camera(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if payload.enabled and not _valid_points(payload.line, 2, 2):
        raise HTTPException(status_code=422, detail="Draw the entry line: two points across the doorway")
    if payload.roi is not None and not _valid_points(payload.roi, 3):
        raise HTTPException(status_code=422, detail="The office area needs at least 3 corner points")
    return svc_mod.set_camera_config(camera_id, payload.enabled, payload.line, payload.inside_sign, payload.roi)


class ResetIn(BaseModel):
    reason: str = "manual reset"


@router.post("/reset")
def reset_occupancy(payload: ResetIn, _: dict = Depends(auth.require_admin)):
    """Mark everyone as having left (logged as AUTO_EXIT), e.g. before a test run."""
    return {"closed": service.manager.auto_exit_all(reason=payload.reason)}


def _ws_scope(token: str | None) -> tuple[bool, set[int] | None]:
    """(allowed, camera scope) for a websocket token, re-evaluated on every
    push so a suspended license or revoked session stops the stream."""
    session = auth.get_session(token)
    if session is None:
        return False, None
    try:
        _staff_viewer(session)
        return True, _scope(session)
    except HTTPException:
        return False, None


def register_websockets(app) -> None:
    @app.websocket("/ws/staff")
    async def ws_staff(websocket: WebSocket, token: str | None = None):
        await websocket.accept()
        try:
            while True:
                allowed, cams = _ws_scope(token)
                if not allowed:
                    await websocket.close(code=4401)
                    return
                await websocket.send_json(service.count(camera_ids=cams))
                await asyncio.sleep(2)
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/staff/debug/{camera_id}")
    async def ws_staff_debug(websocket: WebSocket, camera_id: int, token: str | None = None):
        # Admin-only diagnostic (raw tracks, line geometry, per-frame state).
        await websocket.accept()
        session = auth.get_session(token)
        if session is None or session["principal_type"] != auth.ADMIN:
            await websocket.close(code=4403)
            return
        try:
            while True:
                if auth.get_session(token) is None:
                    await websocket.close(code=4401)
                    return
                await websocket.send_json(service.debug(camera_id))
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            pass
