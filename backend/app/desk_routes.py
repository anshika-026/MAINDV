"""
desk_routes.py

Desk analytics (workforce insights): desk outlines per camera, a live still
to draw them on, and the daily per-employee desk-time report. See desks.py.

Mount with: app.include_router(desk_routes.router) in main.py
"""

import datetime

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from app import auth, camera_db, camera_stream, desk_db
from app.desks import service

router = APIRouter(prefix="/api", tags=["desks"])


@router.get("/cameras/{camera_id}/frame")
def camera_frame(camera_id: int, _: dict = Depends(auth.require_admin)):
    """Latest still from any camera. If it isn't streaming yet, starts it
    for a few seconds to get one."""
    if camera_db.get_camera(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    jpeg = camera_stream.grab_still(camera_id)
    if jpeg is None:
        raise HTTPException(status_code=504, detail="The camera didn't send a picture. Check it on Live Feed.")
    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/desk-zones")
def list_desk_zones(camera_id: int | None = None, _: dict = Depends(auth.require_admin)):
    return desk_db.list_zones(camera_id)


class DeskZoneIn(BaseModel):
    camera_id: int
    polygon: list[list[float]]  # [[x, y], ...] fractions of the frame
    label: str | None = None


@router.post("/desk-zones")
def create_desk_zone(payload: DeskZoneIn, _: dict = Depends(auth.require_admin)):
    if camera_db.get_camera(payload.camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if len(payload.polygon) < 3 or any(len(p) != 2 or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1) for p in payload.polygon):
        raise HTTPException(status_code=422, detail="A desk needs at least 3 corner points inside the picture")
    zone_id = desk_db.create_zone(payload.camera_id, payload.polygon, (payload.label or "").strip() or None)
    service.refresh_zones()
    return desk_db.get_zone(zone_id)


@router.delete("/desk-zones/{zone_id}")
def delete_desk_zone(zone_id: int, _: dict = Depends(auth.require_admin)):
    desk_db.delete_zone(zone_id)
    service.refresh_zones()
    return {"ok": True}


@router.get("/desk-analytics/report")
def desk_report(date: str | None = None, _: dict = Depends(auth.require_admin)):
    day = date or datetime.date.today().isoformat()
    try:
        datetime.date.fromisoformat(day)
    except ValueError:
        raise HTTPException(status_code=422, detail="Dates must look like 2026-09-28")
    return service.report(day)
