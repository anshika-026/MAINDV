"""
intrusion_routes.py

Restricted-area zones and intrusion stats (see intrusion.py). Detections
themselves surface as "Intrusion Detected" alerts on the Alerts page.

Mount with: app.include_router(intrusion_routes.router) in main.py
"""

import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app import auth, camera_db, intrusion

router = APIRouter(prefix="/api/intrusion", tags=["intrusion"])
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _check_window(start, end):
    if (start is None) != (end is None) or any(v is not None and not _HHMM.match(v) for v in (start, end)):
        raise HTTPException(status_code=422, detail="Active hours need both a start and an end, like 19:00 and 08:00")


@router.get("/zones")
def list_zones(camera_id: int | None = None, _: dict = Depends(auth.require_admin)):
    zones = intrusion.list_zones(camera_id)
    names = {c["id"]: c["name"] for c in camera_db.list_cameras()}
    st = intrusion.stats()
    for z in zones:
        z["camera_name"] = names.get(z["camera_id"], f"Camera {z['camera_id']}")
        z["active_now"] = intrusion.is_active(z)
        z["intrusions_today"] = st["per_zone_today"].get(z["id"], 0)
        z["last_intrusion"] = st["last_seen"].get(z["id"])
    return zones


class ZoneIn(BaseModel):
    camera_id: int
    name: str
    polygon: list[list[float]]
    active_from: str | None = None
    active_to: str | None = None


@router.post("/zones")
def create_zone(payload: ZoneIn, _: dict = Depends(auth.require_admin)):
    if camera_db.get_camera(payload.camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not payload.name.strip():
        raise HTTPException(status_code=422, detail="Give the zone a name")
    if len(payload.polygon) < 3 or any(len(p) != 2 or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1) for p in payload.polygon):
        raise HTTPException(status_code=422, detail="A zone needs at least 3 corner points inside the picture")
    _check_window(payload.active_from or None, payload.active_to or None)
    return intrusion.create_zone(payload.camera_id, payload.name.strip(), payload.polygon,
                                 payload.active_from or None, payload.active_to or None)


class ZoneUpdate(BaseModel):
    name: str | None = None
    enabled: bool | None = None
    active_from: str | None = None
    active_to: str | None = None


@router.patch("/zones/{zone_id}")
def update_zone(zone_id: int, payload: ZoneUpdate, _: dict = Depends(auth.require_admin)):
    fields = payload.model_dump(exclude_unset=True)
    if "active_from" in fields or "active_to" in fields:
        fields["active_from"] = fields.get("active_from") or None
        fields["active_to"] = fields.get("active_to") or None
        _check_window(fields["active_from"], fields["active_to"])
    zone = intrusion.update_zone(zone_id, **fields)
    if zone is None:
        raise HTTPException(status_code=404, detail="Zone not found")
    return zone


@router.delete("/zones/{zone_id}")
def delete_zone(zone_id: int, _: dict = Depends(auth.require_admin)):
    intrusion.delete_zone(zone_id)
    return {"ok": True}


@router.get("/stats")
def intrusion_stats(_: dict = Depends(auth.require_admin)):
    return intrusion.stats()
