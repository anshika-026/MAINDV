"""
analytics_routes.py

On/off switches for each analytics feature plus live CPU use (see
analytics_settings.py).

Mount with: app.include_router(analytics_routes.router) in main.py
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app import analytics_settings, auth

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/settings")
def get_settings(_: dict = Depends(auth.get_principal)):
    from app import person_detection

    return {"features": analytics_settings.snapshot(), "cpu": analytics_settings.cpu_usage(),
            "person_detectors": person_detection.service.status()}


class SwitchIn(BaseModel):
    on: bool


@router.put("/settings/{feature}")
def set_feature(feature: str, payload: SwitchIn, _: dict = Depends(auth.require_admin)):
    try:
        analytics_settings.set_enabled(feature, payload.on)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown analytics feature {feature}")
    # Start/stop background streaming right away rather than at the next 15 s sync.
    from app import desks, footfall, intrusion
    from app.staff.service import service as staff_service

    for sync in (footfall.service._sync_gates, desks.service._sync, intrusion.service.sync, staff_service.sync):
        try:
            sync()
        except Exception:
            pass
    return {"features": analytics_settings.snapshot()}
