"""
footfall_routes.py

Unique-footfall reporting across all entry gates (see footfall.py), plus
the UAT panel's reset / people / snapshot endpoints.

Mount with: app.include_router(footfall_routes.router) in main.py
"""

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import auth, camera_stream
from app.footfall import service

router = APIRouter(prefix="/api/footfall", tags=["footfall"])


@router.get("/summary")
def footfall_summary(_: dict = Depends(auth.require_admin)):
    """Today's unique people across every gate (each person once, whichever
    gate(s) they used), plus the per-gate breakdown, hourly arrivals vs
    yesterday, and today's visitor list."""
    return service.summary()


@router.get("/people")
def footfall_people(_: dict = Depends(auth.require_admin)):
    """Every unique person currently counted, newest first, with snapshot ids."""
    return service.people()


@router.get("/snapshots/{snapshot_id}")
def footfall_snapshot(snapshot_id: int, _: dict = Depends(auth.require_admin)):
    path = service.snapshot_path(snapshot_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/reset")
def footfall_reset(_: dict = Depends(auth.require_admin)):
    """UAT: restart the unique count from zero. Deletes every Re-ID identity
    and snapshot; cameras, faces and attendance are untouched."""
    return service.reset()


class ZoneIn(BaseModel):
    # Polygon corners as [x, y] fractions of the frame (0..1); null = whole frame.
    roi: list[list[float]] | None = None


@router.get("/cameras/{camera_id}/frame")
def footfall_camera_frame(camera_id: int, _: dict = Depends(auth.require_admin)):
    """Latest still from a gate camera, for drawing its counting zone on."""
    jpeg = camera_stream.grab_still(camera_id)
    if jpeg is None:
        raise HTTPException(status_code=404, detail="No frame yet — is this camera streaming?")
    return Response(content=jpeg, media_type="image/jpeg")


@router.put("/cameras/{camera_id}/zone")
def footfall_set_zone(camera_id: int, payload: ZoneIn, _: dict = Depends(auth.require_admin)):
    """Only people whose body centre is inside this zone are counted at
    this gate — draw it over the doorway to keep seating areas out."""
    roi = payload.roi
    if roi is not None:
        if len(roi) < 3 or any(len(p) != 2 or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1) for p in roi):
            raise HTTPException(status_code=422, detail="Zone needs at least 3 [x, y] points between 0 and 1")
    service.set_zone(camera_id, roi)
    return {"camera_id": camera_id, "roi": service.get_zone(camera_id)}
