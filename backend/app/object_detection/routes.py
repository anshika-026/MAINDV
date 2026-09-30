"""
routes.py — object detection API.

  GET /api/objects/latest   newest result per camera: detected objects
                            (class, confidence, bbox, track_id, color) and
                            per-class counts. Admin: every camera. Client:
                            only their licensed cameras, and only if the
                            license includes "object_detection".
  GET /api/objects/status   worker/model/recovery state (admin).
"""

from fastapi import APIRouter, Depends

from app import auth

from .service import service

router = APIRouter(prefix="/api/objects", tags=["object-detection"])

_viewer = auth.require_feature("object_detection")


@router.get("/latest")
def latest_objects(principal: dict = Depends(_viewer)):
    """Newest detection result per camera (backpack, handbag, bottle, laptop).
    A result older than a few detection intervals is flagged `stale`."""
    return {"enabled": service.enabled(), "cameras": service.latest(auth.allowed_camera_ids(principal))}


@router.get("/status")
def object_detection_status(_: dict = Depends(auth.require_admin)):
    return service.status()
