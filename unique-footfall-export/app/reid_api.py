"""HTTP API for the body-appearance Re-ID unique-footfall engine — a
separate FastAPI `APIRouter`, `include_router()`'d into main.py's `app`,
mirroring peopleid_api.py's own justification for being the exception to
main.py's flat single-file route list (this module's own endpoint count).

Same JWT-auth requirement as peopleid_api.py, for the same reason (spec
section 34: biometric-adjacent data, stricter than this app's
unauthenticated norm elsewhere) — even though identity here is a generated
PERSON_NNN label rather than an enrolled name, the underlying embeddings and
snapshot photos carry the same sensitivity.

Camera CRUD itself is NOT duplicated here — this module only adds a
per-camera Re-ID config (enabled/thresholds/ROI) on top of the EXISTING
camera registry (camera_db.py / main.py's /api/cameras), same reuse
decision peopleid_api.py already made for its own per-camera config.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from . import auth, camera_db, reid_db
from .pipeline import pipeline_manager

router = APIRouter(prefix="/api/reid", tags=["reid"])

WRITE_ROLES = (auth.ROLE_SUPER_ADMIN, auth.ROLE_COMPANY_ADMIN, auth.ROLE_OPERATOR)


class PersonUpdateIn(BaseModel):
    display_name: str | None = None
    status: str | None = None


class CameraConfigIn(BaseModel):
    enabled: bool
    similarity_threshold: float | None = None
    quality_min_score: float | None = None
    mot_interval_seconds: float | None = None
    roi: list[list[float]] | None = None


# --- Persons ----------------------------------------------------------------

@router.get("/persons")
def list_persons(
    status: str | None = "active",
    include_snapshots: bool = False,
    user: dict = Depends(auth.get_current_user),
):
    """include_snapshots attaches a few snapshot rows per person in ONE
    extra query (see reid_db.list_snapshots_bulk). Off by default so the
    plain list stays cheap; the acceptance-test screen turns it on because
    judging whether an identity is right means looking at the person."""
    persons = reid_db.list_persons(status=status)
    if include_snapshots and persons:
        by_person = reid_db.list_snapshots_bulk([p["id"] for p in persons])
        for person in persons:
            person["snapshots"] = by_person.get(person["id"], [])
    return persons


@router.get("/persons/{person_id}")
def get_person(person_id: int, user: dict = Depends(auth.get_current_user)):
    person = reid_db.get_person(person_id)
    if person is None:
        raise HTTPException(404, "Person not found")
    person["snapshots"] = reid_db.list_snapshots_for_person(person_id)
    return person


@router.put("/persons/{person_id}")
def update_person(person_id: int, payload: PersonUpdateIn, user: dict = Depends(auth.require_role(*WRITE_ROLES))):
    if reid_db.get_person(person_id) is None:
        raise HTTPException(404, "Person not found")
    reid_db.update_person(person_id, **payload.model_dump(exclude_none=True))
    return {"ok": True}


@router.delete("/persons/{person_id}")
def delete_person(person_id: int, user: dict = Depends(auth.require_role(*WRITE_ROLES))):
    if reid_db.get_person(person_id) is None:
        raise HTTPException(404, "Person not found")
    reid_db.delete_person(person_id)
    pipeline_manager.reload_reid_gallery()
    return {"ok": True}


@router.get("/persons/{person_id}/appearances")
def list_appearances(person_id: int, limit: int = 200, user: dict = Depends(auth.get_current_user)):
    if reid_db.get_person(person_id) is None:
        raise HTTPException(404, "Person not found")
    return reid_db.list_events(person_id=person_id, limit=limit)


@router.get("/snapshots/{filename}")
def get_snapshot(filename: str, user: dict = Depends(auth.get_current_user)):
    """Access-controlled (spec section 34), same path-traversal-safe
    pattern as peopleid_api.py's own /photos/{filename} — no public photo
    URLs for either module."""
    path = (reid_db.SNAPSHOTS_DIR / filename).resolve()
    if reid_db.SNAPSHOTS_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "Snapshot not found")
    return Response(content=path.read_bytes(), media_type="image/jpeg")


# --- Events -------------------------------------------------------------

@router.get("/events")
def list_events(
    camera_id: int | None = None, start_ts: float | None = None, end_ts: float | None = None,
    limit: int = 200, user: dict = Depends(auth.get_current_user),
):
    return reid_db.list_events(camera_id=camera_id, start_ts=start_ts, end_ts=end_ts, limit=limit)


# --- Unique footfall (spec sections 12/13/19) --------------------------------

@router.get("/footfall/today")
def footfall_today(user: dict = Depends(auth.get_current_user)):
    return {"unique_count": reid_db.count_unique_today(), "new_today": reid_db.count_new_today(), "returning_today": reid_db.count_returning_today()}


@router.get("/footfall/lifetime")
def footfall_lifetime(user: dict = Depends(auth.get_current_user)):
    return {"unique_count": reid_db.count_unique_lifetime()}


@router.get("/footfall/history")
def footfall_history(days: int = 14, user: dict = Depends(auth.get_current_user)):
    return reid_db.footfall_history(days=days)


@router.get("/stats")
def stats(user: dict = Depends(auth.get_current_user)):
    """Dashboard summary (spec section 19) — one call for every stat tile
    instead of the frontend making 4-5 separate requests. active_people:
    a track currently live (within 30s) on ANY camera, deduped by
    person_id — mirrors reid_db.list_active_tracks' own "recently seen"
    window, not a separate concept."""
    from . import config

    active_tracks = reid_db.list_active_tracks(within_seconds=30.0)
    active_person_ids = {t["person_id"] for t in active_tracks if t["person_id"] is not None}
    cameras = camera_db.list_cameras()
    camera_configs = {c["camera_id"]: c for c in reid_db.list_camera_configs()}
    camera_status = [
        {
            "camera_id": cam["id"], "name": cam["name"],
            "reid_enabled": camera_configs.get(cam["id"], {}).get("enabled", False),
            "live": pipeline_manager.is_live(cam["id"]),
        }
        for cam in cameras
    ]
    unique_today = reid_db.count_unique_today()
    unique_lifetime = reid_db.count_unique_lifetime()
    return {
        "unique_footfall": unique_today if config.REID_FOOTFALL_MODE == "daily" else unique_lifetime,
        "unique_footfall_today": unique_today,
        "unique_footfall_lifetime": unique_lifetime,
        "total_known_people": len(reid_db.list_persons(status=None)),
        "active_people": len(active_person_ids),
        "new_today": reid_db.count_new_today(),
        "returning_today": reid_db.count_returning_today(),
        "cameras": camera_status,
    }


# --- Per-camera config (incl. ROI — spec sections 15/29) --------------------

@router.get("/cameras/{camera_id}/config")
def get_camera_config(camera_id: int, user: dict = Depends(auth.get_current_user)):
    return reid_db.get_camera_config(camera_id)


@router.put("/cameras/{camera_id}/config")
def put_camera_config(camera_id: int, payload: CameraConfigIn, user: dict = Depends(auth.require_role(*WRITE_ROLES))):
    reid_db.set_camera_config(
        camera_id, payload.enabled, payload.similarity_threshold,
        payload.quality_min_score, payload.mot_interval_seconds, payload.roi,
    )
    pipeline_manager.restart_camera_worker(camera_id)
    return {"ok": True}


# --- Acceptance testing -----------------------------------------------------

@router.post("/reset")
def reset_identities(user: dict = Depends(auth.require_role(auth.ROLE_SUPER_ADMIN))):
    """Wipes every identity so a UAT run can start from a known-empty
    state — without it, an acceptance test can't tell a newly created
    identity from one left over from a previous run.

    Restarts each camera's worker afterwards: the worker holds the
    gallery as an in-process NumPy matrix, so one that keeps running
    would still be matching against identities that no longer exist and
    would write back rows referencing deleted people.

    super_admin only, and irreversible — it deletes the snapshot images
    too, not just the rows.
    """
    removed = reid_db.reset_identities()
    for camera in camera_db.list_cameras():
        if reid_db.get_camera_config(camera["id"]).get("enabled"):
            pipeline_manager.restart_camera_worker(camera["id"])
    return {"ok": True, "removed": removed}
