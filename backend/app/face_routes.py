"""
face_routes.py

Mount with: app.include_router(face_routes.router) in main.py

Endpoints:
  GET  /api/faces/pending?hours=24   -> unresolved captures for the review UI
  POST /api/faces/assign             -> human assigns a pending capture to a person_id
                                         (this is the "training" step -> appends
                                         the embedding to that person's gallery)
  POST /api/faces/ignore             -> discard a pending capture (false positive etc.)
  POST /api/faces/enroll             -> directly add a photo for a person_id
                                         (initial enrollment, outside the review flow)
  GET  /api/faces/gallery/{person_id}/count -> how many reference embeddings a person has
  POST /api/faces/gallery/sync-from-identity -> embed the Identity page's enrolled
                                         photos into the local gallery, so enrolled
                                         people are recognisable live without needing
                                         enough captures to be a classifier class
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import uuid
from pathlib import Path

from app import auth, face_db
from app.face_pipeline import CameraFacePipeline

# Admin-only for every route in this router: this is the internal
# review-queue/enrollment tooling, not something the client portal calls
# (the client-facing "Identity" page reads from a separate external
# service — see BACKEND_HANDOFF.md). Previously none of this required
# any authentication at all, meaning anyone who could reach the backend
# could assign/ignore review captures or add arbitrary face embeddings.
router = APIRouter(prefix="/api/faces", tags=["faces"], dependencies=[Depends(auth.require_admin)])

# Resolved relative to this file, not cwd — see the note in face_pipeline.py
# (_DATA_DIR) for why a plain "backend/data/..." relative default is wrong
# for how this project actually launches uvicorn.
ENROLL_DIR = os.environ.get(
    "FACE_ENROLL_DIR", str(Path(__file__).resolve().parent.parent / "data" / "face_enroll")
)
os.makedirs(ENROLL_DIR, exist_ok=True)


class AssignRequest(BaseModel):
    pending_id: int
    person_id: str


class PersonIdOverrideRequest(BaseModel):
    name: str
    employee_id: str


class IgnoreRequest(BaseModel):
    pending_id: int


@router.get("/pending")
def list_pending(hours: int = 24, status: str = "pending"):
    return face_db.get_pending(status=status, hours=hours)


@router.post("/assign")
def assign(req: AssignRequest):
    try:
        face_db.assign_pending(req.pending_id, req.person_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True, "person_id": req.person_id}


@router.post("/ignore")
def ignore(req: IgnoreRequest):
    face_db.ignore_pending(req.pending_id)
    return {"ok": True}


@router.post("/enroll")
async def enroll(person_id: str = Form(...), photo: UploadFile = File(...)):
    """Direct enrollment path (e.g. from the People page's existing photo
    upload UI) — bypasses the review queue since the human is already
    confirming identity by uploading it against a specific person_id."""
    contents = await photo.read()
    ext = os.path.splitext(photo.filename or "")[1] or ".jpg"
    path = os.path.join(ENROLL_DIR, f"{person_id}_{uuid.uuid4().hex[:8]}{ext}")
    with open(path, "wb") as f:
        f.write(contents)

    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Only the embedding model is needed here — this photo is already a
    # framed, human-confirmed face, so there's no detection/tracking step
    # to run first. Deliberately not _ensure_models_loaded(), which would
    # also require the YOLO face-detection weights just to enroll a photo.
    CameraFacePipeline._ensure_arcface_loaded()
    faces = CameraFacePipeline._arcface.get(img)
    if not faces:
        raise HTTPException(status_code=422, detail="No face detected in photo")
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    face_db.add_embedding(person_id, face.normed_embedding.tolist(), source_image_path=path)
    return {"ok": True, "person_id": person_id, "total_embeddings": face_db.count_embeddings_for_person(person_id)}


@router.get("/gallery/{person_id}/count")
def gallery_count(person_id: str):
    return {"person_id": person_id, "count": face_db.count_embeddings_for_person(person_id)}


# ---------------------------------------------------------------------------
# Behavior Analytics — laptop-webcam demonstration page.
#
# The browser captures its own webcam and posts occasional single frames
# here; this runs the SAME detectors the RTSP pipeline already uses
# (CameraFacePipeline's class-level YOLO person/face models, shared and
# already loaded) plus the existing expression service. There is no new
# model, no new camera loop, and no RTSP connection involved — the live
# camera pipeline is completely untouched by this path.
#
# camera_id -1 is a sentinel for "the browser's webcam": it is never a real
# camera row and is deliberately NOT registered in face_pipeline's
# _pipelines dict, so nothing here can interfere with a real camera's
# stream, tracking or footfall.
# ---------------------------------------------------------------------------

WEBCAM_CAMERA_ID = -1
_webcam_pipeline = None
_webcam_pipeline_lock = __import__("threading").Lock()

# Guards one analysis at a time (see the drop-if-busy note in the endpoint)
# and holds the last completed result to hand back to a skipped request, so
# the page shows the most recent real detection rather than blanking.
_webcam_busy = False
_webcam_busy_lock = __import__("threading").Lock()
_webcam_last_result: dict = {"faces": 0, "detections": []}


def _get_webcam_pipeline():
    """One lazily-created pipeline instance used only for webcam frames.
    Its models are CameraFacePipeline class attributes, so this shares the
    exact weights the live cameras already loaded rather than loading a
    second copy."""
    global _webcam_pipeline
    with _webcam_pipeline_lock:
        if _webcam_pipeline is None:
            _webcam_pipeline = CameraFacePipeline(camera_id=WEBCAM_CAMERA_ID)
        return _webcam_pipeline


@router.post("/behavior/analyze")
async def behavior_analyze(frame: UploadFile = File(...)):
    """One webcam frame in, current detection state out.

    Expression is submitted to the existing service (drop-if-busy, so a
    CPU-heavy inference can never make this request hang) and the most
    recent completed result is returned. That means expression appears a
    beat after the first frame rather than blocking on it — deliberate,
    and the same throttling behaviour the live pipeline relies on.
    """
    import cv2
    import numpy as np


    contents = await frame.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode frame")

    from app import behavior_webcam

    h, w = img.shape[:2]

    # Drop-if-busy: this runs the emotion model and a face/landmark pass,
    # which on a box already saturated by the live RTSP cameras can take
    # seconds. Without this guard the page's polling would queue requests
    # behind each other and fall further and further behind. A skipped
    # frame simply returns the previous result.
    global _webcam_busy
    with _webcam_busy_lock:
        if _webcam_busy:
            return {**_webcam_last_result, "frame_width": w, "frame_height": h, "stale": True}
        _webcam_busy = True
    try:
        faces_out = behavior_webcam.analyze(img)
    finally:
        with _webcam_busy_lock:
            _webcam_busy = False

    result = {
        "faces": len(faces_out),
        "frame_width": int(w),
        "frame_height": int(h),
        "detections": faces_out,
        "stale": False,
    }
    _webcam_last_result.clear()
    _webcam_last_result.update(result)
    return result


# ---------------------------------------------------------------------------
# Identity people entered by hand (the Identity page's add/edit flow).
#
# These persist into the SAME employees table the rest of the app uses as
# its local roster — deliberately not a second employee system. The record
# is what survives a refresh/restart; the face images and embeddings behind
# it are saved separately by POST /enroll above, which is what makes such a
# person recognisable live through the enrollment-gallery fallback even
# when they have no classifier training data yet.
# ---------------------------------------------------------------------------

class IdentityPersonIn(BaseModel):
    employee_id: str
    name: str
    department: str | None = None
    person_type: str | None = None


@router.post("/people")
def save_identity_person(req: IdentityPersonIn):
    """Create or update one hand-entered person. Committed to the database
    before this returns, so a success response means the record is durable
    — the caller can treat a thrown error as "nothing was saved" and must
    not report success on its own."""
    employee_id = req.employee_id.strip()
    name = req.name.strip()
    if not employee_id:
        raise HTTPException(status_code=422, detail="Employee ID is required")
    if not name:
        raise HTTPException(status_code=422, detail="Name is required")

    saved = face_db.upsert_manual_person(
        employee_id=employee_id,
        name=name,
        department=(req.department or "").strip() or None,
        person_type=(req.person_type or "").strip() or None,
    )
    # Keeps the People page's name -> employee_id override consistent with
    # the record just saved, so the roster merge resolves this person to
    # the same ID the embeddings are stored under.
    face_db.set_person_employee_id(name, employee_id)
    saved["embedding_count"] = face_db.count_embeddings_for_person(employee_id)
    return saved


@router.get("/people")
def list_identity_people():
    """Every hand-entered person, read straight from the database — this is
    what makes them reappear after a refresh, a backend restart or a
    reboot, rather than living only in the page's React state."""
    people = face_db.list_manual_people()
    for p in people:
        p["photos"] = [
            {"id": e["id"], "enrolled_at": e["enrolled_at"]}
            for e in face_db.list_person_embeddings(p["employee_id"])
        ]
    return people


@router.get("/people/photo/{embedding_id}")
def identity_person_photo(embedding_id: int):
    """Serves back a saved enrollment image so the Identity page can still
    show a person's face samples after a refresh. Reads the file written by
    POST /enroll; 404 rather than an error if that file is gone, since the
    embedding itself (the part recognition actually uses) is still valid."""
    row = face_db.get_embedding_row(embedding_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such enrolled photo")
    path = row.get("source_image_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Enrolled image file is no longer on disk")
    return FileResponse(path)


# Same external enrollment service the People/Identity page reads (see
# client.js's FACES_API_BASE and face_training_routes.EXTERNAL_FACES_API)
# — the Identity roster and its reference photos live there, not in this
# repo's DB.
IDENTITY_SERVICE_BASE = os.environ.get("IDENTITY_SERVICE_BASE", "http://13.61.58.14")


@router.post("/gallery/sync-from-identity")
def sync_gallery_from_identity(force: bool = False, max_photos_per_person: int = 5):
    """Turns the people enrolled on the Identity page into local face
    embeddings, so they can be recognised live even when they have too few
    labelled camera captures to be one of the trained classifier's classes
    (see face_pipeline._identify_for_overlay, which consults this gallery
    whenever the classifier isn't confident).

    Deliberately reuses the existing enrollment storage rather than adding
    a second gallery: each photo is saved into ENROLL_DIR and embedded via
    the same ArcFace model and face_db.add_embedding() call that
    POST /api/faces/enroll already uses.

    Explicit-only (never automatic), idempotent, and additive: a person who
    already has embeddings is skipped unless force=true, and nothing is
    ever deleted — existing embeddings, captures and training data are
    untouched.
    """
    import json as _json
    import urllib.error
    import urllib.request

    import cv2
    import numpy as np

    try:
        with urllib.request.urlopen(f"{IDENTITY_SERVICE_BASE}/api/faces", timeout=15) as resp:
            roster = _json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"Could not reach the Identity enrollment service: {e}")

    # That service's own employee_id field is frequently null/stale, so a
    # local override keyed by name wins where one exists — same precedence
    # the People page itself uses.
    overrides = face_db.get_person_employee_id_overrides()
    CameraFacePipeline._ensure_arcface_loaded()

    people, skipped_no_employee_id, skipped_already_enrolled, photos_without_face = [], 0, 0, 0
    for row in roster:
        name = (row.get("name") or "").strip()
        employee_id = overrides.get(name) or row.get("employee_id")
        if not employee_id:
            skipped_no_employee_id += 1
            continue
        if not force and face_db.count_embeddings_for_person(employee_id) > 0:
            skipped_already_enrolled += 1
            continue

        added = 0
        for photo_path in (row.get("photo_urls") or [])[:max_photos_per_person]:
            url = f"{IDENTITY_SERVICE_BASE}{photo_path}"
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    contents = r.read()
            except (urllib.error.URLError, TimeoutError):
                continue
            img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            faces = CameraFacePipeline._arcface.get(img)
            if not faces:
                photos_without_face += 1
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            ext = os.path.splitext(photo_path)[1] or ".jpg"
            local_path = os.path.join(ENROLL_DIR, f"{employee_id}_{uuid.uuid4().hex[:8]}{ext}")
            with open(local_path, "wb") as f:
                f.write(contents)
            face_db.add_embedding(employee_id, face.normed_embedding.tolist(), source_image_path=local_path)
            added += 1

        if added:
            people.append({"employee_id": employee_id, "name": name, "embeddings_added": added})

    return {
        "enrolled_people": len(people),
        "embeddings_added": sum(p["embeddings_added"] for p in people),
        "skipped_already_enrolled": skipped_already_enrolled,
        "skipped_no_employee_id": skipped_no_employee_id,
        "photos_without_detectable_face": photos_without_face,
        "people": people,
    }


# ---------------------------------------------------------------------------
# People-page employee ID overrides — see face_db.py's table comment. The
# People page's roster itself (name/photos/enrollment) still comes live from
# the external face-enrollment service; this only persists the employee_id
# field, since that service has no write API this app can call.
# ---------------------------------------------------------------------------

@router.get("/people-id-overrides")
def get_people_id_overrides():
    return face_db.get_person_employee_id_overrides()


@router.post("/people-id-overrides")
def set_people_id_override(req: PersonIdOverrideRequest):
    face_db.set_person_employee_id(req.name, req.employee_id)
    return {"ok": True}
