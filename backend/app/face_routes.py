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
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import os
import uuid
from pathlib import Path

from app import face_db
from app.face_pipeline import CameraFacePipeline

router = APIRouter(prefix="/api/faces", tags=["faces"])

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
