"""
face_training_routes.py

Manual labeling pipeline for building a real, human-verified camera-capture
dataset: bulk collection -> label -> train. Deliberately separate from
face_routes.py (live review queue + direct enrollment), which is a
different workflow (correcting one low-confidence match, not building a
training set).

Mount with: app.include_router(face_training_routes.router) in main.py

Endpoints:
  GET  /api/faces/training/next               -> next unlabeled capture (metadata only) + counters
  GET  /api/faces/training/stats               -> {reviewed, total}
  GET  /api/faces/training/image/{capture_id}  -> the actual JPEG, path resolved server-side
  POST /api/faces/training/label               -> {capture_id, employee_id}
  POST /api/faces/training/skip                -> {capture_id}
  GET  /api/faces/training/employees           -> local roster (for reference/debugging)
  POST /api/faces/training/employees           -> add one manually {employee_id, name}
  POST /api/faces/training/employees/sync      -> one-time pull from the external faces API (explicit only)
  POST /api/faces/training/train               -> train the classifier on labeled captures
  GET  /api/faces/training/model-status        -> whether a trained classifier exists, and when
"""

import json
import os
import shutil
import urllib.error
import urllib.request

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import camera_db, face_db
from app.face_pipeline import CLASSIFIER_PATH, TRAINING_CAPTURE_DIR

router = APIRouter(prefix="/api/faces/training", tags=["face-training"])

# Not part of this repo — see BACKEND_HANDOFF.md. Only ever called explicitly
# via /employees/sync, never automatically, so the labeling/training
# pipeline never depends on this host being reachable at request time.
EXTERNAL_FACES_API = "http://13.61.58.14/api/faces"


class LabelRequest(BaseModel):
    capture_id: int
    employee_id: str


class SkipRequest(BaseModel):
    capture_id: int


class EmployeeIn(BaseModel):
    employee_id: str
    name: str


def _capture_public(row: dict) -> dict:
    cam = camera_db.get_camera(row["camera_id"])
    return {
        "id": row["id"],
        "camera_id": row["camera_id"],
        "camera_name": cam["name"] if cam else f"Camera {row['camera_id']}",
        "captured_at": row["captured_at"],
    }


@router.get("/next")
def next_capture():
    row = face_db.get_next_unlabeled_capture()
    stats = face_db.get_training_stats()
    return {"capture": _capture_public(row) if row else None, **stats}


@router.get("/stats")
def stats():
    return face_db.get_training_stats()


@router.get("/image/{capture_id}")
def capture_image(capture_id: int):
    # Path is looked up server-side from the capture id — the client never
    # supplies (or can supply) a filesystem path directly.
    row = face_db.get_training_capture(capture_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    if not os.path.exists(row["image_path"]):
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    return FileResponse(row["image_path"], media_type="image/jpeg")


@router.post("/label")
def label(req: LabelRequest):
    if not face_db.employee_exists(req.employee_id):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown employee_id '{req.employee_id}'. Add it first via "
                f"POST /api/faces/training/employees or /employees/sync."
            ),
        )
    row = face_db.get_training_capture(req.capture_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    if row["label_status"] != "unlabeled":
        raise HTTPException(status_code=409, detail=f"Capture already {row['label_status']}")

    dest_dir = os.path.join(TRAINING_CAPTURE_DIR, req.employee_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(row["image_path"]))
    shutil.move(row["image_path"], dest_path)

    try:
        face_db.label_training_capture(req.capture_id, req.employee_id, dest_path)
    except ValueError as e:
        # DB write failed after the file was already moved — move it back
        # so the capture doesn't end up orphaned (file relocated, row still
        # says unlabeled at the old path).
        shutil.move(dest_path, row["image_path"])
        raise HTTPException(status_code=409, detail=str(e))

    return {"ok": True}


@router.post("/skip")
def skip(req: SkipRequest):
    try:
        face_db.skip_training_capture(req.capture_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True}


@router.get("/employees")
def employees():
    return face_db.list_employees()


@router.post("/employees")
def add_employee(req: EmployeeIn):
    face_db.upsert_employee(req.employee_id, req.name)
    return {"ok": True}


@router.post("/employees/sync")
def sync_employees():
    """Explicit, re-runnable pull from the external face-enrollment service
    to seed the local roster — the only place that host is ever touched by
    the training pipeline, and only when you call this yourself."""
    try:
        with urllib.request.urlopen(EXTERNAL_FACES_API, timeout=10) as resp:
            rows = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"Could not reach external faces API: {e}")

    imported = 0
    skipped_no_id = 0
    for r in rows:
        employee_id = r.get("employee_id")
        if not employee_id:
            skipped_no_id += 1
            continue
        face_db.upsert_employee(employee_id, r.get("name", employee_id))
        imported += 1

    return {"imported": imported, "skipped_no_employee_id": skipped_no_id}


@router.post("/train")
def train():
    from app.face_training import train_classifier

    try:
        result = train_classifier()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


@router.get("/model-status")
def model_status():
    exists = os.path.exists(CLASSIFIER_PATH)
    return {
        "classifier_trained": exists,
        "path": CLASSIFIER_PATH if exists else None,
        "trained_at": os.path.getmtime(CLASSIFIER_PATH) if exists else None,
    }
