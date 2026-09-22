"""
face_training_routes.py

Manual labeling pipeline for building a real, human-verified camera-capture
dataset: bulk collection -> label -> train. Deliberately separate from
face_routes.py (live review queue + direct enrollment), which is a
different workflow (correcting one low-confidence match, not building a
training set).

Mount with: app.include_router(face_training_routes.router) in main.py

Endpoints:
  GET  /api/faces/training/next                -> next unlabeled capture (metadata only) + counters
  GET  /api/faces/training/stats                -> {reviewed, total, all_captures, by_status}
  GET  /api/faces/training/image/{capture_id}   -> the actual JPEG, path resolved server-side
  POST /api/faces/training/label                -> {capture_id, employee_id}
  POST /api/faces/training/skip                 -> {capture_id}
  GET  /api/faces/training/employees            -> local roster (for reference/debugging)
  POST /api/faces/training/employees            -> add one manually {employee_id, name}
  POST /api/faces/training/employees/sync       -> one-time pull from the external faces API (explicit only)
  POST /api/faces/training/train                -> train the classifier on labeled captures
  GET  /api/faces/training/model-status         -> whether a trained classifier exists, and when
  POST /api/faces/training/collection/start      -> start/extend a (default 7-day) collection session
                                                     {camera_ids?: [int], days?: float} — omit both for
                                                     "all configured cameras, 7 days"
  POST /api/faces/training/collection/stop       -> {camera_id?: int} — one camera, or omit to end
                                                     the whole session and stop every camera
  GET  /api/faces/training/collection/status     -> full session + per-camera monitoring (see below)
"""

import json
import os
import shutil
import time
import urllib.error
import urllib.request

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import auth, camera_db, employee_directory, face_collection, face_db, face_training_scheduler
from app.face_pipeline import (
    CLASSIFIER_PATH,
    MAX_CAPTURES_PER_CAMERA,
    MAX_TRAINING_CAPTURES,
    TRAINING_CAPTURE_DIR,
    TRAINING_UNLABELED_DIR,
)

# A camera that's supposedly collecting but hasn't produced anything in this
# long isn't necessarily broken (could just be a quiet area), but it's
# worth surfacing in /collection/status rather than looking identical to a
# healthy, active camera.
STALLED_THRESHOLD_SECONDS = 2 * 3600

# Admin-only for every route: this is the manual-labeling/collection
# control plane (start/stop background collection, train the classifier,
# label real captures), not something the client portal touches. Was
# entirely unauthenticated before this change.
router = APIRouter(prefix="/api/faces/training", tags=["face-training"], dependencies=[Depends(auth.require_admin)])

# Not part of this repo — see BACKEND_HANDOFF.md. Only ever called explicitly
# via /employees/sync, never automatically, so the labeling/training
# pipeline never depends on this host being reachable at request time.
EXTERNAL_FACES_API = "http://13.61.58.14/api/faces"


class LabelRequest(BaseModel):
    capture_id: int
    employee_id: str


class SkipRequest(BaseModel):
    capture_id: int


class RelabelRequest(BaseModel):
    capture_id: int
    employee_id: str


class UnlabelRequest(BaseModel):
    capture_id: int


class EmployeeIn(BaseModel):
    employee_id: str
    name: str


class CollectionStartRequest(BaseModel):
    camera_ids: list[int] | None = None  # default: every currently-configured camera
    days: float | None = None            # default: DEFAULT_COLLECTION_DAYS (7)


class CollectionStopRequest(BaseModel):
    camera_id: int | None = None  # omit to stop everything and end the session


class AutoTrainToggleRequest(BaseModel):
    enabled: bool


def _capture_public(row: dict) -> dict:
    cam = camera_db.get_camera(row["camera_id"])
    return {
        "id": row["id"],
        "camera_id": row["camera_id"],
        "camera_name": cam["name"] if cam else f"Camera {row['camera_id']}",
        "captured_at": row["captured_at"],
        "detection_confidence": row["detection_confidence"],
        "blur_score": row["blur_score"],
        "brightness": row["brightness"],
    }


@router.get("/next")
def next_capture():
    # A capture's row can outlive its file (e.g. the on-disk image was lost
    # to something outside this pipeline — verified by direct audit: 193
    # early captures from this dataset have no file on disk). Such a row can
    # never be shown to a human, so treat "file provably missing" the same
    # as a manual Skip (no employee_id guessed, nothing fabricated) instead
    # of leaving the labeling queue stuck serving a 404 image forever.
    row = face_db.get_next_unlabeled_capture()
    auto_skipped_missing = 0
    while row is not None and not os.path.exists(row["image_path"]):
        face_db.skip_training_capture(row["id"])
        auto_skipped_missing += 1
        row = face_db.get_next_unlabeled_capture()
    stats = face_db.get_training_stats()
    return {
        "capture": _capture_public(row) if row else None,
        "auto_skipped_missing_files": auto_skipped_missing,
        **stats,
    }


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
    if not os.path.exists(row["image_path"]):
        # Same "file provably missing" case /next auto-skips — reachable
        # here only if the file disappeared between /next and this call.
        face_db.skip_training_capture(req.capture_id)
        raise HTTPException(status_code=410, detail="Image file is missing on disk — capture skipped automatically")

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


def _move_for_label(row: dict, employee_id: str) -> str:
    """Shared with /label: moves a capture's file into <employee_id>/,
    returns the new path. Handles both 'this file is still in the
    _unlabeled staging dir' (a skipped capture being labeled for the first
    time) and 'this file is already under a different employee's folder'
    (correcting an existing label)."""
    dest_dir = os.path.join(TRAINING_CAPTURE_DIR, employee_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(row["image_path"]))
    if os.path.abspath(row["image_path"]) != os.path.abspath(dest_path):
        shutil.move(row["image_path"], dest_path)
    return dest_path


@router.post("/relabel")
def relabel(req: RelabelRequest):
    """Corrects the employee_id on a capture that was already labeled or
    skipped — the fix for 'I typed the wrong ID'. Unlike /label, this
    accepts a capture that isn't currently 'unlabeled'."""
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
    if row["label_status"] not in ("labeled", "skipped"):
        raise HTTPException(
            status_code=409,
            detail=f"Capture is {row['label_status']} — use /label for an unlabeled capture instead",
        )
    if not os.path.exists(row["image_path"]):
        raise HTTPException(status_code=410, detail="Image file is missing on disk — cannot relabel")

    dest_path = _move_for_label(row, req.employee_id)
    try:
        face_db.relabel_training_capture(req.capture_id, req.employee_id, dest_path)
    except ValueError as e:
        # Move it back to exactly where it was so nothing is orphaned.
        if os.path.abspath(dest_path) != os.path.abspath(row["image_path"]):
            shutil.move(dest_path, row["image_path"])
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True}


@router.post("/unlabel")
def unlabel(req: UnlabelRequest):
    """Plain undo: sends a labeled/skipped capture back into the unlabeled
    queue. No employee_id guessed or assumed — the next /next call will
    serve it again for a human to label properly."""
    row = face_db.get_training_capture(req.capture_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    if row["label_status"] not in ("labeled", "skipped"):
        raise HTTPException(status_code=409, detail=f"Capture is {row['label_status']} — nothing to undo")

    if row["label_status"] == "labeled" and os.path.exists(row["image_path"]):
        dest_path = os.path.join(TRAINING_UNLABELED_DIR, os.path.basename(row["image_path"]))
        if os.path.abspath(row["image_path"]) != os.path.abspath(dest_path):
            shutil.move(row["image_path"], dest_path)
    else:
        dest_path = row["image_path"]  # skipped captures were never moved

    try:
        face_db.unlabel_training_capture(req.capture_id, dest_path)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True}


@router.get("/recent-labels")
def recent_labels(limit: int = 8):
    """Powers the labeling page's 'just labeled' correction list — recent
    labeled captures only (not skipped), newest first."""
    return [_capture_public(r) | {"employee_id": r["employee_id"]} for r in face_db.get_recent_labeled_captures(limit)]


@router.get("/employees")
def employees():
    return face_db.list_employees()


@router.post("/employees")
def add_employee(req: EmployeeIn):
    face_db.upsert_employee(req.employee_id, req.name)
    return {"ok": True}


@router.post("/employees/sync")
def sync_employees():
    """Pull from the external face-enrollment service to refresh the local
    roster — the only place that host is ever touched by the training
    pipeline. Called explicitly (this route) and automatically whenever the
    /face-training page loads (see FaceTraining.jsx), so an ID or name
    edited on that external service shows up here without a manual step.

    Also prunes local rows whose ID the external service no longer reports
    (see face_db.delete_employees_not_in) — otherwise a renumbered ID would
    leave the old one stuck in the labeling datalist forever. Guarded
    against a malformed/partial response wiping the roster: only prunes if
    the external service actually returned at least one ID.

    For any employee_id also listed in employee_directory.py, that
    directory's name always wins over whatever this external service
    reports — it has had known-wrong/misspelled names in the past (e.g.
    037 as "Shyam" instead of "Shams") and there's no update API on it to
    fix at the source, so the correction has to be reapplied here on every
    sync rather than getting silently overwritten again.
    """
    try:
        with urllib.request.urlopen(EXTERNAL_FACES_API, timeout=10) as resp:
            rows = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"Could not reach external faces API: {e}")

    imported = 0
    skipped_no_id = 0
    current_ids = set()
    for r in rows:
        employee_id = r.get("employee_id")
        if not employee_id:
            skipped_no_id += 1
            continue
        corrected = employee_directory.get_employee(employee_id)
        name = corrected["name"] if corrected else r.get("name", employee_id)
        face_db.upsert_employee(employee_id, name)
        current_ids.add(employee_id)
        imported += 1

    removed = face_db.delete_employees_not_in(current_ids) if current_ids else 0

    return {"imported": imported, "skipped_no_employee_id": skipped_no_id, "removed_stale": removed}


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


@router.get("/auto-train/status")
def auto_train_status():
    """Background periodic-retraining scheduler state — see
    face_training_scheduler.py. Also folded into /collection/status for a
    single combined monitoring call."""
    return face_training_scheduler.get_status()


@router.post("/auto-train/toggle")
def auto_train_toggle(req: AutoTrainToggleRequest):
    """Pause/resume automatic retraining without a restart. Manual
    POST /train is unaffected either way — this only controls the
    background trigger."""
    face_training_scheduler.set_enabled(req.enabled)
    return {"ok": True, **face_training_scheduler.get_status()}


@router.get("/training-history")
def training_history(limit: int = 20):
    """Newest-first record of every completed training run (see
    face_training.train_classifier -> face_db.add_training_run) — real
    measured numbers only, nothing estimated. Powers the 'Face Model
    Training' history panel in FaceTraining.jsx."""
    return face_db.list_training_runs(limit)


# ---------------------------------------------------------------------------
# Background collection — runs independently of any browser live-view, and
# (as of the 7-day plan) independently of this backend process staying up
# the whole time too: see face_collection.py's persisted sessions +
# resume_if_needed() (called from main.py's startup) + the automatic expiry
# watcher. Explicit start/stop only — nothing here starts a session on its
# own just because cameras exist.
# ---------------------------------------------------------------------------

@router.post("/collection/start")
def start_collection(req: CollectionStartRequest = CollectionStartRequest()):
    camera_ids = req.camera_ids
    if camera_ids is None:
        camera_ids = [c["id"] for c in camera_db.list_cameras() if c.get("is_configured")]
    if not camera_ids:
        raise HTTPException(status_code=422, detail="No configured cameras to start collection on")

    session = face_collection.start_session(camera_ids, days=req.days)
    return {"ok": True, "session": session, **face_collection.status()}


@router.post("/collection/stop")
def stop_collection(req: CollectionStopRequest = CollectionStopRequest()):
    if req.camera_id is not None:
        was_running = face_collection.stop(req.camera_id)
        return {"ok": True, "was_running": was_running, **face_collection.status()}

    session = face_collection.stop_session(reason="stopped")
    return {"ok": True, "session": session, **face_collection.status()}


@router.get("/collection/status")
def collection_status():
    active_ids = face_collection.status()["active_camera_ids"]
    session = face_db.get_running_collection_session()
    now = time.time()

    stats = face_db.get_training_stats()
    per_camera_raw = face_db.get_per_camera_capture_breakdown()
    per_camera = {}
    for cid in sorted(set(active_ids) | set(per_camera_raw.keys())):
        cam_stats = per_camera_raw.get(
            cid, {"total": 0, "usable": 0, "no_embedding": 0, "rejected": 0,
                  "labeled": 0, "skipped": 0, "last_capture_at": None}
        )
        is_active = cid in active_ids
        last_at = cam_stats["last_capture_at"]
        stalled = bool(
            is_active and session and (now - session["started_at"]) > STALLED_THRESHOLD_SECONDS
            and (last_at is None or (now - last_at) > STALLED_THRESHOLD_SECONDS)
        )
        cam = camera_db.get_camera(cid)
        per_camera[cid] = {
            "camera_name": cam["name"] if cam else f"Camera {cid}",
            "collecting": is_active,
            "at_per_camera_limit": cam_stats["total"] >= MAX_CAPTURES_PER_CAMERA,
            "health": "stalled" if stalled else ("active" if is_active else "inactive"),
            **cam_stats,
        }

    session_public = None
    if session:
        session_public = {
            **session,
            "elapsed_seconds": now - session["started_at"],
            "time_remaining_seconds": max(0.0, session["planned_end_at"] - now),
        }

    return {
        "session": session_public,
        "active_camera_ids": active_ids,
        "capture_limit": MAX_TRAINING_CAPTURES,
        "per_camera_capture_limit": MAX_CAPTURES_PER_CAMERA,
        "current_total": stats["all_captures"],
        "limit_reached": stats["all_captures"] >= MAX_TRAINING_CAPTURES,
        # by_status has the precise breakdown: unlabeled (= usable, awaiting
        # a human label), labeled, skipped, no_embedding, rejected.
        "by_status": stats["by_status"],
        "reviewed": stats["reviewed"],
        "reviewable_total": stats["total"],  # unlabeled+labeled+skipped — excludes no_embedding/rejected
        "per_camera": per_camera,
        "disk_usage_bytes": face_collection.get_disk_usage_bytes(),
        "training": face_training_scheduler.get_status(),
        "model": model_status(),
    }
