"""
health_routes.py

  GET /health          liveness: the process is up and serving HTTP. Always
                       200 while the event loop runs. For systemd/k8s/load
                       balancer liveness checks and the deploy script.
  GET /ready           readiness: can this instance do its job? 200 or 503,
                       with the reason. Checks the database, configuration,
                       that an admin account exists, and that every REQUIRED
                       model file is present. Also reports (without failing
                       on) per-camera stream state and per-feature recovery
                       state. Unauthenticated, so it carries no names, hosts,
                       URLs, credentials, images or personal data — camera
                       ids and states only.
  GET /api/health/details  the same plus camera names and model paths, admin only.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from . import auth, config, db, models, resilience

router = APIRouter(tags=["health"])

_started_at = time.time()


def _cameras() -> list[dict]:
    try:
        from . import camera_stream

        return camera_stream.all_status()
    except Exception:
        return []


def _capture_cap() -> dict:
    try:
        from . import face_db
        from .face_pipeline import MAX_TRAINING_CAPTURES

        n = face_db.count_training_captures()
        return {"count": n, "limit": MAX_TRAINING_CAPTURES, "reached": n >= MAX_TRAINING_CAPTURES}
    except Exception:
        return {"count": None, "limit": None, "reached": None}


def readiness(detailed: bool = False) -> tuple[bool, dict]:
    db_ok, db_msg = db.check()
    problems = config.validate()
    try:
        admins = auth.count_admin_users() if db_ok else 0
    except Exception:
        admins = 0
    missing = models.missing_required()
    checks = {
        "database": {"ok": db_ok, "detail": db_msg},
        "configuration": {"ok": not (problems and config.IS_PRODUCTION), "problems": len(problems)},
        "admin_account": {"ok": admins > 0},
        "models": {"ok": not missing, "missing": missing},
    }
    ready = all(c["ok"] for c in checks.values())

    cams = _cameras()
    features = resilience.all_health()
    body = {
        "status": "ready" if ready else "not_ready",
        "uptime_seconds": round(time.time() - _started_at),
        "checks": checks,
        "cameras": [
            {"camera_id": c["camera_id"], "state": c["state"],
             "seconds_since_frame": round(time.time() - c["last_frame_at"], 1) if c.get("last_frame_at") else None,
             "face_recognition": c["face_recognition"]["state"]}
            for c in cams
        ],
        "features": [
            {"feature": f["feature"], "camera_id": f["camera_id"], "state": f["state"],
             "consecutive_failures": f["consecutive_failures"], "retry_in_seconds": f["retry_in_seconds"]}
            for f in features
        ],
        "degraded": sorted({f["feature"] for f in features if f["state"] != resilience.RUNNING}),
        "training_captures": _capture_cap(),
    }
    if detailed:
        body["configuration_problems"] = problems
        body["models"] = models.status()
        body["features"] = features
        body["cameras"] = cams
    return ready, body


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    ok, body = readiness()
    return JSONResponse(body, status_code=200 if ok else 503)


@router.get("/api/health/details")
def health_details(_: dict = Depends(auth.require_admin)):
    ok, body = readiness(detailed=True)
    return JSONResponse(body, status_code=200)
