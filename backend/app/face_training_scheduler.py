"""
face_training_scheduler.py

Periodic background training trigger — the missing link between "label some
captures in /face-training" and the classifier the live pipeline picks up in
CameraFacePipeline._get_classifier(). Runs entirely in its own daemon
thread; never called from a request handler, never blocks FastAPI or any
camera's read loop.

Trigger policy:
  - Check every FACE_AUTO_TRAIN_CHECK_INTERVAL_SECONDS.
  - Train as soon as >= FACE_AUTO_TRAIN_MIN_NEW_LABELED new samples have been
    labeled since the last successful run, OR at least
    FACE_AUTO_TRAIN_MAX_INTERVAL_SECONDS has passed since the last run AND at
    least one new sample has been labeled since then. Never retrains on zero
    new data — that would just reproduce the same model for no reason.
  - Every run refits on the COMPLETE labeled dataset (see
    face_training.train_classifier) — this is what makes retraining additive
    rather than incremental-and-forgetful, and is unchanged by this module.

CPU protection: before actually starting a run, checks psutil.cpu_percent()
against FACE_AUTO_TRAIN_CPU_PAUSE_PERCENT. If the box is already busy (live
cameras + person/face detection), the run is deferred to the next check
interval instead of forced through — yield rather than starve the live
pipeline, not a retry-immediately loop.

Training itself (sklearn LogisticRegression.fit on a few thousand frozen
512-dim embeddings already computed at capture time) is a lightweight,
sub-second-to-low-single-digit-seconds operation at this dataset's scale —
unlike YOLO/ArcFace inference, it does not need a separate process to stay
off the live pipeline's critical path; one dedicated thread already
isolates it from request handling and from each camera's own read loop. If
the labeled dataset or classifier ever grows enough to change that
calculus, this is the one place to swap in a subprocess without touching
anything else.
"""

import logging
import os
import threading
import time
from datetime import datetime

import psutil

from app import face_db
from app.face_training import train_classifier

log = logging.getLogger("face_training_scheduler")

_ENABLED_DEFAULT = os.environ.get("FACE_AUTO_TRAIN_ENABLED", "true").strip().lower() not in ("0", "false", "no")
CHECK_INTERVAL_SECONDS = float(os.environ.get("FACE_AUTO_TRAIN_CHECK_INTERVAL_SECONDS", "1800"))  # 30 min
MIN_NEW_LABELED = int(os.environ.get("FACE_AUTO_TRAIN_MIN_NEW_LABELED", "40"))
MAX_INTERVAL_SECONDS = float(os.environ.get("FACE_AUTO_TRAIN_MAX_INTERVAL_SECONDS", "21600"))  # 6h
CPU_PAUSE_PERCENT = float(os.environ.get("FACE_AUTO_TRAIN_CPU_PAUSE_PERCENT", "85"))

_lock = threading.Lock()
_state = {
    "enabled": _ENABLED_DEFAULT,
    "status": "idle",  # idle | running | success | failed
    "last_run_at": None,
    "last_error": None,
    "last_result_summary": None,
    "last_check_at": None,
    "last_skip_reason": None,
}
_thread_started = False
_thread_lock = threading.Lock()


def _last_run() -> dict | None:
    runs = face_db.list_training_runs(limit=1)
    return runs[0] if runs else None


def _start_of_today() -> float:
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def get_status() -> dict:
    with _lock:
        state = dict(_state)
    last_run = _last_run()
    last_trained_at = last_run["trained_at"] if last_run else 0.0
    state.update({
        "last_trained_at": last_trained_at or None,
        # training_runs.id is a plain autoincrement PK — already exactly the
        # "which numbered classifier version is live" answer, no separate
        # version counter needed.
        "classifier_version": last_run["id"] if last_run else None,
        "new_labeled_since_last_training": face_db.count_labeled_since(last_trained_at),
        "samples_added_today": face_db.count_captures_since(_start_of_today()),
        "samples_labeled_today": face_db.count_labeled_since(_start_of_today()),
        "latest_validation_accuracy": last_run["validation_accuracy"] if last_run else None,
        "weak_classes": (last_run.get("per_class_validation") or {}).get("weak_classes", []) if last_run else [],
        "insufficient_sample_classes": (
            (last_run.get("per_class_validation") or {}).get("insufficient_sample_classes", []) if last_run else []
        ),
        "min_new_labeled_threshold": MIN_NEW_LABELED,
        "max_interval_seconds": MAX_INTERVAL_SECONDS,
        "check_interval_seconds": CHECK_INTERVAL_SECONDS,
        "cpu_pause_percent": CPU_PAUSE_PERCENT,
    })
    return state


def set_enabled(enabled: bool) -> None:
    """In-memory only — resets to FACE_AUTO_TRAIN_ENABLED on restart, same as
    every other env-configured knob in this pipeline. Lets an operator pause
    auto-training (e.g. mid-incident) without a restart, without needing a
    persisted settings table for what is otherwise a one-line env var."""
    with _lock:
        _state["enabled"] = enabled


def _run_training_once() -> None:
    with _lock:
        _state["status"] = "running"
    try:
        result = train_classifier()
        with _lock:
            _state["status"] = "success"
            _state["last_run_at"] = time.time()
            _state["last_error"] = None
            _state["last_skip_reason"] = None
            _state["last_result_summary"] = {
                "trained_on_samples": result["trained_on_samples"],
                "employees": result["employees"],
                "validation": result["validation"],
            }
        log.info(
            "auto-train: trained on %d samples across %d employees",
            result["trained_on_samples"], len(result["employees"]),
        )
    except ValueError as e:
        # Not a failure — "not enough labeled data/classes yet" (the same
        # exception the manual /train endpoint turns into a 422). Don't leave
        # the scheduler sitting in a scary "failed" state for something that
        # resolves itself once more labeling happens.
        with _lock:
            _state["status"] = "idle"
            _state["last_skip_reason"] = str(e)
        log.info("auto-train: skipped this cycle — %s", e)
    except Exception as e:
        with _lock:
            _state["status"] = "failed"
            _state["last_error"] = str(e)
        log.exception("auto-train: training run failed unexpectedly")


def _check_once() -> None:
    with _lock:
        enabled = _state["enabled"]
        _state["last_check_at"] = time.time()
    if not enabled:
        return

    last_run = _last_run()
    last_trained_at = last_run["trained_at"] if last_run else 0.0
    new_labeled = face_db.count_labeled_since(last_trained_at)
    if new_labeled == 0:
        return  # nothing new at all — never retrain on zero new data

    time_since = time.time() - last_trained_at if last_trained_at else float("inf")
    over_max_interval = time_since >= MAX_INTERVAL_SECONDS
    should_train = new_labeled >= MIN_NEW_LABELED or over_max_interval
    if not should_train:
        return

    # MAX_INTERVAL_SECONDS is meant as an absolute staleness ceiling, not
    # just another optional trigger alongside MIN_NEW_LABELED. On a box
    # whose live cameras keep CPU chronically high (3+ concurrent
    # detection pipelines can pin every core continuously), the CPU-pause
    # check below would otherwise defer every single cycle forever, and
    # the classifier would never update no matter how much labeled data
    # piles up — exactly what happened here: 1600+ labeled corrections
    # sat unused for ~2 days because CPU never once dropped under the
    # pause threshold. Once time_since is already past that ceiling,
    # train anyway rather than keep deferring — a few extra seconds of
    # CPU contention beats a classifier that silently never learns new
    # corrections. The CPU pause still applies normally to the
    # MIN_NEW_LABELED-only trigger, which is the case it was meant for
    # (yield to a busy live pipeline for a burst of new labels that can
    # easily wait for the next check).
    if not over_max_interval:
        cpu = psutil.cpu_percent(interval=1)
        if cpu >= CPU_PAUSE_PERCENT:
            with _lock:
                _state["last_skip_reason"] = f"deferred — system CPU at {cpu:.0f}% (>= {CPU_PAUSE_PERCENT:.0f}% threshold)"
            log.info("auto-train: deferring this cycle, CPU at %.0f%%", cpu)
            return

    _run_training_once()


def _loop() -> None:
    while True:
        time.sleep(CHECK_INTERVAL_SECONDS)
        try:
            _check_once()
        except Exception:
            log.exception("auto-train scheduler check failed")


def ensure_scheduler_started() -> None:
    """Idempotent — safe to call from app startup. One daemon thread for the
    process's lifetime, same pattern as
    face_collection.ensure_expiry_watcher_started()."""
    global _thread_started
    with _thread_lock:
        if not _thread_started:
            threading.Thread(target=_loop, daemon=True).start()
            _thread_started = True
