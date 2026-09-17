# Deco Vision — People Identification System: Architecture & Implementation Status

**Status of this document:** rewritten from scratch to describe the system as it actually exists in this codebase today, verified by reading the source (`backend/app/*.py`, `frontend/src/**`) and by querying the live running backend, not from the original theoretical design. Where the original design (full-body detection → Re-ID → gallery search → temporal identity fusion) is **not** what's implemented, this document says so explicitly instead of describing it as done.

Every major component below is labeled **IMPLEMENTED**, **TESTED**, or **PLANNED** — do not treat a component as production-proven just because it's implemented; "implemented" and "tested" are tracked separately on purpose.

---

## 1. Project Overview

Deco Vision is a camera-based monitoring system built around CP Plus RTSP cameras, a FastAPI backend, and a React frontend. It currently has two working sub-systems: (1) live camera viewing over WebSocket, and (2) a face-detection/embedding/training pipeline that runs quietly alongside the live view and is being used to build a real, camera-collected face dataset for a future employee-recognition classifier.

It is **not** a finished person-identification product. There is no full-body detection, no Re-ID, no production recognition running against live cameras today. What exists is a working detection → tracking → embedding → capture pipeline, a manual labeling tool, and a classifier-training script — verified with real camera data, but not yet trained on that real data and not yet driving any attendance/access-control decision.

## 2. Objective

The system is being built to eventually support:
- Employee identification via face recognition
- Attendance logging
- Visitor management
- Restricted-area monitoring
- General real-time camera monitoring and future analytics

Today, only **real-time camera monitoring** (live viewing) and **face-data collection for future recognition** are implemented. The other use cases (attendance, visitor management, restricted-area alerts) have no backend logic behind them yet — see Section 21.

## 3. Actual Technology Stack

**Frontend** (`frontend/`, verified against `package.json`)
- React 19.2
- Vite 8.3
- React Router 7.18
- Tailwind CSS 4.3
- `lucide-react` (icons), `recharts` (charts)

**Backend** (`backend/`, verified against `requirements.txt`)
- Python, FastAPI 0.141 + Uvicorn 0.53 (single worker, `uvicorn app.main:app --reload`, run from inside `backend/`)
- OpenCV (`opencv-python`) for RTSP frame reads
- SQLite via the stdlib `sqlite3` module — no ORM
- `python-dotenv` for config (`backend/.env`)
- `ultralytics` 8.4 — YOLOv8, used here as a **face detector** (see Section 6)
- `supervision` 0.30 — provides the ByteTrack implementation actually in use
- `insightface` 2.0 + `onnxruntime` 1.30 (CPU execution provider only — no CUDA on this machine) — ArcFace embeddings
- `scikit-learn` 1.9 (`LogisticRegression`) + `joblib` — the supervised classifier layer
- Communication: WebSocket for live video (`/ws/live/{camera_id}`); plain HTTP/JSON REST for everything else
- Version control: Git

**Database:** SQLite, single file at `backend/data/app.db`.

## 4. Current Architecture

The pipeline that is actually implemented and running:

```
CP Plus Camera
  ↓ RTSP
OpenCV CameraStream (one background thread per camera)
  ↓ every frame
YOLOv8 Face Detection (throttled to ~3 fps, not every frame)
  ↓
ByteTrack Face Tracking (per camera)
  ↓
Best-Crop Selection (per finished track)
  ↓
Padded Face Crop
  ↓
InsightFace / ArcFace re-detect + align + embed
  ↓
512-D Normalized Embedding
  ↓
Quality Filtering + Duplicate Suppression
  ↓
Training Capture saved to disk + SQLite
  ↓ (separate, manual, explicit steps)
Manual Employee Labeling → Supervised Classifier Training → Recognition
```

This matches the "what we actually have" description in the current build, not the original person-detector/Re-ID design. There is **no full-body person detector, no Re-ID, no gallery-search/temporal-fusion state machine** anywhere in the code — see Section 21 for the full implemented-vs-not list.

The same pipeline also has a lightweight **live-matching** path (Section 14) that runs against whatever is currently in the embedding gallery/classifier, independent of the bulk-collection path above — both are driven off the same detected/tracked face, just used for different purposes.

## 5. Camera Architecture

**IMPLEMENTED.** One `CameraStream` instance per camera (`backend/app/camera_stream.py`): a single background thread opens the RTSP stream via `cv2.VideoCapture(url, cv2.CAP_FFMPEG)`, reads frames in a loop, and fans out JPEG-encoded frames to every subscriber (browser WebSocket viewers, or the background face-collection "phantom" subscriber — see Section 9). Multiple viewers of the same camera share one RTSP connection. Automatic reconnect on a failed `cap.read()` or failed `cap.isOpened()` (retry with backoff), inherited by anything that subscribes to the stream, including background collection.

Camera credentials (`host`, `port`, `user`, `password`, `stream_path`) are stored in the `cameras` SQLite table. **The password is never returned to the frontend** — `camera_db._row_to_dict()` strips it from every API response; only the internal `get_camera_connection()` (used to build the RTSP URL server-side) includes it. This document does not list any camera credentials, matching that same rule.

**Currently configured cameras** (verified live against the running backend, IDs as actually assigned by the database — do not assume IDs 1 or 2 exist, they don't):

| Camera ID | Name | Site |
|---|---|---|
| 3 | Main Section 1 | Noida Site |
| 4 | Main Section 2 | Noida Site |
| 5 | Lift Area | Noida Site |
| 6 | Tech Section | Noida Site |

All four are CP Plus cameras, RTSP, currently `status: active`.

**Live viewing**: `GET`-upgraded WebSocket `/ws/live/{camera_id}` streams JPEG frames to the browser at `LIVE_STREAM_FPS` (default 8/sec, independent of the source camera's own frame rate). `/ws/detections/{camera_id}` exists but is a **placeholder** — it sends `{"faces": [], "fire_smoke": []}` on a fixed 1-second timer regardless of what the face pipeline actually sees. No detection results are currently pushed to the frontend live view; there is no bounding-box overlay UI for it either, since there is nothing real to draw yet.

## 6. Face Detection

**IMPLEMENTED, TESTED.**

This is a **dedicated face detector**, not the generic COCO `yolov8n.pt` (which only detects the class "person"). The code loads a face-tuned YOLOv8 checkpoint from `backend/models/yolov8n-face.pt` (resolved via `YOLO_FACE_WEIGHTS`, overridable by env var, resolved relative to the backend package so it doesn't matter what directory the process is launched from).

- Model: `ultralytics.YOLO`, loaded once per process and shared across all camera pipeline instances.
- Runs on every ~3rd-of-a-second sample (`FACE_PIPELINE_FPS`, default `3`), not on every frame off the RTSP stream — this bounds CPU cost (see Section 20).
- Default detection confidence filter: **0.5** (`self._yolo.predict(frame, verbose=False, conf=0.5)`), hardcoded in `face_pipeline.py`'s `_detect_faces()` — not currently an env var.
- Returns bounding boxes (`x1, y1, x2, y2`) and per-box confidence.

**Crop padding (important, and easy to miss):** before a detected face is handed to InsightFace, its bounding box is expanded by a margin (`FACE_CROP_MARGIN`, default `0.4` = 40% on each side, clamped to the frame) via `_pad_bbox()`. This exists because a bbox-tight crop reliably made InsightFace's own internal re-detection step (used for alignment) find **zero** faces — confirmed directly with `scripts/verify_face_detector.py`: YOLO detected a real face at 0.860 confidence, but `FaceAnalysis.get()` on that exact tight crop returned nothing. Padding the crop (not changing what YOLO detected, not changing tracking/quality scoring, which both still use the tight box) fixed this. This is documented here because it is a real, previously-broken part of the pipeline that is easy to silently regress.

## 7. Tracking

**IMPLEMENTED, TESTED.**

ByteTrack, via the `supervision` package (`sv.ByteTrack`) — **not** the original `yolox` package the design may have originally called for; `yolox` requires a native build toolchain (cmake) that isn't available on this machine, so `supervision`'s pure-Python ByteTrack port is used instead. This is a face-track, not a person-track: it operates on the YOLO face-detector's boxes, one tracker per camera.

What it actually does, per camera:
- Assigns a stable `track_id` to each detected face across frames.
- `CameraFacePipeline` keeps a `TrackState` per track: best-seen quality score (bbox area × confidence), the corresponding crop, and last-seen frame index.
- A track is considered finished/stale once it hasn't been seen for `TRACK_MAX_AGE` (30 sample-frames); at that point its single best frame is sent for embedding/recognition — recognition runs **once per finished track**, not once per frame.
- No cross-camera tracking or Re-ID exists — each camera's tracker and track-ID space is entirely independent.

## 8. Face Embedding

**IMPLEMENTED, TESTED.**

InsightFace (`buffalo_l` model pack), used as a **frozen** embedding model — it is not fine-tuned or retrained anywhere in this codebase. `_embed()` in `face_pipeline.py` runs InsightFace's own detector on the padded crop (needed for correct 5-point-landmark alignment before embedding — this re-detection step, not YOLO's box, is what actually determines embedding quality) and returns `face.normed_embedding`.

- Output: 512-dimensional, `float32`, L2-normalized.
- Verified directly via `scripts/verify_face_detector.py`: embedding shape `(512,)`, L2 norm `1.0000`.
- Runs on `CPUExecutionProvider` only — no CUDA available on this machine (confirmed no GPU acceleration is configured anywhere in the code or dependencies).

**This is the single most important architectural decision to get right when reading this codebase:** the system is *not* just building an embedding gallery for nearest-neighbor matching (though that fallback path does exist, see Section 14). It is building toward a **frozen ArcFace embedding + supervised employee classifier**, described next.

## 9. Training Data Collection

**IMPLEMENTED, TESTED, CURRENTLY RUNNING** (see Section 19 for live numbers).

Every finished track — regardless of whether it produces a confident match, a low-confidence one, or fails to embed at all — is unconditionally sent to `_save_training_capture()`. This is separate from the older, narrower `face_pending` review queue (which only ever receives low-confidence *matches*, deduped to once per track per 60s, and could never produce a large dataset on its own).

**Background collection, independent of the browser** — `backend/app/face_collection.py`:
- A `_CollectorSink` object with a no-op `put_nowait()` subscribes to the same `CameraStream` a real browser viewer would use. This keeps the camera's RTSP-reading thread (and therefore `feed_frame()`, and therefore capture generation) alive with **zero real WebSocket viewers**, and without opening a second RTSP connection.
- Collection is **explicit start/stop only** — it never starts on its own just because a camera exists, and stopping the browser's live view does not stop it.
- **Persistent, multi-day sessions**: `start_session(camera_ids, days)` records a row in `collection_sessions` (start time, planned end time, camera list, running activity, duplicates-rejected count). `resume_if_needed()`, called once at FastAPI startup, detects an unfinished session from before a restart and re-subscribes its cameras automatically — verified across real backend restarts. A background thread (`_expiry_loop`, checked every `SESSION_CHECK_INTERVAL_SECONDS` = 60s) stops the session automatically once its planned end time passes, with no need for anyone to poll status.
- Default session length if not specified: `DEFAULT_COLLECTION_DAYS` = **7 days**.

**Global and per-camera capture limits**, enforced atomically via a shared lock (`face_db.capture_limit_lock`) around a count-then-insert sequence, so concurrent camera threads can never overshoot either:
- `MAX_TRAINING_CAPTURES = 15000` — across **all** cameras, **all** statuses combined.
- `MAX_CAPTURES_PER_CAMERA = 5000` (`MAX_TRAINING_CAPTURES // 3`) — stops one busy camera from consuming the whole budget before quieter cameras contribute anything.

Both are configurable via env vars (`MAX_TRAINING_CAPTURES`, `FACE_MAX_CAPTURES_PER_CAMERA`) but these are the actual current defaults in `face_pipeline.py` — do not describe different numbers unless the code changes.

## 10. Data Quality Filtering

**IMPLEMENTED, TESTED.** Applied only to captures that already produced an embedding (an embedding-less capture is always classified `no_embedding`, quality is irrelevant to it). Current thresholds, from `face_pipeline.py`:

| Constant | Default | Purpose |
|---|---|---|
| `FACE_MIN_BLUR_SCORE` | 50 | Variance-of-Laplacian floor — rejects near-degenerate blur |
| `FACE_MIN_BRIGHTNESS` | 20 | Mean grayscale floor — rejects near-black frames |
| `FACE_MAX_BRIGHTNESS` | 235 | Mean grayscale ceiling — rejects blown-out frames |
| `FACE_MIN_AREA` | 900 px² | Minimum tight (unpadded) YOLO bbox area — rejects tiny/distant detections |

These were set as conservative floors/ceilings based on values actually observed in real test captures during development (documented in `backend/FACE_TRAINING.md`), not tuned to hit a target rejection rate. A capture failing this gate is stored with `label_status = 'rejected'` — not deleted, just excluded from the labeling queue and from classifier training.

## 11. Duplicate Suppression

**IMPLEMENTED, TESTED.** `_is_recent_duplicate()` in `face_pipeline.py`: before saving a new capture, its embedding is compared (cosine similarity) against a short, **per-camera, in-memory** list of recently-captured embeddings on that same camera. If similarity is `>= DEDUP_SIMILARITY_THRESHOLD` (default `0.7`) to anything within the last `DEDUP_COOLDOWN_SECONDS` (default `900` = 15 minutes), the capture is dropped entirely — no file written, no DB row.

This is **not identity recognition** — no employee_id or classifier is involved, and nothing is auto-labeled. It exists to stop tracking flicker (ByteTrack losing and re-acquiring the same person seconds later, or a brief RTSP reconnect) from generating many near-identical captures of one appearance. It is deliberately scoped to be narrow:
- Per-camera only — the same person on a different camera is not suppressed.
- 15-minute window — the same person returning later the same day, or several times across a day, still produces separate captures.

Verified with a controlled test: the same embedding submitted twice within the cooldown window produced one capture, not two; a different real embedding submitted immediately after was captured normally.

## 12. Manual Labeling

**IMPLEMENTED, TESTED.** Deliberately a fully separate, human-driven step — nothing in the collection pipeline auto-assigns an employee_id.

Backend (`face_training_routes.py`):
- `GET /api/faces/training/next` — oldest unlabeled capture (metadata only, no image bytes) + `{reviewed, total}` counters.
- `GET /api/faces/training/image/{capture_id}` — the JPEG, path resolved server-side from the ID; the client can never supply a filesystem path.
- `POST /api/faces/training/label` `{capture_id, employee_id}` — validates `employee_id` against a local `employees` roster table before accepting it (rejects unknown IDs, e.g. a typo, with a 422 and a clear message), then moves the file into an `<employee_id>/` subfolder and updates the row in a single transaction (rolled back / file moved back if either half fails).
- `POST /api/faces/training/skip` `{capture_id}` — marks a bad/unclear capture `skipped`, moves to the next one, no employee_id involved.
- `POST /api/faces/training/employees/sync` — one-time, explicit pull of `employee_id`/`name` pairs from the external face-enrollment service (`http://13.61.58.14/api/faces`) into the local roster, used only to validate labels server-side. Never called automatically.

Frontend: a standalone page at `/face-training` (`frontend/src/pages/FaceTraining.jsx`), deliberately outside the normal app shell (no sidebar) — a dark, full-screen, keyboard-driven tool: shows the capture image, camera name, timestamp, an Employee ID input, and an "X / Y reviewed" counter. Enter saves the label and loads the next capture without a page reload; Escape skips. Verified end-to-end with a real headless-browser session: typed an ID, pressed Enter, watched it save, advance, and the counter increment, with no console errors.

There is no automatic labeling anywhere — a human must type or confirm every `employee_id`.

## 13. Supervised Classifier Training

**IMPLEMENTED, TESTED with real data — but not yet run on the current in-progress camera dataset (see Section 19).**

Architecture: ArcFace stays **frozen**. A separate, explicit training step (`backend/app/face_training.py`, triggered only by `POST /api/faces/training/train`) fits a `sklearn.linear_model.LogisticRegression` on top of the frozen 512-D embeddings of every capture with `label_status = 'labeled'`. This is "Option B" — a supervised classifier on frozen embeddings, not fine-tuning the embedding network itself (fine-tuning ArcFace would need GPU-scale batches and margin-loss engineering that doesn't fit this CPU-only machine, and risks degrading a well-trained general embedding space for marginal gain from a small closed set of employees).

- Requires at least 2 employees with at least `MIN_SAMPLES_PER_CLASS = 2` labeled captures each; otherwise raises a clear error listing current per-employee label counts rather than training on too little data.
- Always retrains from scratch on the full current label set — there is no incremental/online update.
- Output: a `joblib`-serialized `LogisticRegression`, written to `classifier.joblib` inside the training-capture directory.
- `CameraFacePipeline._get_classifier()` picks up a newly trained file automatically (checked by file mtime on every match, not re-imported per-frame) — no backend restart needed for a freshly trained model to take effect.

**Training is never triggered automatically** — not after every label, not during collection, not on a schedule. It only runs when `/train` is called explicitly.

**What was actually verified**: earlier in development, a temporary classifier was trained on 4 real, human-labeled camera-style captures across 2 employees, and correctly classified two different, held-out photos of those same two people (not used in training). That test classifier and its test data were deliberately removed afterward — the system was returned to a clean state. **The current, in-progress 7-day camera dataset has not been labeled or trained on** — see Section 19. Do not describe the system as "trained" on real camera data at this point; it is collected, not yet audited, labeled, or trained.

## 14. Recognition Flow

**IMPLEMENTED (mechanism), NOT YET MEANINGFULLY TESTED against a trained real-camera classifier** (no such classifier exists yet — see Sections 13, 19).

`_match()` in `face_pipeline.py` tries, in order:
1. **Trained classifier**, if `classifier.joblib` exists — `predict_proba()`, compared against `FACE_CLASSIFIER_MIN_PROBA` (default `0.6`).
2. **Fallback: cosine similarity against the enrollment gallery** (`face_embeddings` table — populated via direct `/api/faces/enroll` uploads or assigned review-queue captures), compared against `FACE_MATCH_THRESHOLD` (default `0.45`). This is the only mode available before any `/train` call.

A confident match (above whichever threshold applies) sets `state.recognized_person_id` and returns — no downstream attendance/footfall logging is wired to that event today (see Section 21). Anything below threshold, or with an empty gallery, is queued to the separate `face_pending` review table (deduped to once per track per 60s) for a human to assign, which is different from — and narrower than — the bulk training-capture collection described in Section 9.

There is currently no gallery data from real, human-labeled camera captures and no trained classifier, so in practice every live match today falls through to the empty-gallery / no-classifier case and lands in the review queue rather than being confidently auto-recognized.

## 15. Database / Data Storage

SQLite, single file: `backend/data/app.db`. No ORM, no migration tool — schema created with `CREATE TABLE IF NOT EXISTS` plus manual `PRAGMA table_info()` + `ALTER TABLE` backfills for columns added after a table already existed in the wild.

Relevant tables (`backend/app/face_db.py`, `backend/app/camera_db.py`):

| Table | Purpose |
|---|---|
| `cameras` | Camera registry — name, site, `cam_code`, purpose, host/port/user/password/stream_path, vendor, status, `attendance_tracking` flag. Password is stripped from every API response. |
| `sites` | Site registry — name, description. |
| `face_embeddings` | The enrollment gallery — many embeddings per `person_id`. |
| `face_pending` | Live low-confidence review queue (camera_id, track_id, image_path, embedding, best-match candidate, assigned_person_id, status). |
| `employees` | Minimal local roster (`employee_id`, `name`) used only to validate a label server-side — not the system of record (that lives on the external service). |
| `face_training_captures` | The bulk dataset: camera_id, track_id, captured_at, image_path, embedding (nullable), detection_confidence, blur_score, brightness, employee_id (nullable), `label_status`, labeled_at. |
| `collection_sessions` | Persistent multi-day collection session state: start/planned-end time, camera list, status, last activity, duplicates-rejected count. |

`label_status` values on `face_training_captures`:
- `unlabeled` — has a valid embedding and passed quality checks; the only status the labeling queue (`/face-training`) ever shows.
- `labeled` — a human has assigned an `employee_id`.
- `skipped` — a human marked it unusable/unclear.
- `no_embedding` — YOLO found a face but ArcFace's re-detection/alignment step failed on the (padded) crop. Stored for diagnostics, never shown to a human, never trainable.
- `rejected` — has an embedding but failed the quality gate (Section 10). Same treatment as `no_embedding`.

Nothing is ever deleted from this pipeline — rows and image files for `no_embedding`/`rejected` captures are retained on disk and in the DB for diagnostics, just excluded from the human-facing queue and from training.

Image files are stored under `backend/data/face_training/` (unlabeled staging in `_unlabeled/`, moved into `<employee_id>/` subfolders once labeled) and `backend/data/face_captures/` (the separate, older low-confidence review queue's images). Note: on this machine, `backend/data` is a directory junction pointing outside the OneDrive-synced tree, after an earlier incident where some capture files went missing from disk despite valid DB rows — the code now also verifies (`os.path.exists`) that a file actually landed on disk before creating its DB row, regardless of storage location.

## 16. API Architecture

All routes below were read directly from `backend/app/main.py`, `face_routes.py`, and `face_training_routes.py` — nothing here is inferred or assumed.

**Cameras / Sites** (`main.py`)
| Method | Path |
|---|---|
| GET/POST | `/api/cameras` |
| PUT/DELETE | `/api/cameras/{camera_id}` |
| GET/POST | `/api/sites` |
| PUT/DELETE | `/api/sites/{site_id}` |

**Dashboard support** (`main.py`) — `GET /api/stats` (camera counts only — `faces_enrolled`, `active_alerts`, `detections_today` are hardcoded `0`, not real), `GET/POST /api/alerts...` (always returns `[]` / no-op), `GET/PUT /api/settings` (in-memory dict, resets on restart), `POST /api/auth/login` (records the submitted email, does **not** check a password — no user table, no JWT, no session store anywhere in this backend).

**Live video** — `WS /ws/live/{camera_id}` (real, streams JPEGs), `WS /ws/detections/{camera_id}` (placeholder, always empty).

**Face review/enrollment** (`face_routes.py`, prefix `/api/faces`)
| Method | Path |
|---|---|
| GET | `/pending?hours=&status=` |
| POST | `/assign` `{pending_id, person_id}` |
| POST | `/ignore` `{pending_id}` |
| POST | `/enroll` (multipart: `person_id`, `photo`) |
| GET | `/gallery/{person_id}/count` |

**Face training / collection** (`face_training_routes.py`, prefix `/api/faces/training`)
| Method | Path |
|---|---|
| GET | `/next` |
| GET | `/stats` |
| GET | `/image/{capture_id}` |
| POST | `/label` `{capture_id, employee_id}` |
| POST | `/skip` `{capture_id}` |
| GET/POST | `/employees` |
| POST | `/employees/sync` |
| POST | `/train` |
| GET | `/model-status` |
| POST | `/collection/start` `{camera_ids?, days?}` |
| POST | `/collection/stop` `{camera_id?}` |
| GET | `/collection/status` |

No JWT, no API-key auth, no rate limiting on any of the above — authentication for the app overall is the email-only login described above, which is not a real security boundary. Treat this whole API surface as trusted-network-only today.

## 17. Frontend Architecture

React 19 + Vite + React Router. Key structure:
- `src/pages/*` — one file per route.
- `src/layouts/` — `AppShell`, `Sidebar`, `Topbar`.
- `src/components/` — shared UI (`DataTable`, `Modal`, `SidePanel`, `LiveCameraTile`, `CameraViewerModal`, `FaceEnrollment`, etc.).
- `src/hooks/useLiveCameraFeed.js` — WebSocket→canvas streaming logic shared by the live-camera grid and the enlarged viewer modal.
- `src/api/client.js` — the single file every page goes through to reach a backend; **this is where "real vs. mock" is decided per feature.**

**Wired to this repo's real FastAPI backend:** camera CRUD, site CRUD (including the Edit Site camera-assignment flow), the live-view WebSocket, and the entire face-training/labeling page (`/face-training`).

**Wired to a separate, already-deployed external service** (`http://13.61.58.14`, not part of this repo): the People page's Employee tab reads real names/employee IDs/photos directly from the browser via `GET /api/faces` on that host, with a fallback to mock data if it's unreachable. This is read-only — nothing in this app writes back to that service.

**Still mock / local component state only, not backed by any real API in this repo:** Alerts, Attendance, Workforce (Insights), Footfall, Intrusion, the Dashboard's non-camera stats, Settings (profile/notifications/rules), and the People page's "Incoming Guests"/"Validated" tabs plus its Add/Edit/Delete person actions (state resets on page refresh). This is a factual statement about the current build, not a criticism — these pages have UI but no backend behind them yet.

The `/face-training` labeling tool (Section 12) is the one page that is fully live-data end to end, by design — it's the actual working tool used against the real collection pipeline.

## 18. Current Testing

Verified directly, against the real running backend (not mocked), during development of this pipeline:

- YOLO face-model loading and real face detection on a real photo, via `scripts/verify_face_detector.py`.
- Face crop extraction, including confirming the padding fix was required (tight crop → embedding failure; padded crop → success).
- ArcFace embedding generation and normalization (`(512,)`, L2 norm `1.0000`).
- Real camera streaming continuing unaffected even with the face pipeline disabled/failing (isolated try/except in `camera_stream.py`).
- Real camera face capture on one live, configured camera — produced real, non-empty JPEG files and correct database metadata.
- Background collection running with zero WebSocket viewers connected.
- Multi-camera background collection (cameras 3, 4, 5, 6 simultaneously).
- Global capture-limit enforcement (verified with a temporarily lowered limit — collection stopped exactly at the configured ceiling).
- Cross-track duplicate suppression (controlled before/after test with the same vs. a different embedding).
- The manual labeling workflow end-to-end via a real headless-browser session.
- Classifier training workflow end-to-end: trained on real labeled samples, then correctly classified held-out photos of the same people not used in training (test data removed afterward).
- Trained-classifier auto-reload by the recognition pipeline (`_get_classifier()`'s mtime check) without a backend restart.
- Persistent collection sessions surviving a real backend restart (auto-resume confirmed on more than one occasion).

**Not tested:** recognition accuracy/precision against a real, human-labeled camera dataset at any scale (none has been labeled yet — see Section 19); performance/behavior under sustained multi-day unattended operation beyond the current in-progress run; concurrent-write correctness under load higher than what's been manually exercised.

## 19. Current Collection Status

**This section is a snapshot, not a live value — re-check `GET /api/faces/training/collection/status` for current numbers if this document is read later.** Snapshot taken while writing this document (2026-09-16):

- Collection session: **running**, session id 3, cameras `[3, 4, 5, 6]`, started ~2.9 hours before this snapshot, planned to run for the full default 7-day window.
- Total captures so far: **1,094** (well under the 15,000 global limit; no camera near its 5,000 per-camera share).
- Breakdown by status: `unlabeled` (usable, awaiting a human label) **674**, `no_embedding` **418**, `rejected` **2**.
- Reviewed so far: **0** — labeling has not started on this dataset.
- Duplicates rejected by the cooldown filter so far: **55**.
- Trained classifier: **none exists** (`classifier_trained: false`) — this dataset has not been labeled or trained on yet.
- Per-camera activity is uneven (e.g. Tech Section producing a higher `no_embedding` fraction than Main Section 1/2 as of this snapshot) — expected, camera placement/lighting/angle varies; this is exactly the kind of real-world variation the multi-day collection window is meant to sample across, not a fault.

Collecting real data over an extended period (up to the default 7 days) is intended to capture variation in time of day, lighting, face angle, distance, and movement that a short capture window would miss. **This does not by itself guarantee any particular recognition accuracy** — more data improves dataset diversity; actual recognition performance can only be established once real data is labeled, a classifier is trained on it, and that classifier is evaluated against held-out real examples.

## 20. CPU / Performance Considerations

The face detection/tracking/embedding worker is CPU-bound (no GPU/CUDA on this machine — `onnxruntime` runs `CPUExecutionProvider` only). During earlier testing with multiple cameras active, the detection worker was observed using roughly 8 of 12 available CPU threads on average, with higher bursts during inference — enough to visibly affect camera capture smoothness, video encoding, API responsiveness, and general machine responsiveness while it was under that load.

Mitigations already in place:
- Detection is throttled to `FACE_PIPELINE_FPS` (default 3 samples/sec) per camera, not run on every RTSP frame.
- Recognition/embedding runs once per **finished track**, not once per frame — the point of ByteTrack here is exactly to avoid re-embedding the same face dozens of times while it's visible.
- The face pipeline is wrapped in its own try/except per camera stream so a pipeline failure or slowdown can never block the live-video broadcast path itself.

Given this, the guidance for future changes is: don't raise sampling FPS or add more per-frame inference without first confirming there's CPU headroom. If a real accuracy problem shows up, the first lever to reach for is better sampling/tracking/data quality, not simply "run detection more often." Running the face-processing worker in a fully separate process from the FastAPI app (rather than the current in-process background thread per camera) is a reasonable future direction but is **not** currently implemented — see Section 23.

## 21. Implemented vs. Planned Features

### A. IMPLEMENTED
- RTSP camera streaming to the browser over WebSocket, multi-viewer sharing one connection per camera.
- Camera and site CRUD (backend + frontend), including attendance-tracking flag per camera.
- YOLOv8 **face** detection (dedicated face checkpoint), confidence-filtered.
- ByteTrack face tracking, per camera, best-frame-per-track selection.
- Padded-crop handoff from detection to embedding (fixes a real alignment failure).
- InsightFace/ArcFace frozen 512-D embeddings.
- Background, browser-independent, multi-day, multi-camera training-data collection with persistent sessions and automatic expiry.
- Global + per-camera capture limits, cross-track duplicate suppression, quality-based capture filtering.
- Manual labeling tool and workflow, with server-side employee-ID validation.
- Supervised classifier training on frozen embeddings (`LogisticRegression`), with automatic hot-reload into the live matcher.
- Direct photo enrollment endpoint independent of camera detection.
- People page reading real employee data from a separate external service (read-only).

### B. TESTED (see Section 18 for detail)
Detection, embedding, the crop-padding fix, single- and multi-camera real background collection, capture limits, duplicate suppression, the manual labeling UI end-to-end, classifier training end-to-end (on temporary test data), classifier auto-reload, session persistence across restarts.

### C. CURRENTLY RUNNING (see Section 19)
A 7-day background collection session across cameras 3, 4, 5, 6. Not yet labeled, not yet trained on.

### D. NOT IMPLEMENTED — do not describe these as existing
- Full-body person detection or person tracking.
- Person Re-ID (any kind — same-camera or cross-camera).
- A temporal identity fusion / identity state machine.
- An automatic/adaptive gallery (gallery growth today is only via explicit enrollment or explicit review-queue assignment).
- FAISS or any vector-index/ANN search (the current gallery is a plain in-memory cosine-similarity loop over a list — fine at current scale, not a vector DB).
- BoT-SORT (ByteTrack is what's used).
- JWT or any real authentication/authorization for these APIs.
- TensorRT or any GPU-accelerated inference path.
- Multi-model benchmarking/comparison tooling.
- Production deployment infrastructure (this runs as a single local `uvicorn --reload` process).
- PostgreSQL or any database beyond the single SQLite file.
- Attendance/footfall logging triggered by a confident face match (the match happens; nothing downstream consumes it yet).
- Live bounding-box/detection overlay on the frontend video (the websocket for it is a placeholder).

## 22. Current Limitations

- No real authentication — login accepts any email with no password check.
- The employee "recognition" path has no trained classifier yet and an empty enrollment gallery from real camera data, so it currently has nothing meaningful to recognize against — everything falls into the review queue.
- CPU-bound inference limits how many cameras/how high a sample rate can run comfortably on this machine (Section 20).
- Duplicate suppression and per-camera capture budgeting are per-camera only — no cross-camera duplicate suppression or fairness exists yet.
- Several frontend pages (Alerts, Presence/Attendance, Workforce Insights, Footfall, Intrusion, Settings) have no backend behind them — they are UI only.
- `/ws/detections/{camera_id}` and any bounding-box overlay UI are non-functional placeholders.
- SQLite + a single uvicorn worker is adequate for current scale but is not a multi-process/production data layer.
- No automated test suite exists for this pipeline — all verification to date has been manual, against the real running system.

## 23. Future Enhancements

These are genuinely possible future directions, not current features:
- Cross-camera duplicate suppression and dataset balancing across cameras.
- A calibrated "unknown"/rejection threshold once real labeled data and a real trained classifier exist to evaluate against.
- Optional Re-ID as a secondary continuity signal (not a replacement for the classifier).
- GPU acceleration (CUDA execution provider for ONNX Runtime; would materially change the CPU-bound constraints in Section 20) and/or TensorRT.
- A benchmarking pass comparing classifier variants once there's enough labeled real data to make that meaningful.
- Running the face-processing worker as a separate process from the FastAPI app.
- Real authentication (password checking, session/JWT handling) if this app moves beyond local/trusted-network use.
- A production deployment story (process management, and a database choice — SQLite vs. Postgres — appropriate to real concurrent load) once this is no longer a single local dev process.
- A live bounding-box overlay on the video feed, once `/ws/detections` is backed by real pipeline output.
- Wiring a confident recognition match into an actual attendance/footfall record.

None of the above should be described as implemented until it is actually built and verified in this codebase, the same standard this document was written against.
