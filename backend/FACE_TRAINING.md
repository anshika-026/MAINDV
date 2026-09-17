# Face training dataset — camera capture → manual labeling → classifier training

This documents the manual-labeling + classifier-training pipeline built on
top of the live face-recognition pipeline (see `FACE_RECOGNITION_WIRING.md`
for that one). Read that one first if you haven't — this assumes it.

## What was found, before anything was built

- **Detection**: `ultralytics.YOLO`, expecting a face-tuned checkpoint.
  Confirmed by actually running it: **the weight file doesn't exist**
  anywhere in this repo or machine, so live detection/tracking currently
  does nothing (fails safely — see `FACE_RECOGNITION_WIRING.md`).
- **Embedding**: InsightFace `buffalo_l` (ArcFace, 512-dim, frozen, CPU-only
  — no CUDA on this box).
- **Employee IDs**: not stored anywhere in this repo before this change. The
  only real employee_id↔name mapping lived on the separate external service
  (`13.61.58.14/api/faces`, not part of this repo). `face_embeddings` and
  `face_pending` accepted any string as a person_id with zero validation.
- **`face_pending`**: already existed (added in a previous session) —
  columns `id, camera_id, track_id, captured_at, image_path, embedding,
  best_match_person_id, best_match_score, assigned_person_id, status`. It
  only ever receives a capture when a track's match score is **below**
  `MATCH_THRESHOLD`, deduped to once per track per 60s — a live-correction
  queue, not a bulk collector. At that rate it could never produce
  10-15k images/day, which is why it's left untouched and a separate table
  was added instead of repurposing it.
- **Recognition**: per camera, YOLO → ByteTrack (via `supervision`) → best
  crop per finished track → InsightFace re-detect+align → ArcFace embed →
  cosine similarity against `face_embeddings`.
- **A real bug found while testing this**: `CAPTURE_DIR` / `ENROLL_DIR` /
  the new `TRAINING_CAPTURE_DIR` all defaulted to a relative path
  (`"backend/data/..."`) assuming the process runs from the repo root. This
  project always launches uvicorn from *inside* `backend/`
  (`cd backend && uvicorn app.main:app`), so that default silently created a
  nested `backend/backend/data/...` tree. Fixed by resolving relative to
  each file's own location (`Path(__file__).resolve().parent.parent /
  "data"`), matching the convention `camera_db.py`/`face_db.py` already
  used for `DB_PATH`. Caught by actually running the label endpoint and
  checking where the file landed — not by reading the code.

## Architecture decision: Option B, not fine-tuning

Kept ArcFace frozen; added a supervised classifier (`sklearn
LogisticRegression`) trained on top of its embeddings, using real,
human-labeled camera captures. **Not** fine-tuning the embedding network
itself. Why:

- Fine-tuning ArcFace needs GPU-scale batches, careful margin-loss
  engineering (that's literally what "ArcFace" the loss function is), and
  real risk of degrading a well-trained general face-embedding space for
  marginal gain from ~20-40 employee classes. None of that fits a CPU-only
  box (`torch 2.14.0+cpu`, confirmed no CUDA).
- A classifier on frozen embeddings is the standard, well-established
  pattern for exactly this situation (small closed set of known identities,
  limited compute) — fast to train even on CPU (sub-second for thousands of
  512-dim vectors), trivial to retrain as more labels come in, and safe to
  swap in/out without touching the feature extractor at all.
- It's a direct, honest use of what you're actually building: a
  human-verified, real-camera-conditions dataset is exactly what improves a
  classifier's decision boundary. It does nothing for ArcFace's own
  weights, which is fine — that's not what's broken.

## What was different from the given spec

1. **Employee ID validation source**: the spec asked to validate against
   "the existing employee/enrollment data" but that data doesn't live in
   this repo (see above). Added a minimal local `employees` table
   (`employee_id TEXT PRIMARY KEY, name`) used only for this server-side
   check — not a new ID scheme, just a place to say "018 is a real ID"
   without depending on the external service at request time. Seeded via
   an explicit, re-runnable `POST /api/faces/training/employees/sync`
   (never automatic) that pulls from `13.61.58.14/api/faces` once; you can
   also add one manually with `POST /api/faces/training/employees`.
2. **The relative-path bug** above — fixed, not part of the original ask
   but directly caused by it (new `TRAINING_CAPTURE_DIR` hit the same bug
   `CAPTURE_DIR`/`ENROLL_DIR` already had).
3. **Labeling page is a React route** (`/face-training` in the existing
   frontend, port 5180), not a FastAPI-served page at `127.0.0.1:8821`. The
   frontend already has full routing, styling and a dev server; building a
   second, separate HTML page inside the Python backend would duplicate all
   of that. CORS already permits this exact cross-origin pattern (see
   `backend/.env`'s `CORS_ORIGINS` and `BACKEND_HANDOFF.md`) — nothing
   needed to change there.
4. **Quality metadata**: kept to the two cheap ones explicitly suggested
   (blur via variance-of-Laplacian, brightness via mean grayscale) plus
   detection confidence — no additional filtering logic in this first
   version, exactly per "don't overcomplicate the first version."

## Database changes (`backend/app/face_db.py`)

Two new tables, created by the existing `init_face_tables()` (already
called on startup, no new startup hook needed):

- **`employees`**: `employee_id TEXT PRIMARY KEY, name TEXT NOT NULL`
- **`face_training_captures`**: `id, camera_id, track_id, captured_at,
  image_path, embedding (nullable), detection_confidence, blur_score,
  brightness, employee_id (nullable), label_status ('unlabeled'|'labeled'|
  'skipped', default 'unlabeled'), labeled_at`

New CRUD functions: `upsert_employee`, `employee_exists`, `list_employees`,
`add_training_capture`, `get_training_capture`, `get_next_unlabeled_capture`
(oldest-first), `get_training_stats`, `label_training_capture` (single
transaction, validates status before writing, rolls back on failure),
`skip_training_capture`, `get_labeled_training_embeddings` (the actual
training set).

## Files changed / added

- `backend/app/face_db.py` — schema + CRUD above.
- `backend/app/face_pipeline.py` — every finished track now also calls
  `_save_training_capture()` unconditionally (confident match, low
  confidence, or no embedding at all), separate from the existing
  low-confidence-only `face_pending` path. Added `_quality_metrics()`,
  classifier-aware `_match()` (see below), fixed the path bug.
- `backend/app/face_training.py` — **new**. `train_classifier()`, the one
  explicit, separate training operation. Requires >= 2 employees with >= 2
  labeled captures each; raises `ValueError` with the actual label counts
  otherwise (visible as a 422 from the API) rather than failing silently or
  training garbage.
- `backend/app/face_training_routes.py` — **new**. All
  `/api/faces/training/*` endpoints (list below).
- `backend/app/face_routes.py` — fixed the same path bug in `ENROLL_DIR`.
- `backend/app/main.py` — `app.include_router(face_training_routes.router)`.
- `backend/requirements.txt` — added `scikit-learn==1.9.1`,
  `joblib==1.6.0` (both were already present transitively, pinned
  explicitly for a reproducible install).
- `frontend/src/api/client.js` — `getNextTrainingCapture`,
  `getTrainingStats`, `trainingImageUrl`, `labelTrainingCapture`,
  `skipTrainingCapture`, plus a dedicated `trainingRequest()` helper that
  surfaces FastAPI's `detail` string directly (the labeling UI shows it
  verbatim, e.g. "Unknown employee_id '999'...").
- `frontend/src/pages/FaceTraining.jsx` — **new**. The labeling page.
- `frontend/src/App.jsx` — added the `/face-training` route
  (auth-protected, but outside `AppShell` — no sidebar, matches the
  fast-repeat-task design in the spec).

## APIs added (`/api/faces/training/*`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/next` | Next unlabeled capture's metadata (no image binary) + `{reviewed, total}` |
| GET | `/stats` | `{reviewed, total}` only, for polling |
| GET | `/image/{capture_id}` | The actual JPEG — path resolved server-side from the id, client never supplies a path |
| POST | `/label` | `{capture_id, employee_id}` — validates employee_id locally, moves the file into `<employee_id>/`, single-transaction DB update |
| POST | `/skip` | `{capture_id}` |
| GET | `/employees` | Local roster (debugging) |
| POST | `/employees` | `{employee_id, name}` — add one manually |
| POST | `/employees/sync` | Explicit pull from the external faces API — never automatic |
| POST | `/train` | Runs `train_classifier()`; 422 with label counts if not enough data |
| GET | `/model-status` | Whether a trained classifier file exists, and when |

Security, per the spec's requirements: no camera passwords/RTSP credentials
are exposed by any of these; `/image/{capture_id}` never trusts a
client-supplied path; `employee_id` is validated against the local
`employees` table before a label is ever written; `label_training_capture`
uses a single sqlite transaction with rollback so a capture can't end up
half-labeled (and `/label` itself moves the file back if the DB write fails
after the move succeeded).

## How to open the labeling page

`http://localhost:5180/face-training` (frontend dev server must be
running; you'll be redirected to `/login` first if not already
authenticated — any email/password works, this app has no real password
check yet, see `BACKEND_HANDOFF.md`).

Verified end-to-end with an actual headless-browser session (not just curl):
loaded a real capture, typed an employee ID, pressed Enter, watched it save
and advance to "All caught up" with the counter incrementing — no page
reload, no console errors.

## Bootstrap: getting real captures in

Nothing arrives in the queue until live detection actually runs, which
needs the YOLO face weight file (see `FACE_RECOGNITION_WIRING.md` — still
missing, deliberately not auto-downloaded). Once it's in place and cameras
are running, `face_training_captures` fills up on its own. To seed the
local employee roster first: `POST /api/faces/training/employees/sync`.

## How to start training

Once you've labeled at least 2 employees with at least 2 captures each:

```bash
curl -X POST http://127.0.0.1:8821/api/faces/training/train
```

Returns `{"trained_on_samples": N, "employees": [...], "excluded_too_few_samples": {...}, "model_path": "..."}`.
Re-run any time you've labeled more — it always retrains from scratch on
the full current label set, there's no incremental/online update.

## How the trained model connects back to recognition

`CameraFacePipeline._match()` (in `face_pipeline.py`) checks for a trained
classifier file (`CameraFacePipeline._get_classifier()`) on every match:

- **If present**: uses `clf.predict_proba()`, compares the winning class's
  probability against `FACE_CLASSIFIER_MIN_PROBA` (env var, default 0.6).
- **If absent**: falls back to the original cosine-similarity-vs-gallery
  approach (`MATCH_THRESHOLD`, default 0.45) — this is the only mode
  available before you ever run `/train`.

This is automatic and requires no code change or restart — `_get_classifier()`
checks the file's mtime on each call and reloads if it changed, so a
freshly completed training run takes effect on the very next match while
the server keeps running.

## How to verify a newly trained model is actually being used

- `GET /api/faces/training/model-status` — `classifier_trained: true` and a
  `trained_at` timestamp matching your last `/train` call confirms the file
  exists and when it was written.
- Watch the backend log after retraining: nothing logs the mode switch
  explicitly today (kept minimal for this first version), but you can
  confirm behaviorally — feed a labeled employee's face again and check
  `face_pending`/recognition results reflect classifier-style probability
  scores (typically higher-confidence, sharper separation) rather than raw
  cosine similarities.
- **Directly verified this session**: trained a classifier on 4 real,
  human-labeled camera-style captures across 2 employees, then ran
  `_match()` against two different, held-out photos of those same two
  people (not used in training) — both were correctly classified with the
  right employee_id. Test data was removed afterward; the database and
  `data/face_training/` are back to a clean, empty state, with the 21-entry
  employee roster (from a real `/employees/sync` run) kept since it's
  genuinely useful going forward.

## Making 24-hour collection production-ready

A follow-up audit (before any 24-hour run) found five real gaps, all fixed
below. No changes to YOLO, InsightFace/ArcFace, `_embed()`, `_match()`,
the classifier architecture/training code, employee IDs, or the manual
labeling workflow — this section is entirely about the capture/ingestion
side.

### 1. Background collection, independent of any browser

Previously, a capture only happened while `/ws/live/{camera_id}` had a real
subscriber — no browser tab open meant no collection at all, regardless of
foot traffic.

**`backend/app/face_collection.py` (new)**: `start(camera_id)` /
`stop(camera_id)` / `stop_all()` / `status()`. Reuses
`camera_stream.CameraStream.subscribe()/unsubscribe()` unchanged — a
`_CollectorSink` object with a no-op `put_nowait()` just occupies
`CameraStream._subscribers` so the existing per-camera RTSP thread (and
therefore `feed_frame()`) keeps running with zero real viewers. No second
RTSP connection, no new thread, no frontend involvement — it lives entirely
server-side and starts/stops only when explicitly told to.

New endpoints (in `face_training_routes.py`):
- `POST /api/faces/training/collection/start` `{"camera_id": N}`
- `POST /api/faces/training/collection/stop` `{"camera_id": N}`
- `GET /api/faces/training/collection/status` → active cameras + current
  capture count vs. the limit

Because it's the same `CameraStream`, it automatically inherits the
existing RTSP reconnect handling in `camera_stream.py` — nothing new was
needed for "survive a disconnect."

### 2. Global hard capture limit

`MAX_TRAINING_CAPTURES` (env var, default **15000**) in `face_pipeline.py`.
Enforced atomically: `face_db.capture_limit_lock` (a module-level
`threading.Lock`) guards a `count_training_captures()` check immediately
before every insert in `_save_training_capture()`, so concurrent camera
threads can never overshoot it, even by one. The count is across **all**
cameras and **all** statuses combined (unlabeled/labeled/skipped/
no_embedding/rejected) — once reached, new captures are silently dropped
(not written to disk, not inserted); existing rows are never touched.

### 3. Cross-track duplicate protection

`CameraFacePipeline._is_recent_duplicate()` — reuses the exact cosine-
similarity math `_match()` already does, no new model. Each pipeline
instance keeps a short, per-camera, in-memory list of `(timestamp,
embedding)` for captures with a real embedding. Before saving a new one, if
its cosine similarity to anything in that list (pruned to the last
`FACE_DEDUP_COOLDOWN_SECONDS`, default **120s**) is `>=
FACE_DEDUP_SIMILARITY_THRESHOLD` (default **0.7**), it's dropped entirely
— no file, no DB row.

Deliberately narrow, per the requirement not to eliminate useful
variation: per-camera only, and only a 2-minute window — this catches
"ByteTrack lost and re-acquired the same person 10 seconds later" (a
tracking artifact), not "the same employee walked past again an hour
later" or "seen on a different camera," both of which are legitimate,
separate training samples and are captured normally. **Verified with a
controlled test**: same real embedding submitted twice within the cooldown
→ second one deduped (capture count +0); a different real person's
embedding submitted right after → inserted normally (+1).

### 4. Embedding-less captures no longer pollute the labeling queue

`face_training_captures.label_status` gained two new values, set at insert
time in `_save_training_capture()`:
- **`no_embedding`** — YOLO found a face but `_embed()` returned `None`.
- **`rejected`** — has an embedding, but fails the quality gate (below).

Neither is ever inserted as `'unlabeled'`, so `get_next_unlabeled_capture()`
(unchanged — still just `WHERE label_status = 'unlabeled'`) never surfaces
them in `/face-training`, and `get_labeled_training_embeddings()` (also
unchanged) never trains on them. Nothing is deleted — both remain on disk
and in the DB, queryable via `GET /api/faces/training/stats`'s new
`by_status` breakdown, for diagnostics. `get_training_stats()`'s
`reviewed`/`total` counters were also rescoped to only
unlabeled/labeled/skipped, so the `/face-training` UI's "X / Y reviewed"
counter isn't inflated by captures a human never actually saw.

### 5. Quality filtering — thresholds from real measured data, not guesses

Applied only to captures that already have an embedding (an embedding-less
capture is always `no_embedding` regardless of quality). All configurable
via env vars, in `face_pipeline.py`:

| Constant | Default | Based on (real captures from this session) |
|---|---|---|
| `FACE_MIN_BLUR_SCORE` | 50 | Observed range 2776–4432 (variance of Laplacian) — the floor sits ~55-90x below every real sample, so it only rejects genuinely degenerate frames |
| `FACE_MIN_BRIGHTNESS` / `FACE_MAX_BRIGHTNESS` | 20 / 235 | Observed range 109–138 (mean grayscale) — wide margin on both sides, only catches near-black/near-blown-out frames |
| `FACE_MIN_AREA` | 900 px² | Observed approx. tight-bbox areas: the two captures that *failed* to embed were 1517 and 2022 px²; the two that succeeded were 2400 and 2654 px². Set well below even the failing samples deliberately — not tuned to this n=4 sample, just a sanity floor against genuinely tiny/distant detections |

None of these were picked to hit a target rejection rate — they're
conservative floors documented against the actual numbers observed,
per the instruction not to invent aggressive thresholds on a small sample.

### Verified (tests A–J, all against the real running backend, not mocked)

| # | What | Result |
|---|---|---|
| A | Background collection runs with zero browser/websocket viewers | **PASS** — started via API only, `netstat` showed no ESTABLISHED connections throughout |
| B | One camera produces real captures | **PASS** — camera 3, 7 new captures over ~75s unattended |
| C | Valid JPEGs + valid ArcFace embeddings | **PASS** — 5/7 had real 512-dim embeddings, all image files verified on disk |
| D | `employee_id` NULL, correct `label_status` | **PASS** |
| E | Embedding-less captures excluded from the labeling queue | **PASS** — confirmed directly in the DB (`no_embedding` status) and by the unchanged, already-filtered `/next` query |
| F | Global limit never exceeded | **PASS** — restarted with `MAX_TRAINING_CAPTURES=14` (12 existing + margin), ran collection 90s, landed at exactly 14, `limit_reached: true` |
| G | Cross-track dedup, controlled test | **PASS** — same embedding twice within cooldown → +1 then +0; different person right after → +1 |
| H | Existing live-view unaffected | **PASS** |
| I | Existing `/face-training` labeling unaffected | **PASS** |
| J | Explicit classifier training unaffected | **PASS** — trained on 4 real labeled samples across 2 employees |

All test-generated captures, images, and the test-only classifier were
removed afterward; `data/face_training/` and `face_training_captures` are
back to empty, the 21-entry real employee roster was kept.

### Remaining limitations, going into a real 24-hour run

- **Still no automatic "restart collection on backend restart" or
  persistence of which cameras were collecting** — if the backend process
  restarts mid-collection, you'll need to call `/collection/start` again
  for each camera. Not implemented (wasn't asked for); worth knowing before
  a real unattended 24-hour run.
- **The dedup window is per-camera, not cross-camera** — the same person
  visible on two adjacent cameras within the cooldown window would still
  produce two captures (by design — different cameras are different
  vantage points, arguably legitimately different training samples, but
  worth knowing).
- **`MAX_TRAINING_CAPTURES` is a blunt global cutoff, not a per-camera
  fair-share** — if one busy camera fills the quota first, quieter cameras
  collecting at the same time could end up under-represented. Not fixed,
  since the requirement was "never exceed," not "distribute fairly."
