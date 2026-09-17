# Face Recognition — Deco Vision

Face detection, tracking, and recognition for the 4 live RTSP cameras, plus
a manual review workflow for training the recognition gallery against
already-enrolled employee IDs.

## How this works (read this first)

There is no classifier and no training run. **ArcFace is a frozen,
pretrained embedding model** — it is never retrained. What actually gets
"trained" is a lookup table in SQLite: `person_id → many embedding vectors`.

- **Enrollment / labeling** = appending a new vector to a person's bucket.
- **Recognition** = comparing a new face's vector against every vector in
  the table (cosine similarity) and taking the closest match.

More labeled photos per person → better matching on future camera frames.
There is no separate "build the model" step; assigning an ID to a captured
photo *is* the training action, and it's instant.

## Pipeline

```
RTSP camera (existing camera_stream.py loop)
        │
        ▼
YOLO-face detection  →  ByteTrack (tracks a face across frames)
        │
        ▼
best frame per track (sharpest/largest — avoids re-processing every frame)
        │
        ▼
InsightFace FaceAnalysis: align + embed (512-dim ArcFace vector)
        │
        ▼
cosine similarity vs. gallery
        │
   ┌────┴─────┐
   ▼          ▼
match ≥      below
threshold    threshold
   │            │
auto-tagged   → face_pending (waits for manual review)
```

## Tech stack

| Stage | Library |
|---|---|
| Face detection | YOLOv8-face (fine-tuned weights — **not** generic `yolov8n.pt`) |
| Tracking | `supervision`'s ByteTrack (pure Python, no native toolchain) |
| Embedding | InsightFace `buffalo_l` (ArcFace) |
| Matching | in-memory cosine similarity over embeddings cached from SQLite |
| Storage | raw `sqlite3`, no ORM — matches existing `camera_db.py` style |

## Files

- `backend/app/face_db.py` — schema + CRUD for the embeddings gallery and
  the `face_pending` review queue
- `backend/app/face_pipeline.py` — per-camera worker: detect → track →
  embed → match/queue. `feed_frame()` is called from the existing
  `camera_stream.py` read loop — no second connection to the camera.
- `backend/app/face_routes.py` — FastAPI routes (below)
- `tools/face_review.html` — standalone local page for manual labeling

## Setup

```bash
pip install -r backend/requirements.txt
```

Set `YOLO_FACE_WEIGHTS` (env var or `backend/.env`) to a face-specific
YOLOv8 checkpoint — e.g. from `akanametov/yolo-face` or
`derronqi/yolov8-face`. **Vet the source yourself before loading it** — it's
a pickle-based checkpoint, and loading an untrusted one can execute
arbitrary code. Until this file exists, live camera streaming works
normally and detection stays silently off.

InsightFace's `buffalo_l` model downloads itself automatically on first run
(needs internet access once).

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/faces/enroll` | Bootstrap: directly add a known photo for a `person_id` |
| GET | `/api/faces/pending?hours=24` | List unresolved camera captures |
| GET | `/api/faces/pending/{id}/image` | Serve a capture's image for review |
| POST | `/api/faces/assign` | `{pending_id, person_id}` — labels a capture, appends its embedding to that person's gallery |
| POST | `/api/faces/ignore` | `{pending_id}` — discard a bad/unusable capture |
| GET | `/api/faces/gallery/{person_id}/count` | How many reference embeddings a person has |

## Step 1 — Bootstrap the gallery

Before cameras have anything to match against, seed the gallery from
existing enrollment photos (e.g. from the People page / external faces
service):

```bash
curl -X POST http://127.0.0.1:8821/api/faces/enroll \
  -F "person_id=018" \
  -F "photo=@/path/to/photo.jpg"
```

Repeat for every enrolled person's existing photo(s).

## Step 2 — Let cameras run

Once `YOLO_FACE_WEIGHTS` is set and `camera_stream.py` is calling
`pipeline.feed_frame(frame)`, every camera continuously detects, tracks,
and embeds faces. Confident matches are silently tagged; anything below
`MATCH_THRESHOLD` (env var, default `0.45`) lands in `face_pending`.

## Step 3 — Manual review / training

Open `tools/face_review.html` in a browser (double-click, or VS Code Live
Server). It shows one pending capture at a time:

- photo, camera, timestamp, and the pipeline's best guess + confidence if
  it has one
- type the employee ID → **Enter** → saved via `/api/faces/assign`, next
  photo appears immediately
- **Esc** or Skip → `/api/faces/ignore`, moves on
- polls every 30s to pick up newly-queued captures without a manual refresh

If opening the page from `file://` or a dev server other than the backend
itself, make sure `CORS_ORIGINS` in `backend/.env` allows that origin (or
set it to `*` temporarily for local review sessions).

## Tuning

- **`MATCH_THRESHOLD`** — the main lever for false "unknown" results. If
  correct matches keep landing in the review queue, lower it slightly
  (e.g. 0.4). If two different people are ever auto-matched to the same
  ID, raise it.
- **`FACE_PIPELINE_FPS`** (default 3) — detection sample rate per camera,
  independent of the stream's actual FPS. Keeps 4 concurrent cameras from
  overloading the CPU/GPU; 2–5 fps is enough to catch someone walking past.
- More labeled photos per person = fewer future review-queue hits for
  them — this compounds the more you review.

## Known issues

- On Windows, `uvicorn --reload` can hang after cameras have been
  streaming, because blocking `cv2.VideoCapture` reads in a background
  thread don't respond cleanly to the reloader's restart signal. If the
  backend seems stuck after an edit while cameras are live, kill and
  restart manually rather than waiting on it.
- CP Plus NVR cameras can temporarily lock (401) after repeated failed
  connection attempts — back off rather than retrying in a loop.
