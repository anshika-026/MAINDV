# Unique Footfall (Re-ID) — standalone export

Everything from the `deco vision backend` repo that implements **unique
footfall via body-appearance Re-ID** — counting *distinct people*, not raw
line crossings — pulled out on its own so it can be dropped into another
portal/repo. Nothing in the original repo was touched to produce this; every
file here is an unmodified copy read straight out of git history
(`HEAD` of `slim/event-3-features`).

Read `UNIQUE_FOOTFALL.md` first — it's the original design doc and explains
*why* the engine is built the way it is (model choice, thresholds, the
margin-based dedup logic). This README only covers what's in this folder and
how to wire it up.

## What's in here

```
app/
  config.py              # REID_*/PEOPLEID_* settings (env-var driven, self-contained)
  reid_db.py              # SQLite: persons, events, embeddings, camera config, unique-count queries
  reid_embedding.py       # OSNet (torchreid) body-appearance embedder
  reid_quality.py         # crop quality gate (size/blur/etc.) before a crop is embedded
  reid_worker.py          # per-frame integration hook — the main entry point (see below)
  reid_api.py             # FastAPI router: /api/reid/* + the REST contract (reference only, see caveats)
  peopleid_tracker.py     # PersonBodyTracker: YOLOv8 + ByteTrack body tracking, shared by reid_worker
  peopleid_gallery.py     # VectorGallery: embedding similarity search + margin-based match decision
  peopleid_fusion.py      # IdentityFusion: temporal voting so one bad frame can't flip an identity
  peopleid_enrollment.py  # curates a track's accumulated crops into an identity's reference embeddings
  peopleid_db.py          # used by peopleid_fusion for track-state persistence
  peopleid_reid.py        # used by peopleid_fusion
scripts/
  demo_reid.py            # run the WHOLE engine against a local video file, no camera/API/DB server needed
  fetch_reid_model.py     # downloads the MSMT17-trained OSNet checkpoint (see UNIQUE_FOOTFALL.md)
  eval_reid_weights.py    # measures same-person vs different-person embedding separation on real crops
  replay_reid_identities.py  # replays saved crops through the matching pipeline, writes per-identity montages
  reset_reid_identities.py   # wipes all identities (e.g. after swapping the model)
tests/
  test_reid_unique_footfall.py  # the primary spec: person A ten times = 1, A+B = 2, A returning stays 2, etc.
UNIQUE_FOOTFALL.md         # original design doc — read this
requirements.txt           # trimmed to just what this engine needs
```

## Fastest way to see it work

```bash
cd unique-footfall-export
pip install -r requirements.txt
python -m scripts.fetch_reid_model        # optional but strongly recommended — see UNIQUE_FOOTFALL.md
python -m scripts.demo_reid --video path/to/some_video.mp4 --show
```

`demo_reid.py` is a complete, self-contained driver — no RTSP, no FastAPI, no
auth, no orchestrator. Read it as the reference for how to call the engine
yourself: it builds a `PersonBodyTracker`, a `BodyReIdEmbedder`, a
`VectorGallery`, and an `IdentityFusion`, then feeds it frames.

## Wiring it into a live camera feed / your portal

The real entry point for a live feed is **`reid_worker.process_frame(camera_id,
frame, state)`** (`app/reid_worker.py`). Call it once per camera per frame
(it internally rate-limits itself to `REID_MOT_INTERVAL_SECONDS`, so calling
it every frame is fine — it'll no-op until the cadence elapses and return
`None`).

```python
from app import reid_db, config
from app.reid_worker import ReidWorkerState, process_frame

reid_db.init_db()                          # once at startup
state = ReidWorkerState(camera_config={"enabled": True})  # one instance per camera, kept alive across frames

# per frame:
result = process_frame(camera_id, frame, state)
if result:
    for track in result["tracks"]:
        ...  # track["state"] is "unknown" | "candidate" | "confirmed"
             # track["person_id"] is set once confirmed
    if any("pending_new_person" in t for t in result["tracks"]):
        ...  # see "Auto-enrollment" below — this is the one part reid_worker
             # deliberately does NOT do for you
```

### Auto-enrollment — the one piece of glue you have to write

`reid_worker.py` never writes to `reid_db` directly (that boundary is
explained in its own docstring — it's designed to run inside a worker
process/thread separate from the process doing SQLite writes). When a track
accumulates enough unmatched, quality-passing samples, `process_frame`
returns `pending_new_person` — a list of curated embeddings/crops — and it's
the caller's job to actually create the person:

```python
if "pending_new_person" in track:
    person_id = reid_db.create_person(now=now)
    for sample in track["pending_new_person"]:
        reid_db.add_embedding(person_id, sample["embedding"], sample["quality_score"])
        # persist sample["crop_jpeg"] under reid_db.SNAPSHOTS_DIR if you want snapshot photos
    # then rebuild every live worker's gallery (VectorGallery caches embeddings
    # in-process) so the new identity is matchable — see reid_db.load_all_embeddings
```

If you don't need the full multi-camera-worker-process architecture the
original app uses, `demo_reid.py` shows the simpler single-process version of
this same flow (it calls `reid_db` directly, no pending/dispatch split at
all) — that's the easier starting point for a smaller portal.

### `reid_api.py` — reference, not drop-in

This is the original REST contract (`/api/reid/persons`, `/api/reid/footfall/today`,
`/api/reid/events`, `/api/reid/stats`, `/api/reid/cameras/{id}/config`, …) and
is worth keeping as the shape to replicate, but it won't run as-is here: it
imports `auth` (JWT/role checks) and `pipeline_manager` (the original app's
camera-worker orchestrator — `is_live()`, `restart_camera_worker()`,
`reload_reid_gallery()`), neither of which is in this export since they're
generic app infrastructure, not footfall-specific. Swap those two imports for
your own portal's auth and camera-worker-manager equivalents and the rest of
the router works unchanged against `reid_db.py` as included here.

Minimal read-only version if you don't need per-camera admin config yet:

```python
from fastapi import APIRouter
from app import reid_db

router = APIRouter(prefix="/api/reid", tags=["reid"])

@router.get("/footfall/today")
def footfall_today():
    return {
        "unique_count": reid_db.count_unique_today(),
        "new_today": reid_db.count_new_today(),
        "returning_today": reid_db.count_returning_today(),
    }
```

## Config

Everything is env-var driven via `app/config.py` (`.env` file or real env
vars, using `python-dotenv`). The relevant settings are `REID_*` (the engine
itself) and `PEOPLEID_*` (the tracker/gallery/fusion classes the Re-ID engine
reuses — see `config.py`'s comments around line 577 for why they're a
separate-but-parallel namespace from `REID_*`). Defaults are sane; the one
you'll actually want to set is `REID_MODEL_PATH` after running
`fetch_reid_model.py`, and `REID_MOT_MODEL_PATH` / `PEOPLEID_MOT_MODEL_PATH`
(YOLOv8n weights — `yolov8n.pt`, auto-downloaded by ultralytics on first use
if not present).

## What was deliberately left out

Two other, unrelated "footfall" implementations exist in the source repo and
were **not** included here, because they're a different feature despite the
name overlap:

- **Raw line-crossing counting** (`footfall_db.py`, `footfall_gate_db.py`,
  `gate_tracker.py`) — counts every crossing of a drawn line, no identity, no
  dedup (the same person walking past 10 times = 10). Different question from
  "how many distinct people."
- **Face-embedding footfall** (`footfall_counter.py`) — an older, disabled
  (per `UNIQUE_FOOTFALL.md`: "off for this deployment") face-based dedup
  attempt, superseded by this Re-ID engine.

Also not included: `main.py`'s startup wiring (`reid_db.init_db()`,
`app.include_router(reid_router)`) and `pipeline.py`'s
`_dispatch_reid_result` (the original app's specific glue between its
multi-process camera workers and its FastAPI process) — those are
infrastructure of the *original* app, not portable, and your portal will
have its own equivalents. The "Wiring it in" section above is the
replacement for that glue.
