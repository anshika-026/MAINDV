# Deco Vision — Object Detection Module

Detects 4 classes on live camera feeds: **backpack, handbag, bottle, laptop**
(cell phone and chair were tried and dropped — see notes in `detector.py`
and `.env.example` for why). Built as a specialist model + zero-shot
ensemble, tuned and battle-tested against real live CCTV footage, not
just offline metrics.

## What's in here

- `detector.py` — the main entry point. Call `run_full_detection(frame,
  object_tracker=None, run_ppe=False)` with a BGR frame (OpenCV format)
  to get back a dict with `objects` (list of detections: name, bbox,
  confidence, color, track_id).
- `object_tracker.py` — cross-frame tracking (assigns stable `track_id`s,
  smooths out single-frame misses instead of instantly hiding a track).
- `ppe_detector.py`, `fixtures_detector.py`, `color_detector.py` — direct
  dependencies of `detector.py` (helmet/mask detection, office fixture
  classification, dominant-color extraction). Pass `run_ppe=False` if you
  don't need the PPE output; they still need to be importable.
- `config.py`, `video_source.py` — camera/env config and RTSP frame
  capture helpers.
- `models/` — the model weights this actually uses:
  - `imc_general_best.pt` — fine-tuned specialist for backpack/handbag/bottle.
  - `yolo26s.pt` — general COCO model (also gives laptop, person, and
    office-furniture classes for free).
  - `office_fixtures.pt` — office fixture classifier (AC/cabinet/fan/etc),
    optional, only used for the dashboard's fixture overlay.

  Two more models are used by the zero-shot fallback ensemble but are
  **not** in this folder — Ultralytics auto-downloads them into its own
  cache on first use, no setup needed: `rtdetr-l.pt` (YOLO-World/RT-DETR
  fallback for objects the specialist misses) and `yolov8s-world.pt`.
  CLIP (`openai/clip-vit-base-patch32`) is pulled automatically via
  `transformers` the first time backpack/handbag disambiguation runs.

## Architecture (why it's not just one model)

1. **Specialist model** (`imc_general_best.pt`) — fine-tuned on ~32k
   images (COCO subset + LVIS + Open Images + Roboflow + real office
   footage) for the 4 target classes. High precision, moderate recall.
2. **Zero-shot fallback** (YOLO-World-S + RT-DETR-L) — catches what the
   specialist misses, throttled (`YOLOS_FALLBACK_MIN_INTERVAL_S`) since
   it's expensive on CPU.
3. **Weighted Boxes Fusion** — merges specialist + fallback boxes for the
   same class instead of showing duplicates.
4. **CLIP disambiguation** — backpack and handbag get confused with each
   other more than any other pair (COCO's own class boundary is blurry
   here); when two boxes overlap and are close in confidence, or a lone
   low-confidence guess needs a second opinion, CLIP zero-shot re-checks
   the crop.
5. **Full-frame sanity filter** — every box gets checked against a
   plausible area-fraction range for its class, independent of
   confidence (catches "oversized box" bugs no threshold alone fixes).
6. **Cross-source dedup** — the same physical object can get boxed by
   more than one of the paths above (e.g. two overlapping person-crops);
   deduped by IoU, highest confidence wins.

## Setup

```bash
pip install ultralytics opencv-python numpy python-dotenv transformers torch ensemble-boxes
cp .env.example .env   # fill in real camera credentials
```

GPU is auto-detected (`torch.cuda.is_available()`) and used automatically
if present — no config needed. On CPU-only hardware, expect 5-100s per
detection cycle depending on load; on GPU this drops to well under 1s,
which is what makes real-time multi-camera detection and smooth
per-frame box tracking (not just periodic updates) actually practical.

## Usage

```python
import cv2
from detector import run_full_detection
from object_tracker import ObjectTracker

tracker = ObjectTracker()  # one per camera - keeps track_ids stable
frame = cv2.imread("some_frame.jpg")
result = run_full_detection(frame, object_tracker=tracker, run_ppe=False)

for obj in result["objects"]:
    print(obj["name"], obj["confidence"], obj["bbox"])
```

## Tuning notes

Every threshold in `.env.example` has a comment explaining *why* it's
set where it is, usually citing a specific real-footage measurement
that justified it (e.g. a genuine bottle scoring 0.252 that a naive
0.30 floor would silently drop). Read those before changing anything —
several of them look "obviously wrong" in isolation (a 0.15 confidence
floor looks too permissive) but exist because the naive value was
measured to actively cost real detections or actively introduce false
positives. If you change hardware (especially moving to GPU), several
of the CPU-driven throttle intervals (`YOLOS_FALLBACK_MIN_INTERVAL_S`,
`OBJECT_DETECTION_FPS`) can likely be tightened back down.
