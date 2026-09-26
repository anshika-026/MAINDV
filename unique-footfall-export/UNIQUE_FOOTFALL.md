# Unique Footfall (person Re-ID)

Counts **distinct people**, not visits. The same person entering ten times is
`1`, not `10`, even though their tracker ID changes every time they leave and
come back.

This is implemented by the `reid_*` modules in `app/`. It is separate from,
and should not be confused with, the two other counters in this codebase:

| Number | Source | What it means |
|---|---|---|
| **Unique footfall** | `reid_db.count_unique_today()` | distinct people (this system) |
| People counted | `face_db.get_people_counting_report()` | raw line crossings — counts the same person on every pass |
| Footfall visits | `footfall_db` (face embeddings) | older face-based dedup, **off** for this deployment (see below) |

## Pipeline

```
RTSP → frame capture → YOLO person detection → ByteTrack (track_id)
     → quality filter → body crop → OSNet Re-ID embedding
     → vector search over stored identities → temporal confirmation
     → persistent PERSON_ID → unique footfall → SQLite → REST/WebSocket
```

`track_id` is temporary and per-appearance. `person_id` is the durable
identity, and **only a newly confirmed `person_id` may increment footfall**
(`reid_worker.py`, `pipeline.py::_dispatch_reid_result`).

## The model matters more than the thresholds

The engine's accuracy is bounded by how well the embedding separates people.
Measured on 186 real crops from this deployment's own reception camera
(`python -m scripts.eval_reid_weights`):

| Weights | same person | different people | separation |
|---|---|---|---|
| ImageNet backbone (torchreid default) | median 0.862 | median 0.688 | **+0.174** |
| **MSMT17 Re-ID-trained** (in use) | median 0.858 | median 0.445 | **+0.413** |

The ImageNet backbone is a general image classifier, never trained to tell
people apart. It could not separate identities at *any* threshold: a third of
different-person pairs scored above 0.72 while same-person pairs fell as low
as 0.63. In practice that produced **34 "unique" people in one hour** on a
single reception door, via both failure modes at once — genuinely different
people scoring 0.86-0.89 (merged), and the same person landing too close to
the runner-up to pass the `REID_MIN_MARGIN` decisiveness check, so each
appearance spawned a fresh identity.

The MSMT17 checkpoint is the same `osnet_x0_25` architecture — identical
speed and memory, weights-only swap — trained for person Re-ID. It scored all
nine visually-verified crop pairs correctly (same 0.94-0.95, different
0.40-0.50).

Get it (it is gitignored, not committed):

```bash
cd backend
python -m scripts.fetch_reid_model
```

Without it the engine still boots on the ImageNet fallback and will
over-count badly. `scripts/eval_reid_weights.py` re-runs the comparison above
against whatever crops the deployment has collected.

## The decisiveness margin had a feedback loop (fixed)

Better weights alone made things *worse* — replaying the same 186 crops
produced **131** identities, against 80 for the old model. The embedding was
not the whole problem.

`REID_MIN_MARGIN` requires the best-matching person to beat the runner-up
person by 0.05, so that genuinely ambiguous scores resolve to `UNKNOWN`
instead of being forced into an identity. But once auto-enrolment has split
one person into two identities, a later crop of that person matches *both*
fragments almost equally (0.94 vs 0.93). The margin test saw a near-tie,
rejected the match, and created a **third** fragment — which made the next
near-tie more likely. Runaway fragmentation. In the replay this rejected 94
of 186 crops, over half the data.

The fix distinguishes the two things a near-tie can mean, via
`VectorGallery(duplicate_identities_expected=...)`:

- **Re-ID gallery** (identities auto-created, duplicates expected): a
  near-tie where *both* candidates clear the threshold means the gallery
  holds two fragments of the same human. Match the best one — rejecting
  guarantees another duplicate. → `True`
- **Face gallery** (identities enrolled by a human, all genuinely distinct):
  a near-tie is real ambiguity about *who* this is, and assigning the
  nominally-higher one risks putting someone else's name on a person. →
  `False` (unchanged behaviour)

A near-tie where the runner-up is *below* the threshold is still rejected in
both modes — that is the 0.61/0.60/0.59 case the margin exists for.

Replaying the same crops with both fixes: **37 identities, 149/186 crops
matched**, and the largest identity correctly absorbed 27 of the old
fragment IDs (verified by eye — one woman at the reception desk who had been
counted as 27 separate "unique" people).

| | identities from 186 real crops |
|---|---|
| ImageNet weights, original margin rule | 80 |
| MSMT17 weights, original margin rule | 131 |
| **MSMT17 weights + margin fix** | **37** |

Re-run it yourself against any folder of crops:

```bash
python -m scripts.replay_reid_identities --snapshots data/reid_snapshots
```

It prints why each crop failed to match and writes a montage per identity,
so the grouping can be checked by eye rather than taken on trust.

## Why the threshold is 0.75 and not lower

Lowering it merges more views of the same person, so the identity count
drops — which looks like an improvement until you check what got merged
(`--sweep` re-runs the matching at several thresholds off one embedding
pass):

| threshold | identities from the same 186 crops |
|---|---|
| 0.50 | 4 |
| 0.60 | 10 |
| 0.62 | 14 |
| 0.70 | 27 |
| **0.75** | **37** |

At 0.62 the largest identity had swallowed 96 crops spanning 44 of the old
IDs — and the montage shows it contains a man in a white shirt, a man in a
dark jacket and several dark silhouettes by the doorway. Different people,
silently merged. At 0.75 the large groupings are visually correct (one
identity of 53 crops is genuinely one woman at the reception desk).

So 0.75 is kept deliberately, accepting that it still **over**-counts
somewhat. The two errors are not symmetrical:

- **Over-count (fragmentation)**: one person becomes two identities. Visible
  in the people list, correctable, and obvious when it happens.
- **Under-count (false merge)**: two people permanently collapse into one
  identity. Invisible, unrecoverable, and quietly wrong forever.

Given the choice, this system errs toward the visible failure. If your site
has people in distinct clothing (no uniforms) you can try a lower threshold —
but run the sweep and look at the montages first.

### Residual over-counting, honestly

Even at 0.75 this deployment still splits some people, because the reception
camera sees several men in near-identical white shirts and red lanyards
(effectively a uniform — the hardest case for appearance-based Re-ID), plus
front/back view changes and people backlit to silhouettes against the glass
door. Expect the daily number to be somewhat higher than the true count of
humans, not lower.

## After changing the model, reset the identities

Embeddings from different models live in different vector spaces and are not
comparable, so identities created by the old model can never be matched again:

```bash
python -m scripts.reset_reid_identities        # dry run
python -m scripts.reset_reid_identities --yes  # actually clear
```

Back up `backend/data/app.db` first if you might want the old identities.

## Configuration

All in `app/config.py`, all overridable via environment/`.env`:

| Setting | Default | Notes |
|---|---|---|
| `REID_MODEL_NAME` | `osnet_x0_25` | architecture |
| `REID_MODEL_PATH` | auto-detected | MSMT17 checkpoint; empty = ImageNet fallback |
| `REID_SIMILARITY_THRESHOLD` | `0.75` | midpoint of the verified 0.50/0.94 gap |
| `REID_MIN_MARGIN` | `0.05` | required lead over the second-best *person* |
| `REID_CONFIRMED_MIN_VOTES` | `3` | frames that must agree before an identity sticks |
| `REID_AUTO_ENROLL_MIN_OBSERVATIONS` | `3` | quality-passing crops before creating a new identity |
| `REID_FOOTFALL_MODE` | `daily` | `daily` (resets at midnight) or `lifetime` |
| `REID_SNAPSHOT_RETENTION_DAYS` | `30` | privacy retention |

Per-camera overrides (enable/disable, ROI, threshold) live in
`reid_camera_config` and are editable via `PUT /api/reid/cameras/{id}/config`.

## Verifying it works

```bash
python -m pytest tests/test_reid_unique_footfall.py -v
```

These cover the primary success criterion directly — person A ten times is
`1`, B makes it `2`, A returning keeps it `2` — plus tracker-ID changes,
ambiguous-match rejection, and single-frame/flapping rejection.

## Real-world limitations — read this before trusting a number

Person Re-ID is **probabilistic**. It matches people by body appearance, so
it degrades or fails when:

- two people wear similar clothing (uniforms are the worst case)
- someone changes clothes between visits — they will become a *new* identity
- the body is heavily occluded, or only partly in frame
- resolution is low, lighting changes sharply, or the camera angle differs
- people walk closely together and the detector merges/swaps boxes
- a long time passes between visits

The system is therefore deliberately built to answer "I don't know": below
the similarity threshold, or when the top two candidates are within
`REID_MIN_MARGIN`, a track stays `UNKNOWN` rather than being forced into an
identity. Expect the count to be a good estimate, not an exact turnstile
reading — and expect it to drift over long periods as people change clothes.

Face recognition would be more precise for identity, but is not used here:
this deployment's camera is wide-angle, its faces are ~30px and usually in
profile, so face-based counting missed essentially everyone. That is why
`FOOTFALL_CAMERAS` (the face-based engine) is empty and Re-ID is primary.
