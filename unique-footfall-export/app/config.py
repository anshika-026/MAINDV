import json
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CAMERA_HOST = os.getenv("CAMERA_HOST", "")
CAMERA_RTSP_PORT = int(os.getenv("CAMERA_RTSP_PORT", "554"))
CAMERA_USER = os.getenv("CAMERA_USER", "")
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "")
CAMERA_STREAM_PATH = os.getenv("CAMERA_STREAM_PATH", "/h264/ch1/sub/av_stream")
CAMERA_ADMIN_PORT = int(os.getenv("CAMERA_ADMIN_PORT", "443"))

# Camera 2 ("Main gate camera") and camera 3 ("Technical section") are two
# RTSP channels off the same NVR unit at CAMERA2_HOST — same login as camera
# 1 (CAMERA_USER/CAMERA_PASSWORD above), just different ports per channel.
CAMERA2_HOST = os.getenv("CAMERA2_HOST", "")
CAMERA2_RTSP_PORT = int(os.getenv("CAMERA2_RTSP_PORT", "554"))
CAMERA2_ADMIN_PORT = int(os.getenv("CAMERA2_ADMIN_PORT", "443"))

CAMERA3_HOST = os.getenv("CAMERA3_HOST", "")
CAMERA3_RTSP_PORT = int(os.getenv("CAMERA3_RTSP_PORT", "554"))
CAMERA3_ADMIN_PORT = int(os.getenv("CAMERA3_ADMIN_PORT", "443"))

SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8811"))

# Many RTSP cameras/NVRs never reply over UDP from behind NAT/firewalls,
# which makes OpenCV's ffmpeg backend fail to open the stream with no
# useful error - forcing TCP fixes that for the large majority of devices.
RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp")
# fflags;nobuffer + flags;low_delay + max_delay;0: ffmpeg's RTSP demuxer
# otherwise keeps its own internal jitter/probe buffer on top of whatever
# OpenCV does (CAP_PROP_BUFFERSIZE only controls OpenCV's own queue, not
# ffmpeg's) - that's real, observed latency a purpose-built live-viewing
# NVR client doesn't have on the identical camera/network, since it isn't
# using ffmpeg's general-purpose (buffer-for-seekability) defaults.
#
# max_delay;0 means zero tolerance for any out-of-order/late packet — fine
# on a short, low-jitter path, but diagnosed live on a longer network path
# (AWS server, further from the camera than a local machine) causing real
# HEVC decode corruption: "Could not find ref with POC N" / "Error
# constructing the frame RPS" repeating in the raw ffmpeg log, because a
# reordered reference frame gets dropped instead of waited for. Overridable
# per-deployment via RTSP_FFMPEG_OPTIONS so a longer/jitterier path can add a
# small reorder buffer without changing the aggressive default everywhere
# (e.g. this machine's own direct connection, where this hasn't been an
# issue) — set to a plain string like
# "rtsp_transport;tcp|max_delay;500000" to allow ~0.5s of reorder tolerance.
_DEFAULT_FFMPEG_OPTIONS = f"rtsp_transport;{RTSP_TRANSPORT}|fflags;nobuffer|flags;low_delay|max_delay;0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = os.getenv("RTSP_FFMPEG_OPTIONS", _DEFAULT_FFMPEG_OPTIONS)

# Cameras confirmed to support on-demand playback from their own onboard
# recording (ONVIF Profile G / Replay - see onvif_client.py), mapped to the
# recording's ONVIF channel number. Only camera 1 is in here: validated
# live this session (real HEVC+audio pulled via ffmpeg for an exact
# requested time window). Cameras 2/3 live on a different physical device
# whose ONVIF replay support was NOT confirmed (its ONVIF port didn't
# respond the same way camera 1's did) - they keep local self-recording
# (pipeline.py) until/unless that's verified too. A camera in this dict
# skips local recording entirely; clips are fetched from the camera only
# when actually played, never stored permanently on this machine.
CAMERA_ONVIF_REPLAY_CHANNEL = {
    1: 1,
}

# Unique footfall (people counting, see footfall_counter.py) is opt-in per
# camera, not automatic for every camera in the system: comma-separated list
# of entry/exit camera identifiers to run it on. Each entry may be a numeric
# camera ID, a cam_code, or a (case/spacing-insensitive) substring of the
# camera's name — e.g. "main_gate" matches a camera named "Main gate camera".
# Empty (the default) means footfall counting is disabled everywhere until a
# gate camera is explicitly named here.
FOOTFALL_CAMERAS = os.getenv("FOOTFALL_CAMERAS", "")

# How long a face embedding stays valid for re-identification before a
# re-appearance at the same camera counts as a brand-new visit.
FOOTFALL_REID_WINDOW_MINUTES = float(os.getenv("FOOTFALL_REID_WINDOW_MINUTES", "10"))

# Cosine similarity floor for matching two embeddings as the same person —
# only consulted for a face with no recognized name (footfall_counter.py
# prefers matching by name outright when one's available, since it's far
# more reliable). Measured live on the Entry/Exit camera: this camera's own
# same-person embedding pairs ranged 0.20-0.72 (median 0.36) while
# different-person pairs ranged up to 0.385 (99th percentile ~0.30) — the
# distributions genuinely overlap, so no value here is exact; 0.30 leans
# toward not merging two different anonymous people rather than toward
# catching every re-appearance of the same one.
FOOTFALL_SIMILARITY_THRESHOLD = float(os.getenv("FOOTFALL_SIMILARITY_THRESHOLD", "0.30"))

# When the end-of-day footfall report job (scheduler.py) runs, as "HH:MM" —
# shortly after midnight by default so it finalizes the day that just ended.
FOOTFALL_REPORT_FINALIZE_TIME = os.getenv("FOOTFALL_REPORT_FINALIZE_TIME", "00:05")

# Rolling local storage window for recognition clips (see clips_db.py's
# delete_expired_clips, run daily by scheduler.py): a clip and its video
# file are deleted once older than this, on every camera. Note this is a
# ceiling on what OUR storage keeps, not a guarantee - camera 1's own
# onboard recording (the source replay_prefetch.py / on-demand fetches pull
# from) independently only holds ~3 days before it overwrites itself
# (confirmed live), so its practical availability window is whichever is
# smaller: this setting, or however far back the camera's own memory still
# reaches. Self-recording cameras (anything NOT in CAMERA_ONVIF_REPLAY_
# CHANNEL) save locally as they're recorded, so for them this setting is
# the real ceiling.
CLIP_RETENTION_DAYS = int(os.getenv("CLIP_RETENTION_DAYS", "7"))

# When the daily clip-retention prune job (scheduler.py) runs, as "HH:MM" -
# shortly after the footfall finalize job so both maintenance jobs land
# together just after midnight.
CLIP_RETENTION_PRUNE_TIME = os.getenv("CLIP_RETENTION_PRUNE_TIME", "00:15")

# License & Camera Access Management (auth.py / license_db.py): JWT signing
# secret for that module's real password-based login — separate from the
# existing trivial /api/auth/login (user_db.record_login), which has no
# password and stays exactly as-is for the main dashboard's "who's using
# this" tracking. Persisted to a file rather than regenerated per process,
# so a backend restart doesn't invalidate every signed-in session (this app
# gets restarted often during development) - only used if JWT_SECRET isn't
# set in the environment, which is the recommended path for production.
_JWT_SECRET_FILE = Path(__file__).resolve().parent.parent / "data" / "jwt_secret.key"


def _load_or_create_jwt_secret() -> str:
    env_secret = os.getenv("JWT_SECRET")
    if env_secret:
        return env_secret
    _JWT_SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _JWT_SECRET_FILE.exists():
        return _JWT_SECRET_FILE.read_text().strip()
    secret = secrets.token_hex(32)
    _JWT_SECRET_FILE.write_text(secret)
    return secret


JWT_SECRET = _load_or_create_jwt_secret()
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_MINUTES", "60"))

# Bootstrap Super Admin (user_db.py's init_db creates this account if no
# super_admin exists yet) - otherwise there'd be no way to sign into the
# License module at all on a fresh database. Change the password after
# first login; this is a development-friendly default, not a production
# secret.
SUPER_ADMIN_EMAIL = os.getenv("SUPER_ADMIN_EMAIL", "admin@deco-vision.local")
SUPER_ADMIN_PASSWORD = os.getenv("SUPER_ADMIN_PASSWORD", "ChangeMe123!")

# Rate limiting (slowapi, in-memory — no Redis dependency at this scale)
# for the License module's public-ish endpoints (login, activation), which
# are the ones worth throttling against brute-force/abuse.
AUTH_RATE_LIMIT = os.getenv("AUTH_RATE_LIMIT", "10/minute")

# Desk-time analytics (see desk_tracker.py): how long an employee can go
# unconfirmed at their desk zone — by face OR by the pose-tracking bridge —
# before that stretch of presence ends and they're marked Away. Long enough
# that one missed detection cycle (face recognition runs ~1x/sec via
# detection_fps) doesn't fragment one sitting into several; short enough
# that a real "got up and left" registers as Away within a reasonable time.
DESK_SESSION_GRACE_SECONDS = int(os.getenv("DESK_SESSION_GRACE_SECONDS", "20"))

# --- Recognition pipeline tuning (all environment-overridable) ----------
# These were previously hardcoded constants scattered across detection_worker.py
# / pipeline.py / recognizer.py, each tuned against a specific real camera this
# session (see the git history / inline comments in those files for the exact
# measurements behind each default below). Centralizing them here as env vars
# means retuning for a different camera, a different physical install, or
# different deployment hardware never requires editing the recognition code
# itself. The defaults below reproduce exactly what was already running —
# setting no env vars changes no behavior.

# Cosine-similarity floor for a face embedding to count as a recognized match
# (recognizer.py). Below this, a face is reported as "Unknown" regardless of
# whose embedding it's closest to.
#
# Confirmed live there is NO clean separation between "correct match" and
# "wrong match" scores with the current single-reference-photo-per-person
# enrollment (38 people): 0.55 was tried and left almost nothing recognized
# at all (observed max score was ~0.541 across a full session of live
# traffic); 0.30 let clearly wrong matches through. 0.40 is a pragmatic
# middle ground, not a value backed by a clean threshold in the data — real
# improvement here needs multiple/better-quality reference photos per
# person, not a different single number.
RECOGNITION_SIMILARITY_THRESHOLD = float(os.getenv("RECOGNITION_SIMILARITY_THRESHOLD", "0.40"))

# Face-detector confidence floor used when no per-camera override applies
# (detection_worker.py's CAMERA_DET_THRESH still takes priority for cameras
# 1/2 — see CAMERA_DET_THRESH_JSON below to override those too).
RECOGNITION_DET_THRESH_DEFAULT = float(os.getenv("RECOGNITION_DET_THRESH_DEFAULT", "0.65"))

# Optional JSON object mapping camera_id -> detection threshold, e.g.
# '{"1": 0.5, "2": 0.45}', to override detection_worker.py's measured
# per-camera defaults without touching code. Unset (the default) keeps those
# measured values exactly as they are.
_camera_det_thresh_json = os.getenv("CAMERA_DET_THRESH_JSON")
CAMERA_DET_THRESH_OVERRIDES = (
    {int(k): float(v) for k, v in json.loads(_camera_det_thresh_json).items()} if _camera_det_thresh_json else None
)

# Optional JSON object mapping camera_id -> detection resolution (longer side,
# px), e.g. '{"2": 1280}', overriding detection_worker.py's
# CAMERA_DETECTION_MAX_DIM. Unset keeps the measured per-camera defaults.
_camera_max_dim_json = os.getenv("CAMERA_DETECTION_MAX_DIM_JSON")
CAMERA_DETECTION_MAX_DIM_OVERRIDES = (
    {int(k): int(v) for k, v in json.loads(_camera_max_dim_json).items()} if _camera_max_dim_json else None
)

# How many of a frame's largest not-yet-matched faces get a second, full-
# resolution recognition pass (detection_worker.py) — the expensive step that
# fixes small/distant faces scoring far lower than their true similarity.
# NOT lowered to save CPU: confirmed live on a real 6-person frame that
# rechecking only the closest 3 faces missed 2 of 3 real, strong matches —
# see this constant's other use site in detection_worker.py for that
# measurement. recognition_track_cache.py is what actually cuts recheck
# volume now (an already-confirmed or already-recently-tried face doesn't
# consume a slot at all), not a lower ceiling here.
RECOGNITION_MAX_FULL_RES_RECHECKS = int(os.getenv("RECOGNITION_MAX_FULL_RES_RECHECKS", "8"))
RECOGNITION_RECHECK_DET_THRESH = float(os.getenv("RECOGNITION_RECHECK_DET_THRESH", "0.3"))
RECOGNITION_RECHECK_CROP_PADDING = float(os.getenv("RECOGNITION_RECHECK_CROP_PADDING", "0.8"))

# recognition_track_cache.py: how long a confidently-recognized face is
# trusted before spending one recheck to reconfirm it, rather than paying
# for a fresh recheck every single cycle it stays in frame. Note: on a
# camera whose real per-cycle time already exceeds this (measured 24-100+s
# under load on the wide-angle "Main gate" camera — see
# RECOGNITION_TRACK_TIMEOUT_SECONDS below), this is effectively a no-op —
# expected, not a bug. The actual latency win on that camera comes from the
# recheck BUDGET no longer being spent on already-known faces at all
# (see RECOGNITION_MAX_FULL_RES_RECHECKS above), not from this interval,
# which mainly helps lighter-load cameras with faster cycles.
RECOGNITION_CACHE_REVERIFY_SECONDS = float(os.getenv("RECOGNITION_CACHE_REVERIFY_SECONDS", "10"))

# Same module, the opposite case: a tracked face that has never been
# confidently identified only gets another expensive recheck attempt this
# often, not on every single cycle — avoids repeatedly paying for a
# recheck on a real Unknown/unregistered visitor who was never going to
# match. No stale-identity risk here (unlike the reverify window above) —
# "still Unknown" is always safe to keep reporting while waiting.
RECOGNITION_RETRY_INTERVAL_SECONDS = float(os.getenv("RECOGNITION_RETRY_INTERVAL_SECONDS", "3"))

# How long a recognition_track_cache.py track survives with no matching
# detection before it's dropped (person left frame / long occlusion).
# Deliberately its OWN, much shorter constant than
# RECOGNITION_TRACK_TIMEOUT_SECONDS (180s, below) even though both are
# IoU-based face tracking: that 180s value was tuned for
# recognition_stabilizer.py, a purely cosmetic main-process layer where a
# stale cached identity only mislabels the overlay for a while. This
# cache's decisions also reach footfall/desk/zone-violation logic in
# pipeline.py (whatever name a track reports IS what those consume), so a
# wrong identity surviving 180s there risks a real desk-session or
# zone-alert misattribution, not just a mislabeled frame. Start
# conservative; widen only if live use shows tracks being needlessly
# recreated more than this protects against.
RECOGNITION_CACHE_TRACK_TIMEOUT_SECONDS = float(os.getenv("RECOGNITION_CACHE_TRACK_TIMEOUT_SECONDS", "20"))

# How many consecutive detection cycles the SAME name must appear on a camera
# before it's logged as a detection_event (attendance/analytics) — filters out
# a one-off spurious match (embedding noise on a single frame) without
# touching what the live overlay shows immediately. 1 (the default) reproduces
# the previous behavior exactly: any single hit logs, same as before this
# setting existed. Raise it to require the match to repeat before it's
# recorded; keep DETECTION_LOG_COOLDOWN_SECONDS below as the separate
# duplicate-suppression window for an already-confirmed, continuously-present
# person.
RECOGNITION_MIN_CONSECUTIVE_HITS = int(os.getenv("RECOGNITION_MIN_CONSECUTIVE_HITS", "1"))

# Once a name has been logged, don't log it again for the same camera more
# often than this — avoids flooding detection_events while someone stands
# continuously in frame.
DETECTION_LOG_COOLDOWN_SECONDS = int(os.getenv("DETECTION_LOG_COOLDOWN_SECONDS", "30"))

# How stale a camera's last-computed recognition result is allowed to look
# before the LIVE OVERLAY hides it, independent of how often the frontend
# happens to re-request it. Diagnosed live this session: /ws/detections
# polls get_latest_detections() every ~167ms (DETECTIONS_FPS=6 in main.py)
# regardless of whether the underlying result actually changed, so a
# frontend "clear if no message arrives for 1s" timer can never fire — a
# message always arrives. The camera's own compute cycle is what actually
# refreshes the data (measured: ~1-2s on a light scene, 8-11s on the busy
# Main gate camera's 6-8 person frame), so THAT is the real staleness clock.
# Each detections payload now carries computed_at (the timestamp
# set_detections() last ran) and the frontend clears locally once
# now - computed_at exceeds this, rather than relying on message silence.
# This does not — and physically cannot — make a busy multi-person scene's
# true removal latency faster than its own recognition cycle time; it only
# fixes the frontend showing data that's already known to be older than
# this, whatever the real cycle time turns out to be this cycle.
IDENTITY_LOST_TIMEOUT_SECONDS = float(os.getenv("IDENTITY_LOST_TIMEOUT_SECONDS", "1.0"))

# A flat IDENTITY_LOST_TIMEOUT_SECONDS alone caused a real, reported bug: on
# a camera whose actual recognition cycle takes longer than this (measured
# 8-12s on a busy multi-person scene), the still-present person's name
# flashed on for under a second and then disappeared every single cycle,
# even though they never left — the fixed timeout was simply shorter than
# the time between genuinely fresh results. CameraPipeline tracks each
# camera's own last real cycle gap and widens the effective timeout to
# max(IDENTITY_LOST_TIMEOUT_SECONDS, last_cycle_gap * this factor) — see
# get_effective_identity_lost_timeout(). >1 on purpose: cycle time varies
# cycle to cycle, so sizing the timeout to exactly the last gap would still
# false-clear whenever the next cycle happens to run a bit slower than the
# one before it.
IDENTITY_LOST_TIMEOUT_SAFETY_FACTOR = float(os.getenv("IDENTITY_LOST_TIMEOUT_SAFETY_FACTOR", "1.5"))

# How long a synchronous embedding request (enrollment / camera Allow List
# sync) waits for the detection worker to respond before giving up. Must
# comfortably exceed the worker's worst-case single-frame processing time on
# whatever hardware this is running on, or a slow (but eventually successful)
# detection gets wrongly reported as "no face detected".
DETECTION_EMBED_TIMEOUT_SECONDS = float(os.getenv("DETECTION_EMBED_TIMEOUT_SECONDS", "25"))

# How often the Honeywell recognition poller (honeywell_recognition_poller.py)
# queries each camera's own onboard SnapedFaces recognition log for new
# matches. This — not anything in the local detection pipeline — is what
# now decides how fast a recognized person shows up in Attendance/People
# Analytics, since identity comes from the camera's own engine rather than
# local ArcFace matching. A short interval (originally 2s) was tried and
# confirmed too aggressive: this device starts refusing/timing out new
# connections after just a handful of requests in quick succession, live-
# verified this session (a People List fetch — 3 requests — succeeded, but
# a SnapedFaces fetch moments later was refused, then timed out on retry).
# 20s matches the value already in production use as a stability mitigation.
HONEYWELL_POLL_INTERVAL_SECONDS = float(os.getenv("HONEYWELL_POLL_INTERVAL_SECONDS", "20"))

# How fresh a Honeywell recognition (face_db.detection_events,
# recognition_source='honeywell') must be for pipeline.py's
# CameraPipeline.set_detections to prefer it over the local worker's own
# match for the live overlay — only when exactly one face is in frame,
# since Honeywell's own events carry no bounding box and there's no way to
# tell which face a Honeywell identity belongs to once more than one
# person is present. Set comfortably above HONEYWELL_POLL_INTERVAL_SECONDS
# (20s) and the poller's own write-cooldown (DETECTION_LOG_COOLDOWN_SECONDS,
# 30s) — a shorter window would frequently find "nothing fresh enough" even
# for someone Honeywell recognized moments ago, purely from the poller's
# own cadence, silently defeating this most of the time. 0 disables the
# feature entirely (always falls back to local recognition).
HONEYWELL_LIVE_PREFERENCE_WINDOW_SECONDS = float(os.getenv("HONEYWELL_LIVE_PREFERENCE_WINDOW_SECONDS", "45"))

# The one Honeywell device treated as the authoritative source for the People
# List (see main.py's people/sync-from-camera). Enrolling people directly on
# this camera and re-syncing is meant to fully reconcile enrolled_faces
# (update existing, add new, remove anything no longer on this device's Allow
# List) — never a blind merge across every configured camera/device, which
# previously pulled in an unrelated device's stale Allow List alongside this
# one's real data.
PRIMARY_PEOPLE_SOURCE_HOST = os.getenv("PRIMARY_PEOPLE_SOURCE_HOST", "103.204.0.122")

# Honeywell recognition-poller reconnect backoff, mirroring
# CAMERA_RECONNECT_*_DELAY_SECONDS below but tracked per physical device host
# rather than per RTSP stream — a host that's failing backs off up to the max
# instead of being retried every HONEYWELL_POLL_INTERVAL_SECONDS regardless,
# and resets to base the moment a poll against it succeeds again.
HONEYWELL_RECONNECT_BASE_DELAY_SECONDS = float(os.getenv("HONEYWELL_RECONNECT_BASE_DELAY_SECONDS", "5"))
HONEYWELL_RECONNECT_MAX_DELAY_SECONDS = float(os.getenv("HONEYWELL_RECONNECT_MAX_DELAY_SECONDS", "120"))

# Diagnosed live (scripts/diagnose_honeywell.py, both from this machine and
# from AWS): this device reliably serves roughly one request per connection
# before resetting it, so a second failure right after a fresh relogin is
# expected device behavior, not something more retries would fix. Bounded at
# 2 (1 initial attempt + 1 reconnect-and-retry) rather than higher — a higher
# value would turn a device that's already struggling under one request into
# a request storm without meaningfully improving success odds; a
# persistently-down device should fail fast here and let the poller's own
# HONEYWELL_RECONNECT_*_DELAY_SECONDS backoff decide when to try again.
HONEYWELL_MAX_REQUEST_ATTEMPTS = int(os.getenv("HONEYWELL_MAX_REQUEST_ATTEMPTS", "2"))

# How often the recognition poller's in-memory Honeywell-person-ID -> name
# cache (built from our already-synced enrolled_faces, not a fresh camera API
# call) is refreshed on a timer, independent of the immediate on-demand
# refresh that already happens the moment an unresolved person ID is seen.
HONEYWELL_PEOPLE_CACHE_REFRESH_INTERVAL_SECONDS = float(
    os.getenv("HONEYWELL_PEOPLE_CACHE_REFRESH_INTERVAL_SECONDS", "300")
)

# Optional, off by default (None = disabled): Honeywell's recognition-score
# semantics have never been confirmed against real documentation, so this
# never discards a low-scoring event — Honeywell remains the recognition
# authority. If set, an event scoring below this is still written normally,
# just tagged recognition_source='low_confidence' instead of 'honeywell' for
# a reviewer to notice, rather than silently trusted or silently dropped.
_low_conf_raw = os.getenv("HONEYWELL_LOW_CONFIDENCE_THRESHOLD")
HONEYWELL_LOW_CONFIDENCE_THRESHOLD = float(_low_conf_raw) if _low_conf_raw else None

# RTSP reconnect backoff (pipeline.py): starts at the base delay, doubles on
# each consecutive failure up to the max, resets to base on a successful
# reconnect. Prevents a real outage from hammering the camera's own login
# endpoint (some devices self-lockout after repeated rapid auth failures).
CAMERA_RECONNECT_BASE_DELAY_SECONDS = float(os.getenv("CAMERA_RECONNECT_BASE_DELAY_SECONDS", "3"))
CAMERA_RECONNECT_MAX_DELAY_SECONDS = float(os.getenv("CAMERA_RECONNECT_MAX_DELAY_SECONDS", "60"))

# Total CPU core budget for the whole detection subsystem (all per-camera
# worker processes combined), split proportionally by relative cost — see
# PipelineManager._worker_core_allocation. Explicitly set
# DETECTION_WORKER_MAX_CPU_CORES to tune this for the actual deployment
# machine — see the deployment note on this in recognition_config's module
# docstring below. Left unset, the fallback below reserves 2 cores for
# everything that ISN'T detection (the API/WebSocket event loop, nginx,
# ffmpeg clip transcodes, the OS) rather than a fixed number that can equal
# — or exceed — a small box's entire core count. That exact bug shipped to
# a 4-vCPU EC2 instance with this defaulted to 4: the detection workers
# were allowed the whole machine, starving the API process and making
# every dashboard fetch slow regardless of how fast the DB or network was.
DETECTION_WORKER_MAX_CPU_CORES = int(
    os.getenv("DETECTION_WORKER_MAX_CPU_CORES", str(max(1, (os.cpu_count() or 4) - 2)))
)

# Default face-recognition sampling rate (frames/sec sent to the detection
# worker) before any /api/settings override — that DB-backed "detection_fps"
# setting (see pipeline.py's _sender_loop) already lets this be changed live
# from the UI without a restart; this env var only changes the fallback used
# before that setting has ever been saved.
DEFAULT_DETECTION_FPS = float(os.getenv("DEFAULT_DETECTION_FPS", "1"))

# --- Temporal identity stabilization (recognition_stabilizer.py) --------
# Every detection cycle matches faces independently -- nothing carries
# identity between cycles by default, so a single noisy frame can flip a
# confidently-recognized person to "Unknown" or a different name for one
# cycle, then flip back (the classic "Rahul, Unknown, Amit, Rahul" flicker).
# This tracks faces frame-to-frame by bounding-box overlap and only reports
# a name once it has a clear majority across the recent window, holding
# onto a stable identity through brief single-frame dips.
#
# Disabled by default (window=1, min_votes=1 is a no-op: every single vote
# already "wins" its window of size 1) so this ships without changing
# existing behavior; set RECOGNITION_STABILIZATION_ENABLED=true to turn it on.
RECOGNITION_STABILIZATION_ENABLED = os.getenv("RECOGNITION_STABILIZATION_ENABLED", "false").lower() == "true"
RECOGNITION_STABILIZATION_WINDOW = int(os.getenv("RECOGNITION_STABILIZATION_WINDOW", "5"))
RECOGNITION_STABILIZATION_MIN_VOTES = int(os.getenv("RECOGNITION_STABILIZATION_MIN_VOTES", "3"))
# Two detections across consecutive cycles are considered the same physical
# face if their boxes overlap at least this much (Intersection-over-Union).
RECOGNITION_TRACK_IOU_THRESHOLD = float(os.getenv("RECOGNITION_TRACK_IOU_THRESHOLD", "0.3"))
# How long a track survives with no matching detection before it's dropped
# (person left frame, or was fully occluded/undetected for a while).
# Confirmed live and the actual reason this feature failed its first
# production trial: this was left at 5s, assuming a detect cycle roughly
# matches detection_fps's ~1s interval -- but a real per-camera cycle
# (queue wait + recognize time) measured 24-100+ seconds on the wide-angle
# "Main gate" camera under normal load. Every track expired before it could
# accumulate votes, permanently forcing that camera to Unknown. Raised well
# past that camera's worst measured cycle time.
RECOGNITION_TRACK_TIMEOUT_SECONDS = float(os.getenv("RECOGNITION_TRACK_TIMEOUT_SECONDS", "180"))

# fire_smoke_detector.py is a classical HSV-color/flicker heuristic, not a
# trained model — confirmed live to false-positive on skin tone/warm-toned
# clothing under bright lighting when a face wasn't detected that exact
# frame (the face-exclusion zone it relies on needs a face box to exist).
# Turned off by default per that false positive; set back to true once a
# real trained fire/smoke model replaces this heuristic, or the thresholds
# are retuned against real fire footage.
FIRE_SMOKE_DETECTION_ENABLED = os.getenv("FIRE_SMOKE_DETECTION_ENABLED", "false").lower() == "true"

# How often the sender loop checks whether each camera's detection worker
# process is still alive, and respawns it if not. Detection worker crashes
# (confirmed live: a native-level crash with no Python exception, no OOM,
# and no log line at all — multiprocessing.Process has no built-in health
# check or auto-restart) previously left a camera silently unrecognized
# indefinitely, invisible from the API (the last cached result just never
# updated again) until someone noticed and manually restarted the whole
# backend. This closes that gap without needing a full service restart.
WORKER_HEALTH_CHECK_INTERVAL_SECONDS = float(os.getenv("WORKER_HEALTH_CHECK_INTERVAL_SECONDS", "15"))

# --- People Identification module ---------------------------------------
# New, separate module (peopleid_*.py) — face-embedding gallery + vector
# search + tracking + temporal fusion. Deliberately its OWN threshold/config
# namespace, not reusing RECOGNITION_SIMILARITY_THRESHOLD etc. above: those
# were tuned against the existing single-photo `enrolled_faces` gallery and
# the Honeywell-preference live overlay; this module's multi-embedding,
# quality-gated gallery has a genuinely different score distribution and
# must be validated independently (see backend/scripts/benchmark_peopleid.py)
# rather than copying a number tuned for a different gallery.

# Cosine-similarity floor for a candidate match — below this, or within
# PEOPLEID_MIN_MARGIN of the runner-up person, the result is UNKNOWN rather
# than the nominally-closest person (spec: three close scores like
# 0.61/0.60/0.59 must never be force-resolved to the highest one). Both
# NOT validated against real deployment data yet — placeholders pending
# benchmark_peopleid.py's threshold sweep against real footage; see this
# module's benchmark/known-limitations notes.
PEOPLEID_SIMILARITY_THRESHOLD = float(os.getenv("PEOPLEID_SIMILARITY_THRESHOLD", "0.45"))
PEOPLEID_MIN_MARGIN = float(os.getenv("PEOPLEID_MIN_MARGIN", "0.05"))

# Face quality gate (peopleid_quality.py) — a face below this composite
# score is never used for an identity decision; the track just waits for a
# better observation instead (spec section 8).
PEOPLEID_QUALITY_MIN_SCORE = float(os.getenv("PEOPLEID_QUALITY_MIN_SCORE", "0.55"))
PEOPLEID_MIN_FACE_SIZE_PX = int(os.getenv("PEOPLEID_MIN_FACE_SIZE_PX", "40"))
PEOPLEID_MIN_DET_SCORE = float(os.getenv("PEOPLEID_MIN_DET_SCORE", "0.5"))
PEOPLEID_BLUR_VARIANCE_FLOOR = float(os.getenv("PEOPLEID_BLUR_VARIANCE_FLOOR", "150.0"))
PEOPLEID_BLUR_VARIANCE_HARD_MIN = float(os.getenv("PEOPLEID_BLUR_VARIANCE_HARD_MIN", "40.0"))
PEOPLEID_BRIGHTNESS_HARD_MIN = float(os.getenv("PEOPLEID_BRIGHTNESS_HARD_MIN", "25.0"))
PEOPLEID_BRIGHTNESS_HARD_MAX = float(os.getenv("PEOPLEID_BRIGHTNESS_HARD_MAX", "235.0"))
PEOPLEID_MAX_POSE_DEVIATION = float(os.getenv("PEOPLEID_MAX_POSE_DEVIATION", "0.9"))
PEOPLEID_POSE_HARD_MAX = float(os.getenv("PEOPLEID_POSE_HARD_MAX", "1.4"))
PEOPLEID_QUALITY_WEIGHT_SIZE = float(os.getenv("PEOPLEID_QUALITY_WEIGHT_SIZE", "0.15"))
PEOPLEID_QUALITY_WEIGHT_BLUR = float(os.getenv("PEOPLEID_QUALITY_WEIGHT_BLUR", "0.30"))
PEOPLEID_QUALITY_WEIGHT_BRIGHTNESS = float(os.getenv("PEOPLEID_QUALITY_WEIGHT_BRIGHTNESS", "0.15"))
PEOPLEID_QUALITY_WEIGHT_POSE = float(os.getenv("PEOPLEID_QUALITY_WEIGHT_POSE", "0.25"))
PEOPLEID_QUALITY_WEIGHT_DET = float(os.getenv("PEOPLEID_QUALITY_WEIGHT_DET", "0.15"))

# Person-body MOT (peopleid_tracker.py: YOLOv8n + ByteTrack). NOT run every
# detect cycle — see this module's own cadence gate, deliberately separate
# from POSE_INTERVAL_SECONDS above even though both are Ultralytics-model
# cadence gates protecting the same shared per-camera CPU budget; this one
# has its own knob because it has different accuracy-vs-cost tradeoffs
# (ByteTrack degrades with sparser sampling — see the module's plan doc).
# The 3s default below is a starting point, NOT a validated number — must
# be benchmarked on the real deployment box (backend/scripts/
# benchmark_peopleid.py's latency section) before being trusted; this
# codebase already measured a same-cost-class model (YOLOv8n-pose) needing
# to be throttled from 5s to 20s on CPU to avoid starving face recognition
# in the same process (see POSE_INTERVAL_SECONDS above).
PEOPLEID_MOT_INTERVAL_SECONDS = float(os.getenv("PEOPLEID_MOT_INTERVAL_SECONDS", "3.0"))
PEOPLEID_MOT_MODEL_PATH = os.getenv("PEOPLEID_MOT_MODEL_PATH", "yolov8n.pt")
# Confirmed live this session against a REAL camera (1920x1080, a person
# standing clearly in frame): 384 was too aggressive a downscale — YOLOv8n's
# own confidence for that exact same person dropped from 0.83 (at imgsz=960)
# to fragmented/sub-threshold boxes (0.10-0.20) at 384, silently missing a
# real, unoccluded person entirely (both here and in the already-shipped
# peopleid_* module, which shares this same constant). 640 recovered a
# solid 0.70-0.71 confidence on the same real frames while staying fast
# (~174ms/frame measured on this machine's CPU, vs 384's own ~120-140ms) —
# a good balance until this is validated against more real cameras.
PEOPLEID_MOT_IMGSZ = int(os.getenv("PEOPLEID_MOT_IMGSZ", "640"))
PEOPLEID_MOT_CONFIDENCE = float(os.getenv("PEOPLEID_MOT_CONFIDENCE", "0.4"))

# Identity fusion (peopleid_fusion.py) — see that module's docstring for the
# full state-machine rationale. Window/vote defaults grounded in this
# codebase's own existing (disabled-by-default) cosmetic stabilizer
# (RECOGNITION_STABILIZATION_WINDOW=5/MIN_VOTES=3 above), tightened since
# PeopleID additionally quality-gates every observation before it counts.
PEOPLEID_FUSION_WINDOW = int(os.getenv("PEOPLEID_FUSION_WINDOW", "5"))
PEOPLEID_CANDIDATE_MIN_VOTES = int(os.getenv("PEOPLEID_CANDIDATE_MIN_VOTES", "2"))
PEOPLEID_CONFIRMED_MIN_VOTES = int(os.getenv("PEOPLEID_CONFIRMED_MIN_VOTES", "3"))
# How many CONSECUTIVE quality-passing observations naming a DIFFERENT
# person are required before a CONFIRMED identity is dropped — deliberately
# > 1 so a single contradicting frame can't flip an established identity
# (spec: "Anshika, Anshika, Unknown, Anshika, Anshika" must stay Anshika).
PEOPLEID_CONFIRMED_CONTRADICTION_LIMIT = int(os.getenv("PEOPLEID_CONFIRMED_CONTRADICTION_LIMIT", "2"))
# Track-level expiry — mirrors RECOGNITION_CACHE_TRACK_TIMEOUT_SECONDS
# (20s) rather than the cosmetic stabilizer's 180s, since a CONFIRMED
# People-ID track feeds real events/analytics, not just a display overlay.
PEOPLEID_TRACK_TIMEOUT_SECONDS = float(os.getenv("PEOPLEID_TRACK_TIMEOUT_SECONDS", "20.0"))

# Person Re-ID (peopleid_reid.py) — bridges an already-CONFIRMED track
# through a face-absent stretch ONLY; can never establish an identity on
# its own (see peopleid_fusion.py). Deliberately a short, separate window
# from FOOTFALL_REID_WINDOW_MINUTES (10 min) above — that's for a coarser
# "is this the same visitor returning" use case; this is "did the camera
# just lose their face for a few seconds," a much weaker signal that
# shouldn't be trusted nearly as long.
PEOPLEID_REID_ENABLED = os.getenv("PEOPLEID_REID_ENABLED", "true").lower() == "true"
PEOPLEID_REID_SIMILARITY_THRESHOLD = float(os.getenv("PEOPLEID_REID_SIMILARITY_THRESHOLD", "0.55"))
PEOPLEID_REID_BRIDGE_SECONDS = float(os.getenv("PEOPLEID_REID_BRIDGE_SECONDS", "45.0"))

# Enrollment curation (peopleid_enrollment.py) — spec section 15: 10-20
# diverse, quality-gated reference embeddings per person, chosen
# automatically from however many raw frames/photos were captured.
PEOPLEID_ENROLLMENT_TARGET_EMBEDDINGS = int(os.getenv("PEOPLEID_ENROLLMENT_TARGET_EMBEDDINGS", "15"))
# Two candidate reference embeddings this similar are treated as
# near-duplicates during curation — keeps the final set from being 15
# nearly-identical frontal frames grabbed a fraction of a second apart.
PEOPLEID_DEDUP_SIMILARITY = float(os.getenv("PEOPLEID_DEDUP_SIMILARITY", "0.92"))

# Unknown-person review (spec section 25) — a NOT-yet-assigned-person
# cluster this close to a new unmatched observation is treated as a repeat
# sighting of the same unknown individual rather than a brand-new cluster.
PEOPLEID_UNKNOWN_CLUSTER_SIMILARITY = float(os.getenv("PEOPLEID_UNKNOWN_CLUSTER_SIMILARITY", "0.55"))
# How often a CONFIRMED track logs another peopleid_events "sighting" row
# while it stays continuously confirmed — mirrors
# DETECTION_LOG_COOLDOWN_SECONDS's existing role for the unrelated
# enrolled_faces/detection_events pipeline, same reasoning: avoid flooding
# the events table while someone stands continuously in frame.
PEOPLEID_EVENT_LOG_COOLDOWN_SECONDS = float(os.getenv("PEOPLEID_EVENT_LOG_COOLDOWN_SECONDS", "30.0"))


# --- Body-appearance Re-ID unique footfall engine (reid_*.py) -----------
# A SEPARATE module/namespace from PEOPLEID_* above, not a replacement of
# it: peopleid_* identifies people by FACE embedding (deliberately, for the
# existing named-attendance/enrollment use case); this module identifies
# people by BODY-APPEARANCE embedding instead, so someone facing away from
# the camera (never contributing a usable face crop) is still tracked as a
# distinct, returning visitor — the actual ask behind "unique footfall,"
# where most people crossing a gate are never enrolled/named at all. Both
# modules reuse the same tracker/fusion/gallery CLASSES (see reid_worker.py)
# with separate instances and separate DB tables (reid_db.py), since a face
# embedding and a body embedding are different-dimension, different-space
# vectors that must never be compared against each other.
#
# Model: torchreid's OSNet (osnet_x0_25 by default — 203K params, ~82M
# FLOPs, verified live on this machine's CPU at ~85ms/crop single, ~30ms/
# crop batched — see reid_embedding.py). REID_MODEL_PATH empty (default)
# uses OSNet's auto-downloaded ImageNet-pretrained backbone (verified
# working live this session) — a real pretrained CNN producing genuine
# appearance features, but NOT fine-tuned on a person-Re-ID task/dataset
# (Market1501/MSMT17) specifically. Point REID_MODEL_PATH at a local .pth
# from the deep-person-reid Model Zoo (a task-fine-tuned checkpoint) to
# swap in a materially more discriminative embedding with no code change —
# torchreid.utils.load_pretrained_weights (see reid_embedding.py) loads
# either the same way.
#
# MEASURED, this deployment, against 186 real person crops already collected
# from the reception camera (backend/scripts/eval_reid_weights.py):
#
#                          same-person   different-person   separation
#   ImageNet backbone        med 0.862       med 0.688        +0.174
#   MSMT17 Re-ID-trained     med 0.858       med 0.445        +0.413
#
# The ImageNet fallback could not separate people at ANY threshold: 33% of
# different-person pairs scored above 0.72 while same-person pairs ran as
# low as 0.63. Visually spot-checked both failure directions on real crops
# from this camera — the ImageNet model scored three pairs of plainly
# DIFFERENT people at 0.86-0.89 (would merge them: under-count), while six
# pairs of the plainly SAME person landed close enough to the runner-up to
# fail the REID_MIN_MARGIN decisiveness test and spawn a new identity
# (over-count). That combination is what produced 34 "unique" people in a
# single hour on one reception door. The MSMT17 checkpoint scored all nine
# of those verified pairs correctly (same 0.94-0.95, different 0.40-0.50).
#
# The checkpoint is osnet_x0_25 trained on MSMT17 (combineall) from OSNet's
# own author, mirrored on HuggingFace at kaiyangzhou/osnet — same
# architecture as the ImageNet default, so this is a weights-only swap with
# no code, no speed and no memory change. Download it with:
#   python -m scripts.fetch_reid_model
# Falls back to the ImageNet backbone automatically if the file is absent,
# so a fresh checkout still runs (just far less accurately — fetch it).
REID_MODEL_NAME = os.getenv("REID_MODEL_NAME", "osnet_x0_25")
_REID_DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "osnet_x0_25_msmt17.pth"
REID_MODEL_PATH = os.getenv("REID_MODEL_PATH", str(_REID_DEFAULT_MODEL) if _REID_DEFAULT_MODEL.exists() else "")
REID_DEVICE = os.getenv("REID_DEVICE", "auto")  # "auto" | "cpu" | "cuda"

# Cosine-similarity floor + minimum margin over the runner-up, same
# decisiveness rule as PEOPLEID_SIMILARITY_THRESHOLD/PEOPLEID_MIN_MARGIN
# above (see peopleid_gallery.VectorGallery.best_match), reused by
# reid_worker's gallery as-is. This threshold assumes reid_embedding.py's
# own fixed-calibration mean-centering already ran (see that module's
# docstring) — confirmed LIVE this session that the OSNet ImageNet
# -pretrained fallback's RAW features are barely discriminative at all (20
# different synthetic textures scored 0.90-0.96 similarity to EACH OTHER
# uncentered), which this threshold would force-merge outright without that
# fix. With it applied on synthetic textures: same-image repeat 1.0,
# different-texture pairs 0.09-0.52.
#
# 0.72 was the old ImageNet-backbone value, derived from a single real
# pair. It is not the right number for the MSMT17 checkpoint now in use
# (see REID_MODEL_PATH above), whose similarity distribution is different
# and far better separated. Re-derived from visually-verified real crops
# off this camera:
#
#   verified SAME person      0.94, 0.95, 0.94, 0.94, 0.94, 0.94
#   verified DIFFERENT people 0.48, 0.50, 0.40
#
# 0.75 sits in the middle of that gap — 0.25 above the verified
# different-person ceiling, 0.19 below the verified same-person floor — so
# both failure modes need a large, unlikely excursion to trigger. Leaning
# slightly toward the merge-protection side of the midpoint on purpose: a
# false merge silently under-counts forever (two real people permanently
# collapsed into one identity), whereas a fragmentation over-counts once
# and is visible/correctable in the people list.
REID_SIMILARITY_THRESHOLD = float(os.getenv("REID_SIMILARITY_THRESHOLD", "0.75"))
# Required lead over the SECOND-BEST PERSON (not the second-best embedding —
# peopleid_gallery.search() already reduces to one best score per person),
# so three close scores resolve to Unknown instead of a forced pick. This
# test is why the old backbone fragmented so badly: when a third of all
# different-person pairs score above threshold, the runner-up person is
# always within 0.05 of the best, every match is rejected as indecisive,
# and each appearance spawns a fresh identity. It only does its intended
# job on top of an embedding that actually separates people.
REID_MIN_MARGIN = float(os.getenv("REID_MIN_MARGIN", "0.05"))

# Quality gate (reid_quality.py) — mirrors PEOPLEID_QUALITY_MIN_SCORE's
# role, but for a body crop: no face landmarks exist here, so pose/
# occlusion is proxied by bbox aspect ratio instead of the 5-point yaw/
# pitch estimate peopleid_quality.py uses.
REID_QUALITY_MIN_SCORE = float(os.getenv("REID_QUALITY_MIN_SCORE", "0.55"))
REID_MIN_BODY_SIZE_PX = int(os.getenv("REID_MIN_BODY_SIZE_PX", "80"))
REID_BLUR_VARIANCE_FLOOR = float(os.getenv("REID_BLUR_VARIANCE_FLOOR", "80.0"))
REID_BLUR_VARIANCE_HARD_MIN = float(os.getenv("REID_BLUR_VARIANCE_HARD_MIN", "20.0"))

# Identity fusion (reid_fusion.py reuses peopleid_fusion.IdentityFusion's
# class directly — these mirror PEOPLEID_FUSION_WINDOW etc. as this
# module's OWN copy of the same tunables, so retuning one module never
# silently retunes the other).
REID_FUSION_WINDOW = int(os.getenv("REID_FUSION_WINDOW", "5"))
REID_CANDIDATE_MIN_VOTES = int(os.getenv("REID_CANDIDATE_MIN_VOTES", "2"))
REID_CONFIRMED_MIN_VOTES = int(os.getenv("REID_CONFIRMED_MIN_VOTES", "3"))
REID_CONFIRMED_CONTRADICTION_LIMIT = int(os.getenv("REID_CONFIRMED_CONTRADICTION_LIMIT", "2"))
REID_TRACK_TIMEOUT_SECONDS = float(os.getenv("REID_TRACK_TIMEOUT_SECONDS", "20.0"))

# Auto-enrollment (spec: no human ever has to manually create a person) —
# after this many quality-passing CANDIDATE-state observations with no
# confident gallery match, reid_worker.py auto-creates a new PERSON_XXX
# identity from the track's own collected crops (peopleid_enrollment.curate
# reused for the dedup+diversity selection) instead of just flagging an
# "unknown cluster" for a human to assign, unlike the face-based peopleid_*
# module (spec section 2/7: identity creation must be fully automatic).
REID_AUTO_ENROLL_MIN_OBSERVATIONS = int(os.getenv("REID_AUTO_ENROLL_MIN_OBSERVATIONS", "3"))
REID_ENROLLMENT_TARGET_EMBEDDINGS = int(os.getenv("REID_ENROLLMENT_TARGET_EMBEDDINGS", "12"))

# Per-camera cadence gate, mirrors PEOPLEID_MOT_INTERVAL_SECONDS — the body
# tracker + Re-ID embedding pass doesn't run every frame.
REID_MOT_INTERVAL_SECONDS = float(os.getenv("REID_MOT_INTERVAL_SECONDS", "3.0"))

# Retention (spec section 17) — mirrors CLIP_RETENTION_DAYS's role/pattern
# above, applied to reid_snapshots/reid_embeddings/reid_events instead of
# recognition clips.
REID_SNAPSHOT_RETENTION_DAYS = int(os.getenv("REID_SNAPSHOT_RETENTION_DAYS", "30"))
REID_EMBEDDING_RETENTION_DAYS = int(os.getenv("REID_EMBEDDING_RETENTION_DAYS", "0"))  # 0 = never expire
REID_APPEARANCE_RETENTION_DAYS = int(os.getenv("REID_APPEARANCE_RETENTION_DAYS", "90"))

# How often a CONFIRMED reid track logs another reid_events "sighting" row
# while it stays continuously confirmed — mirrors
# PEOPLEID_EVENT_LOG_COOLDOWN_SECONDS's identical role for the unrelated
# face-based module.
REID_EVENT_LOG_COOLDOWN_SECONDS = float(os.getenv("REID_EVENT_LOG_COOLDOWN_SECONDS", "30.0"))

# Footfall counting mode (spec section 13): "daily" resets the unique count
# at midnight (reid_db.count_unique_today), "lifetime" never resets
# (reid_db.count_unique_lifetime) — both counts are always available via
# the API regardless of this setting; this only selects which one the
# dashboard's headline "Unique Footfall" stat tile shows.
REID_FOOTFALL_MODE = os.getenv("REID_FOOTFALL_MODE", "daily")


# --- Real-Time Emotion & Engagement Detection (emotion_*.py) -------------
# A webcam-driven feature, unrelated in purpose to the RTSP surveillance
# pipeline above but living in the same app per instruction to reuse this
# codebase's existing architecture. Classifies the person in front of the
# BROWSER's own webcam (not a configured camera_db camera) into HAPPY /
# ENGAGED / DISTRACTED / NO_FACE. See emotion_api.py's /ws/emotion — the
# browser pushes frames TO this backend (the reverse of every /ws/live/{id}
# feed, which pushes FROM the backend), since the browser is the one that
# owns the camera here.

# Face landmarks (face_landmarks.py) — MediaPipe's Tasks API FaceLandmarker,
# NOT the old mp.solutions.face_mesh (removed entirely from the currently
# -installable mediapipe 1.0.1 — confirmed live this session, no
# mp.solutions attribute exists at all). FaceLandmarker's 478 points
# include iris landmarks (468-477) in the same pass used for head pose,
# covering gaze.py's needs too without a second model/pass.
FACE_LANDMARKER_MODEL_PATH = os.getenv("FACE_LANDMARKER_MODEL_PATH", "face_landmarker.task")
FACE_LANDMARKER_MODEL_URL = os.getenv(
    "FACE_LANDMARKER_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
)
# Reported live: a real, well-lit, front-facing person intermittently read
# "No face" — the box would appear for a frame then vanish, and the
# majority-vote smoothing (7-frame window) would land on NO_FACE whenever
# enough of those frames missed. Measured cause, not a guess: dumping every
# candidate BlazeFace short-range considered (min_detection_confidence
# lowered to 0.01 for the measurement only) across 4 real frames of that
# same person:
#
#   true face score      0.596, 0.817, 0.899, 0.923   (varies with pose/light)
#   best FALSE candidate  0.188, 0.130, 0.118, 0.462   (background clutter)
#
# 0.5 sat almost on top of the worst true-face score (0.596, a margin of
# just 0.096) — ordinary frame-to-frame noise was enough to flip it below
# threshold. False candidates topped out at 0.19 in three of four frames
# (the one 0.462 outlier was also the smallest box, so it loses
# detect_primary_face's area-first tiebreak to the real face regardless).
# 0.35 sits roughly in the middle of that gap: a real margin over the worst
# true-face score seen, and still well clear of the typical false-candidate
# ceiling.
FACE_DETECTION_MIN_CONFIDENCE = float(os.getenv("FACE_DETECTION_MIN_CONFIDENCE", "0.35"))

# Head pose (head_pose.py, solvePnP) — a face within this many degrees of
# yaw/pitch on BOTH axes counts as "facing the camera" for the ENGAGED
# check. Starting values per spec; not yet validated against real users at
# varied desk/camera distances — tighten/loosen per real usage.
LOOKING_TOWARD_CAMERA_YAW = float(os.getenv("LOOKING_TOWARD_CAMERA_YAW", "20.0"))
LOOKING_TOWARD_CAMERA_PITCH = float(os.getenv("LOOKING_TOWARD_CAMERA_PITCH", "20.0"))

# Gaze (gaze.py) — iris-center position within the eye, normalized to a
# 0 (looking away) .. 1 (looking at camera) score. Used as SUPPORTING
# evidence alongside head pose, never the sole signal (spec: "head pose
# remains the primary large-direction signal") — a gaze_score at or above
# this floor is required for ENGAGED, but a low score alone does not by
# itself force DISTRACTED the way a large head-pose deviation does.
GAZE_TOWARD_THRESHOLD = float(os.getenv("GAZE_TOWARD_THRESHOLD", "0.5"))

# Emotion model (emotion_model.py) — kept swappable per spec: point this at
# a local fine-tuned checkpoint directory later with no code change. See
# that module's docstring for why the processor loader tries
# AutoImageProcessor first and falls back to ViTImageProcessor — the
# default model's own preprocessor_config.json declares a processor class
# name ("ViTFeatureExtractor") removed from current transformers, confirmed
# live this session; ViTImageProcessor loads the identical config fine.
EMOTION_MODEL_NAME = os.getenv("EMOTION_MODEL_NAME", "HardlyHumans/Facial-expression-detection")
EMOTION_DEVICE = os.getenv("EMOTION_DEVICE", "auto")  # "auto" | "cpu" | "cuda"
# Two DIFFERENT questions, deliberately different thresholds:
#   EMOTION_CONFIDENCE_THRESHOLD  "is the model saying ANYTHING with real
#                                  conviction?" — below this, MOOD_UNKNOWN
#                                  rather than a forced NEUTRAL (spec:
#                                  "never force a prediction when confidence
#                                  is extremely low").
#   HAPPY_CONFIDENCE_THRESHOLD    "is it specifically confident about HAPPY?"
#                                  — stricter, because a false HAPPY is a
#                                  more visible mistake than defaulting to
#                                  NEUTRAL. A score in between (the model has
#                                  SOME real opinion, just not "happy") reads
#                                  NEUTRAL, not UNKNOWN — that gap is the
#                                  point of keeping the two separate.
EMOTION_CONFIDENCE_THRESHOLD = float(os.getenv("EMOTION_CONFIDENCE_THRESHOLD", "0.35"))
HAPPY_CONFIDENCE_THRESHOLD = float(os.getenv("HAPPY_CONFIDENCE_THRESHOLD", "0.60"))
# The model's own label for "happy" and its neutral-ish labels — kept as
# config rather than hardcoded so a differently-labeled custom model (spec:
# "later allow a custom model") doesn't require a code change, only an env
# var update.
EMOTION_HAPPY_LABEL = os.getenv("EMOTION_HAPPY_LABEL", "happy")
EMOTION_NEUTRAL_LABELS = [s.strip() for s in os.getenv("EMOTION_NEUTRAL_LABELS", "neutral").split(",") if s.strip()]

# State engine (emotion_state_engine.py) — debounce/stability durations
# (seconds) a CANDIDATE state must hold continuously before it becomes the
# DISPLAYED state; this is the only mechanism allowed to change the
# displayed state, specifically to prevent the rapid ENGAGED<->DISTRACTED
# flicker the spec calls out. Per-state because HAPPY should register
# almost immediately (a smile is a short, deliberate expression) while
# DISTRACTED should NOT fire on a brief, normal glance away.
HAPPY_STABLE_TIME = float(os.getenv("HAPPY_STABLE_TIME", "0.5"))
# Mood settles back to NEUTRAL a little slower than it fires HAPPY, so a
# smile that briefly drops below the confidence threshold mid-smile doesn't
# blink the readout back and forth.
NEUTRAL_STABLE_TIME = float(os.getenv("NEUTRAL_STABLE_TIME", "0.8"))
ENGAGED_STABLE_TIME = float(os.getenv("ENGAGED_STABLE_TIME", "1.0"))
DISTRACTED_STABLE_TIME = float(os.getenv("DISTRACTED_STABLE_TIME", "1.75"))
# NO_FACE still gets a short debounce (spec says "if no face is detected,
# state = NO_FACE" unconditionally, but also "never make the final decision
# from one frame") — short enough to feel immediate, long enough that one
# missed detection on an otherwise-present face doesn't flash NO_FACE.
NO_FACE_STABLE_TIME = float(os.getenv("NO_FACE_STABLE_TIME", "0.3"))
# How many raw observations the rolling history keeps for majority-vote
# -style smoothing within the ~1-2s window these stable-times span, at the
# ~10-15 FPS inference rate below.
EMOTION_HISTORY_SIZE = int(os.getenv("EMOTION_HISTORY_SIZE", "45"))

# Person-presence fallback (emotion_presence.py). MediaPipe's face detector
# is frontal-biased and sees nothing once a head turns past roughly profile,
# so "nobody is in the room" and "someone is here but turned away" both
# arrive as face_detected=False. That made a person who turned away read
# NO_FACE, when the rule for this feature is that looking away is
# DISTRACTED. Running the person detector (the same yolov8n the rest of the
# app already uses) only when no face was found restores the distinction.
#
# It runs in the API process, so the interval matters: without it an empty
# room would run YOLO at the full send rate forever. 0.5s is frequent enough
# that walking into frame registers within one DISTRACTED_STABLE_TIME window,
# and the detector is skipped entirely on any frame where a face WAS found.
# How often the ViT expression classifier may actually run. The two axes
# have very different cost and very different update needs:
#
#   attention (head pose + gaze, from landmarks already computed) is cheap
#             and must be responsive — you notice a lag in "am I looking
#             at the camera"
#   mood      (ViT, 86M params) is by far the most expensive thing in the
#             pipeline and changes slowly — a smile lasts seconds
#
# Measured on this machine with the API process pinned to its 2 reserved
# cores: 3.9s per classification at 1 torch thread, 5.3s at 2, 12.2s at 8.
# Running it per frame made the whole feed sit at ~19s per update. Running
# it on an interval and reusing the last expression in between keeps
# attention at the full frame rate while mood refreshes often enough that
# a smile still registers.
#
# This interval must comfortably EXCEED how long one classification takes,
# or it saves nothing: at 2.0s against a ~4s classification, the next run
# was already due the moment the previous finished, so it effectively ran
# every frame and every frame cost ~3s. At 6s there is real space between
# runs, so the frames in between only pay for landmarks/pose/gaze and the
# attention axis stays responsive.
#
# The honest fix is a smaller expression model — a ViT-base is heavy for
# 8-class expression, and on a GPU or a small CNN this could drop toward
# zero. Until then this trades mood freshness for a usable frame rate.
EMOTION_CLASSIFY_INTERVAL_SECONDS = float(os.getenv("EMOTION_CLASSIFY_INTERVAL_SECONDS", "6.0"))

# How long a face that has just gone out of view still counts as proof
# somebody is there. Turning to profile drops the face detector instantly,
# but the person obviously hasn't left — and the YOLO presence check only
# refreshes every couple of seconds, so on its own it can still be saying
# "nobody here" while someone sits right in front of the camera. Long
# enough to cover a turn and the presence refresh behind it; short enough
# that actually walking away still reads as an empty frame quickly.
EMOTION_FACE_MEMORY_SECONDS = float(os.getenv("EMOTION_FACE_MEMORY_SECONDS", "3.0"))

EMOTION_PRESENCE_ENABLED = os.getenv("EMOTION_PRESENCE_ENABLED", "true").lower() == "true"
EMOTION_PRESENCE_INTERVAL_SECONDS = float(os.getenv("EMOTION_PRESENCE_INTERVAL_SECONDS", "0.5"))
# Smaller than the surveillance path's 640: this frame is a webcam close-up
# where a present person fills much of it, so it doesn't need the resolution
# the reception camera needed to catch a distant body.
EMOTION_PRESENCE_IMGSZ = int(os.getenv("EMOTION_PRESENCE_IMGSZ", "320"))
EMOTION_PRESENCE_CONFIDENCE = float(os.getenv("EMOTION_PRESENCE_CONFIDENCE", "0.4"))

# Target inference cadence — the FRONTEND is what actually throttles how
# often a frame is sent (see EmotionEngagement.jsx), this just documents/
# validates the intended rate; the camera PREVIEW itself stays ~30fps
# locally in the browser regardless, since it's rendered straight from the
# webcam's own <video> element with no backend round-trip.
EMOTION_TARGET_INFERENCE_FPS = float(os.getenv("EMOTION_TARGET_INFERENCE_FPS", "12.0"))

# --- Frame quality gate (face_quality.py) -----------------------------------
# "Accuracy is more important than always producing an answer" — a face that
# is too small, too blurry, too dark/bright, or too low-confidence should
# read UNKNOWN on both axes rather than a confident-looking guess from a bad
# frame. Same classical-CV approach and scoring shape as peopleid_quality.py
# (Laplacian-variance blur, mean brightness, weighted composite), not reused
# directly because that module is built around InsightFace's 5-point kps and
# a different call signature — this one works from face_landmarks.py's
# DetectedFace instead. Soft-normalized against a floor rather than a hard
# min/max, so a slightly soft frame is penalized in the composite rather
# than binary pass/fail; HARD_MIN values below are the only true cliff-edge
# rejections.
FACE_QUALITY_MIN_SIZE_PX = int(os.getenv("FACE_QUALITY_MIN_SIZE_PX", "40"))
FACE_QUALITY_BLUR_VARIANCE_FLOOR = float(os.getenv("FACE_QUALITY_BLUR_VARIANCE_FLOOR", "60.0"))
FACE_QUALITY_BLUR_VARIANCE_HARD_MIN = float(os.getenv("FACE_QUALITY_BLUR_VARIANCE_HARD_MIN", "15.0"))
FACE_QUALITY_BRIGHTNESS_HARD_MIN = float(os.getenv("FACE_QUALITY_BRIGHTNESS_HARD_MIN", "25.0"))
FACE_QUALITY_BRIGHTNESS_HARD_MAX = float(os.getenv("FACE_QUALITY_BRIGHTNESS_HARD_MAX", "235.0"))
FACE_QUALITY_WEIGHT_SIZE = float(os.getenv("FACE_QUALITY_WEIGHT_SIZE", "0.25"))
FACE_QUALITY_WEIGHT_BLUR = float(os.getenv("FACE_QUALITY_WEIGHT_BLUR", "0.30"))
FACE_QUALITY_WEIGHT_BRIGHTNESS = float(os.getenv("FACE_QUALITY_WEIGHT_BRIGHTNESS", "0.15"))
FACE_QUALITY_WEIGHT_DET = float(os.getenv("FACE_QUALITY_WEIGHT_DET", "0.30"))
FACE_QUALITY_MIN_SCORE = float(os.getenv("FACE_QUALITY_MIN_SCORE", "0.45"))

# --- Emotion probability smoothing (emotion_api.py's _EmotionCache) --------
# The classifier only actually runs once per EMOTION_CLASSIFY_INTERVAL_SECONDS
# (it's too slow to run per-frame — see that setting's own comment), so
# "temporal smoothing" for mood has to mean smoothing across successive REAL
# classifier runs, not across frames (the majority-vote window mostly just
# sees the same cached value repeated). EMA over the full probability
# distribution, not just the top label, is what actually gives one noisy run
# real resistance: a single flukey run flips the TOP LABEL easily but moves
# an already-confident EMA distribution only a little. alpha is the weight
# on the new run; 1.0 disables smoothing entirely (matches old behavior).
EMOTION_EMA_ALPHA = float(os.getenv("EMOTION_EMA_ALPHA", "0.5"))

# --- Smile detection (smile.py) --------------------------------------------
# A second, independent signal alongside the classifier's own "happy" output
# — mouth geometry from landmarks, not learned. mouth_width is measured
# against inter-eye distance (scale-invariant: works the same whether the
# face fills the frame or sits far from the camera) and compared to this
# person's own RELAXED baseline width, established from the first
# SMILE_BASELINE_SAMPLES good-quality observations of a session (a neutral
# mouth is already wider on some faces than others — comparing to a fixed
# universal ratio would bias toward whoever it was tuned on). Not validated
# against a labeled ground-truth dataset, same honest caveat as head_pose.py
# and gaze.py's own sign/threshold comments — tune SMILE_SCORE_THRESHOLD
# against real use before trusting it for anything beyond a rough signal.
SMILE_BASELINE_SAMPLES = int(os.getenv("SMILE_BASELINE_SAMPLES", "10"))
# How far mouth-width must stretch beyond this person's own relaxed
# baseline (as a ratio) before smile_score reads 1.0. 1.0 = no stretch at
# all (score floors at 0), SMILE_STRETCH_FOR_MAX_SCORE = full smile.
SMILE_STRETCH_FOR_MAX_SCORE = float(os.getenv("SMILE_STRETCH_FOR_MAX_SCORE", "1.18"))
SMILE_SCORE_THRESHOLD = float(os.getenv("SMILE_SCORE_THRESHOLD", "0.45"))

# --- Attention: UNKNOWN state + look-at-screen calibration ------------------
# UNKNOWN is distinct from both NO_FACE (nobody in frame) and DISTRACTED
# (clearly looking away): a face IS present but the frame quality is too
# poor to trust a pose/gaze reading from it. Forcing DISTRACTED from a bad
# frame would be exactly the "confident wrong answer" the quality gate
# exists to prevent.
ATTENTION_UNKNOWN_STABLE_TIME = float(os.getenv("ATTENTION_UNKNOWN_STABLE_TIME", "0.5"))

# A laptop webcam is rarely mounted dead-center on the screen the user is
# actually looking at, so yaw=0/pitch=0 is not a reliable "looking at
# screen" baseline for every setup. Calibration records this person's own
# resting yaw/pitch while they look at the screen normally, and attention is
# then judged against the DEVIATION from that baseline instead of the raw
# angle. Optional: a session that never calibrates just uses raw angles
# against LOOKING_TOWARD_CAMERA_YAW/PITCH exactly as before.
CALIBRATION_SECONDS = float(os.getenv("CALIBRATION_SECONDS", "2.0"))
CALIBRATION_MIN_SAMPLES = int(os.getenv("CALIBRATION_MIN_SAMPLES", "8"))

# --- POSTER V2 alternate emotion engine (emotion_model_poster.py) ----------
# EMOTION_ENGINE picks which classifier init_models() (emotion_api.py)
# constructs behind the EmotionClassifier-compatible classify() interface —
# "vit" (default, unchanged) is the existing HF ViT model (emotion_model.py);
# "poster" is POSTER V2 (backend/external/POSTER_V2-main), a heavier
# CNN+ViT ensemble that computes its own internal face-landmark features
# from the crop itself rather than consuming MediaPipe's landmarks. Nothing
# downstream of classify() (EMA smoothing, smile combination, the state
# engine, the WebSocket payload, the frontend) knows or cares which engine
# produced the scores — see emotion_api.py's init_models().
EMOTION_ENGINE = os.getenv("EMOTION_ENGINE", "vit")  # "vit" | "poster"

POSTER_ROOT_DIR = os.getenv(
    "POSTER_ROOT_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "POSTER_V2-main")),
)
# The two BACKBONE checkpoints (ir50.pth, mobilefacenet_model_best.pth.tar)
# are not separately configurable — POSTER's own model code
# (models/PosterV2_7cls.py) loads them from a `pretrained_dir` that
# defaults to POSTER_ROOT_DIR/models/pretrain, matching the upstream
# README's documented "put the pretrain folder under models" layout.
# POSTER_CHECKPOINT_PATH is the separate, FINAL fine-tuned classification
# head+backbone checkpoint (RAF-DB/AffectNet/CAER-S) — the one that
# actually determines what the 7 output classes mean.
POSTER_CHECKPOINT_PATH = os.getenv(
    "POSTER_CHECKPOINT_PATH",
    os.path.join(POSTER_ROOT_DIR, "checkpoint", "raf-db-model_best.pth"),
)
POSTER_NUM_CLASSES = int(os.getenv("POSTER_NUM_CLASSES", "7"))

# CLASS ORDER IS NOT VERIFIED BY THIS CODEBASE — left empty on purpose.
# The upstream repo has genuinely conflicting evidence for what its 7
# output indices mean: models/matrix.py hardcodes one plotting order
# (Surprise/Fear/Anger/Happy/Sad/Disgust/Neutral) for a single past
# experiment's confusion matrix, which does NOT match the officially
# published RAF-DB numeric label convention (Surprise/Fear/Disgust/Happy/
# Sad/Anger/Neutral), and the real ground truth is whatever alphabetical
# ImageFolder ordering the SPECIFIC checkpoint you download was actually
# trained with — information this repo does not contain. Guessing wrong
# here would make the model confidently report Happy when it means
# Angry/Disgust, silently. emotion_model_poster.py therefore refuses to
# attach human labels to scores while this is empty (returns "class_0".."
# class_6" instead) — set this only after running
# backend/scripts/verify_poster_label_order.py against the real checkpoint
# and labeled sample images. Comma-separated, index 0 first, e.g.:
# POSTER_LABEL_ORDER=surprise,fear,disgust,happy,sad,angry,neutral
POSTER_LABEL_ORDER = [s.strip() for s in os.getenv("POSTER_LABEL_ORDER", "").split(",") if s.strip()]

# POSTER was trained on RAF-DB's "aligned" crops (rotation-normalized so
# the eye line is horizontal — see README.md's "train_00001_aligned.jpg"
# naming). The existing pipeline's crop is a plain axis-aligned bounding
# box, which POSTER will still run on without error, just possibly below
# its published benchmark accuracy. Off by default so POSTER works with
# zero other pipeline changes first; flip on to test whether alignment
# (computed from the existing MediaPipe eye landmarks, no new landmark
# pass) measurably helps once a real checkpoint is in place.
POSTER_ALIGN_FACE = os.getenv("POSTER_ALIGN_FACE", "false").strip().lower() == "true"


# Root log level for both the main API process (main.py) and each per-camera
# detection worker process (detection_worker.py runs in a separate OS
# process — see pipeline.py's module docstring — so it configures its own
# logging independently; this one env var controls both). The recognition
# pipeline's per-frame stage tracing (frame received -> sent to worker ->
# recognized -> stored) logs at DEBUG specifically so it stays silent by
# default and can be switched on for a session without a code change when
# actively diagnosing an accuracy/latency issue.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
