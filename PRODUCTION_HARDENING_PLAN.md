# Production Hardening Plan

Branch: `feature/production-hardening` (created from `main` @ `697aabf`).
Pre-existing uncommitted work was committed first, unchanged, as `1ee8ffc`, so
every hardening change below is a separate, reviewable diff on top of it.

Nothing on this branch is pushed. `main` auto-deploys to EC2 on push, so this
branch must be reviewed and merged deliberately (see Phase 3).

Severity: **C** = critical (blocks any production use), **H** = high,
**M** = medium, **L** = low.

---

## Phase 1: Security blockers

| # | Sev | Issue | Files | Fix | Tests | Deploy impact | Rollback |
|---|-----|-------|-------|-----|-------|---------------|----------|
| 1.1 | C | Admin login accepts any email and has no password | `main.py`, `auth.py`, frontend `client.js`, `Login.jsx` | New `admin_users` table (PBKDF2-SHA256 at 600k iterations, the same stdlib approach `license_db` already uses, so no new dependency). `/api/auth/login` checks email and password and gives a generic 401 on failure. First admin is created with `python -m app.manage create-admin`, or from `ADMIN_EMAIL`/`ADMIN_PASSWORD` when no admin exists yet. | valid, wrong password, unknown user, missing password, expired, malformed, unauthorized, authorized | **Breaking:** an admin account must be created before anyone can log in. Existing admin sessions are revoked by migration. | Revert the commit; the `admin_users` table is additive and harmless. |
| 1.2 | C | Enrollment upload builds file paths from `person_id` and the uploaded extension | `face_routes.py` (`/enroll`, `sync-from-identity`) | New `uploads.py`: validate `person_id` against `^[A-Za-z0-9_-]{1,64}$`, 5 MB cap, decode with OpenCV and re-encode server-side to `.jpg` (the extension and bytes from the client are never trusted), UUID filename, resolved path must stay inside `face_enroll/`, write only after decode and face detection succeed, `O_EXCL` create. | `../`, `..\`, absolute, drive letter, URL-encoded, null byte, fake extension, oversized, non-image, duplicate | None | Revert |
| 1.3 | H | Clients can read every tenant's staff data (HTTP and websockets) | `staff/routes.py`, `main.py` | Staff HTTP routes: admin sees everything; a client needs the `attendance` feature (what the sidebar gates on) and results are filtered to their licensed cameras. `/ws/staff` is filtered per client. `/ws/staff/debug/{id}` goes through the camera authorization check. | client A vs camera B (HTTP and ws), admin unaffected | None | Revert |
| 1.4 | H | `/api/settings` GET/PUT has no auth; PUT accepts any dict | `main.py` | Admin only; Pydantic model with bounds (`detection_fps` 0.1 to 30); changes written to the audit log. Remove the dead unauthenticated `/api/alerts` stubs (shadowed by `alerts_routes`) and put `/api/stats` behind admin. | unauthenticated 401, client 403, invalid value 422, unknown key 422 | None | Revert |
| 1.5 | H | No login rate limiting | `ratelimit.py` (new), admin and client login | In-process sliding window keyed by IP and by username (defaults: 10 per IP and 5 per username per 5 min), 429 with `Retry-After`, failures logged without passwords. Configurable. Single-worker deployment makes in-process state valid (documented). | limit hit, per-user isolation, window expiry, success doesn't lock | None | Revert |
| 1.6 | H | Frontend fake auth and mock fallbacks | `client.js`, `AuthContext.jsx`, `App.jsx`, `Login.jsx`, `People.jsx` | Remove `/signup` and the `demo-token`; login sends a password; `getPeople` shows an error instead of mock people; `getValidatedPeople` returns an empty list; only 401 clears the session (403 no longer logs you out). `VITE_DEMO_MODE` exists only as an explicit opt-in for dev data. | lint and build | Users must use real credentials | Revert |
| 1.7 | C | Biometric data, `.env` and runtime files tracked in git | `.gitignore`, git index | Ignore all of `backend/data/`, backups, `.env`; `git rm --cached` the tracked face captures and `backend/.env` (files stay on disk). History purge is **documented, not executed**. See `GIT_DATA_CLEANUP.md`. | `git ls-files` audit | Deploy host keeps its own `.env` (Phase 3) | Re-add to the index |
| 1.8 | H | Secrets: `JWT_SECRET` random per process; hardcoded external IP over HTTP | `config.py`, `face_routes.py`, `face_training_routes.py`, `client.js` | `APP_ENV=production` refuses to start without `JWT_SECRET`, an admin account and `IDENTITY_SERVICE_BASE`; the identity service URL comes from config; the browser no longer calls the external HTTP service directly (backend proxy). | config validation tests | Production `.env` needs these set | Revert |

## Phase 2: Reliability blockers

| # | Sev | Issue | Files | Fix | Tests |
|---|-----|-------|-------|-----|-------|
| 2.1 | C | One exception permanently turns face recognition off for that camera | `camera_stream.py` | New `resilience.py` `FeatureHealth` (RUNNING, DEGRADED, FAILED, RECOVERING, STOPPED): failures back off exponentially (base 5s, max 300s) and then retry; the component is rebuilt after repeated failures; state is exposed to the health endpoints. | model exception, then recovery; repeated failure backs off; unrelated feature unaffected |
| 2.2 | C | Person detection latches `failed` forever, which stops footfall, staff count and intrusion | `person_detection.py` | Same `FeatureHealth`; the model is reloaded on recovery. | same |
| 2.3 | H | Staff, expression and footfall per-camera latches | `staff/service.py`, `expression.py`, `footfall.py` | Same pattern where a permanent latch exists. | targeted |
| 2.4 | C | RTSP reader can block forever; no stall watchdog | `rtsp_reader.py`, `camera_stream.py` | Open and read timeouts passed to `VideoCapture`; reconnect with exponential backoff and jitter (2s to 60s; protects NVR lockout); the parent watchdog kills and restarts a reader whose frame counter hasn't moved in `CAMERA_FRAME_TIMEOUT`; stream state RUNNING, RECONNECTING, STALLED, STOPPED plus `last_frame_at`. | backoff schedule, stall detection with a fake reader, timeouts passed through |
| 2.5 | H | No graceful shutdown | `main.py`, `camera_stream.py`, services | FastAPI `lifespan`: stop streams, signal readers, join with timeout then terminate, shut down executors, stop background loops. | start then stop leaves no child processes |
| 2.6 | H | No liveness or readiness endpoint (the deploy script already curls `/health`, which doesn't exist) | `health_routes.py` (new) | `/health` for liveness; `/ready` checks the DB, required model files, config and per-camera/per-feature state. No secrets and no personal data. | payload shape, 503 when not ready |

## Phase 3: Dependencies and deployment

| # | Sev | Issue | Fix |
|---|-----|-------|-----|
| 3.1 | C | `requirements.txt` missing torch, torchvision, torchreid, transformers, numpy, PyYAML, tensorflow, and more | Full import audit, then `requirements.txt` (runtime, pinned to the versions verified working on this machine), `requirements-ml-optional.txt` (tensorflow for the webcam demo only), `requirements-dev.txt` (pytest, ruff). CPU and GPU torch notes. |
| 3.2 | C | scikit-learn 1.9.1 pin vs installed 1.5.2; the pickled classifier depends on the version | Pin to the version actually in use; write `classifier.meta.json` with the sklearn version; refuse to load on a mismatch, with a clear log and fallback to the gallery, instead of crashing per frame. |
| 3.3 | H | Models download at runtime | `MODEL_DIR`, `MODEL_OFFLINE_MODE` (default true in production): set `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, InsightFace root inside `MODEL_DIR`; startup validates required files; `scripts/fetch_models.py` provisions them. |
| 3.4 | C | Push to `main` deploys with no tests; `.env` overwritten; `StrictHostKeyChecking=no`; health port mismatch | Split into `ci.yml` (ruff, pytest, frontend lint and build on every PR and push) and `deploy.yml` (manual or tag only, `needs` CI, protected `production` environment, pinned host key from a secret, keeps the server `.env`, records the previous commit, health check, automatic rollback to the previous commit on failure). |

## Phase 4: Data and database

| # | Sev | Issue | Fix |
|---|-----|-------|-----|
| 4.1 | H | 12 separate `sqlite3.connect` sites, inconsistent timeouts; `face_db` leaks connections on error | `db.py`: `connect(path)` context manager (timeout 30s, `busy_timeout`, `synchronous=NORMAL`, commit, rollback, close). Every module routes through it; `face_db` converted to context-managed connections. |
| 4.2 | H | Absolute image paths in the DB | `storage.py`: store paths relative to `DATA_DIR` (POSIX), resolve at runtime, legacy absolute paths still readable; `scripts/migrate_paths.py` (dry run by default) rewrites existing rows. |
| 4.3 | H | Unbounded growth: pending faces, snapshots, sessions, classifier backups; training cap reached silently | `retention.py` daemon (hourly): expired sessions, pending captures older than `FACE_PENDING_RETENTION_DAYS` (file and row, unassigned only), Re-ID snapshots older than `SNAPSHOT_RETENTION_DAYS`, alert snapshots of resolved alerts, orphaned files, classifier backups beyond `MODEL_BACKUP_RETENTION_COUNT`. Never touches labeled training data or the active model. A warning is logged and shown on `/ready` when the capture cap is reached. |

## Phase 5: Configuration

One `app/config.py` settings layer: `APP_ENV`, `DATA_DIR`, `DB_PATH`, `MODEL_DIR`, RTSP timeouts and backoff, feature retry, retention, auth and rate limits, CORS, logging, identity service URL. Every module reads from it; `DB_PATH` stops being recomputed in 14 files. The complete `.env.example` is tracked.

## Phase 6: Logging, tests, lint

- `logging_setup.py`: timestamped format, `LOG_LEVEL`, rotating file handler (`LOG_DIR`), redaction filter for `password=`, `token=` and `rtsp://user:pass@`.
- Replace silent `except: pass` on important paths with logged handling.
- New tests: auth, rate limit, uploads, tenant isolation, settings, health, resilience, RTSP backoff and watchdog, db helper, storage paths, retention, config.
- `ruff check`, `pytest`, frontend `npm run lint` and `npm run build`.

## Phase 7: Documentation

`README.md`, `DEPLOYMENT.md`, `API_DOCUMENTATION.md` (generated from the live OpenAPI schema plus the websocket list), `PRODUCTION_READINESS_CHECKLIST.md`, `GIT_DATA_CLEANUP.md`, this file.

---

## Requires a human decision (will not be done automatically)

1. **Rewriting git history** to purge face images, `app.db`, the classifier and `.env` from all commits and branches. This needs a force-push and team coordination. Procedure is in `GIT_DATA_CLEANUP.md`.
2. **Whether the GitHub repo was ever public or shared outside the team.** If so, treat this as a personal-data exposure (DPDP Act / GDPR obligations).
3. **Rotating credentials** that have been in git or plaintext: camera or NVR passwords (the same `admin` credential is used on every camera), the EC2 deploy key if ever shared.
4. **Merging this branch to `main`**, which auto-deploys under the old workflow until the new workflow is on `main`.
5. **Production host details** (EC2 IP, systemd unit, nginx config, TLS certificate) live on the server and are not in the repo, so they can only be documented, not verified from here.
6. **Hardware or GPU choice:** this machine is CPU-only; GPU requirements are documented but not tested.
