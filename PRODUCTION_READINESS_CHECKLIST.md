# Production Readiness Checklist

Status as of the `feature/production-hardening` merge. `[x]` = done and
verified by tests or a live run on this codebase. `[ ]` = still requires an
action outside the code (listed under "Before go-live").

## Security
- [x] Real admin authentication (email + password, `admin_users`)
- [x] Secure password hashing (PBKDF2-SHA256, 600k iterations, timing-safe)
- [x] Authorization on every endpoint (111 HTTP operations; only 5 intentionally public)
- [x] Tenant isolation (cameras, sites, staff count HTTP + websockets; IDOR tests)
- [x] Upload validation (id allowlist, magic bytes + decode, size cap, re-encode, no traversal)
- [x] Login rate limiting (per IP and per username, admin and client)
- [x] No secrets in git (`backend/.env` untracked; CI guard)
- [x] No biometric data tracked in the current tree (CI guard)
- [ ] Biometric data purged from git **history** (see GIT_DATA_CLEANUP.md; needs team decision + force-push)
- [ ] HTTPS on the server (nginx config with HSTS provided in `deploy/`; certificate must be issued on the server)
- [x] Secure CORS (explicit origins; https required in production)
- [x] Secure settings API (admin only, validated, audited)
- [ ] Camera/NVR credentials rotated (they were in git history in `app.db`)

## Reliability
- [x] Camera watchdog (stalled reader killed and restarted)
- [x] RTSP connect/read timeouts
- [x] Automatic reconnect with jittered exponential backoff
- [x] Model failure recovery (backoff, rebuild after repeated failures)
- [x] Feature recovery (face recognition, person detection, footfall, staff, intrusion, expression)
- [x] Graceful shutdown (bounded drain, readers and workers stopped; verified live)
- [x] Orphaned readers exit when the backend dies (verified with kill)
- [x] Health endpoint (`/health`)
- [x] Readiness endpoint (`/ready`)

## Data
- [x] Relative image paths (+ migration script, legacy paths still readable)
- [x] Retention policy (configurable, never touches training data)
- [x] Database cleanup (batched)
- [x] Snapshot cleanup (Re-ID, resolved alerts)
- [x] Session cleanup
- [x] Model backup cleanup (keep newest N)
- [x] Centralized, context-managed DB connections with busy timeout

## Deployment
- [x] Reproducible dependencies (full import audit; clean venv install + full test pass)
- [x] Clean installation documented (DEPLOYMENT.md)
- [x] Automated tests in CI (ruff, pytest, frontend lint/build, audit)
- [x] CI/CD protection (deploy only on tag/manual, CI on same SHA, pinned host key)
- [x] Secret management (server EnvironmentFile; startup refuses unsafe prod config)
- [x] Deployment health check (`/health` + `/ready`)
- [x] Rollback (automatic on failed deploy; manual procedure)
- [x] Offline model provisioning with checksums
- [ ] GitHub secrets/variables + `production` environment reviewers configured
- [ ] Server migrated to the new systemd unit / env file (DEPLOYMENT.md section 11)

## Testing
- [x] Auth tests
- [x] Upload security tests
- [x] DB tests (rollback, cleanup, concurrency)
- [x] Camera tests (recovery, watchdog, orphan handling)
- [x] RTSP tests (unavailable, wrong credentials, interruption, corrupt frames, heartbeats)
- [x] Face tests (pipeline failure/recovery, classifier compatibility)
- [x] API tests (authorization, settings, health)
- [x] Tenant isolation tests
- [x] Live end-to-end run against real cameras (startup, login, RTSP, camera failure and recovery, shutdown, restart)
- [x] Object detection: service + API tests; live run on 4 cameras, offline, no stdout output, 4 target classes only

## Before go-live (human actions)
1. Decide on and perform the git history purge (GIT_DATA_CLEANUP.md), and treat the data as exposed if the repo was ever public.
2. Rotate NVR/camera passwords and client license passwords.
3. Configure GitHub secrets/variables and the `production` environment (DEPLOYMENT.md section 8).
4. On the server: env file, systemd unit, nginx + certificate, `migrate_paths --apply`, `create-admin` (DEPLOYMENT.md section 11).
5. Fix camera 9 ("entry/exit"): port 31 and stream path `subtype=011` look mistyped; it never connects.
