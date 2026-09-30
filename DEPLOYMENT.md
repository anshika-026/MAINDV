# Deployment Guide

This covers a fresh production install, upgrading the existing EC2 server
from the old auto-deploy setup, and day-to-day operations. Commands assume
Ubuntu 22.04/24.04 and the default paths used by `deploy/` (repo at
`/home/ubuntu/Deco-vision`). Adjust them if yours differ.

## 1. Requirements

| | |
|---|---|
| OS | Ubuntu 22.04 or 24.04 LTS (x86_64). Windows works for development. |
| Python | **3.12** (tested on 3.12.7) |
| Node.js | 22 LTS (frontend build only) |
| CPU / RAM | At least 8 cores and 16 GB RAM for 4 to 5 cameras with all analytics on (CPU-only inference). |
| Disk | 20 GB free for the app, venv and models, plus data: about 50 MB per 1,000 face captures and a database of a few hundred MB. Retention keeps this bounded (section 9). |
| GPU (optional) | NVIDIA with CUDA 12.x drivers. See section 3.3. Not required, and not what this project is tested on. |
| Network | The server must reach the cameras/NVR over RTSP. Internet is **not** needed at runtime once models are provisioned (section 4). |
| System packages | `python3.12-venv libgl1 libglib2.0-0 ffmpeg nginx`, plus `certbot` for TLS |

**Single process.** The backend must run as exactly one process with one
worker. Camera readers, the training-capture limit, login rate limiting and
the in-memory staff/footfall state all assume this. `python -m app.serve`
enforces it. Never run it with `--workers` or `--reload`.

## 2. Configuration

All configuration is environment variables, read in `backend/app/config.py`.
The complete, commented list is `backend/.env.example`.

In production, keep it **outside the repository** at
`/etc/deco-vision/backend.env` (loaded by the systemd unit). Deploys never
write it.

```bash
sudo mkdir -p /etc/deco-vision /var/lib/deco-vision /var/log/deco-vision
sudo cp backend/.env.example /etc/deco-vision/backend.env
sudo chown root:root /etc/deco-vision/backend.env && sudo chmod 600 /etc/deco-vision/backend.env
sudo chown ubuntu:ubuntu /var/lib/deco-vision /var/log/deco-vision
sudoedit /etc/deco-vision/backend.env
```

Minimum production values:

```ini
APP_ENV=production
HOST=127.0.0.1
PORT=8811
CORS_ORIGINS=https://vision.example.com
JWT_SECRET=<python -c "import secrets; print(secrets.token_hex(32))">
DATA_DIR=/var/lib/deco-vision
LOG_DIR=/var/log/deco-vision
MODEL_OFFLINE_MODE=true
TRUST_PROXY_HEADERS=true
IDENTITY_SERVICE_BASE=https://identity.example.com   # the face-enrollment service, if used
```

With `APP_ENV=production` the backend **refuses to start** if it's missing
`JWT_SECRET`, has a wildcard or plain-http CORS origin, allows runtime model
downloads, uses a plain-http Identity service, or has no admin account. The
reason is in the log. Check a config without starting the server:

```bash
cd backend && set -a && . /etc/deco-vision/backend.env && set +a && ./venv/bin/python -m app.manage check-config
```

## 3. Install

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv libgl1 libglib2.0-0 ffmpeg nginx git
git clone https://github.com/anshika-026/MAINDV.git /home/ubuntu/Deco-vision
cd /home/ubuntu/Deco-vision/backend
python3.12 -m venv venv
```

### 3.1 Python dependencies (CPU, the default)

Install PyTorch first from the CPU index, then everything else. The default
PyPI torch wheel on Linux is the multi-gigabyte CUDA build.

```bash
./venv/bin/pip install --upgrade pip
./venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
./venv/bin/pip install -r requirements.txt
./venv/bin/pip check
```

- `requirements-optional.txt` adds TensorFlow, which only the laptop-webcam
  Behavior Analytics demo page uses. It isn't needed on camera servers.
- `requirements-dev.txt` adds pytest and ruff.

### 3.2 Dependency constraints (don't casually upgrade these)

- **scikit-learn 1.5.2.** The trained face classifier is a scikit-learn
  pickle. After changing the version, retrain (Face Training page, or
  `python -m app.train_faces`). A model that doesn't run under the installed
  version is refused at load time, and recognition falls back to gallery
  matching (`classifier_io.py`). It won't crash, but accuracy drops until
  you retrain.
- **supervision 0.30.3.** Face tracking uses `sv.ByteTrack`, which is
  deprecated and removed in supervision 0.31. Keep the pin until
  `face_pipeline.py`'s tracker is migrated to the `trackers` package.
- **torchreid 0.2.5** imports `gdown`, `tensorboard` and `scipy` without
  declaring them. They're pinned in `requirements.txt`.

### 3.3 GPU (optional, untested here)

Install the CUDA wheels instead
(`--index-url https://download.pytorch.org/whl/cu124 torch==2.13.0 torchvision==0.28.0`),
replace `onnxruntime` with `onnxruntime-gpu==1.30.0`, and set `REID_DEVICE=cuda`.
The InsightFace provider list in `face_pipeline.py` is CPU-only and needs
`CUDAExecutionProvider` added.

## 4. Models (offline-first)

Production never downloads models at runtime (`MODEL_OFFLINE_MODE=true`).
Provision them once, on the server or on any machine with internet, then
copy them over:

```bash
cd backend
MODEL_OFFLINE_MODE=false ./venv/bin/python -m scripts.fetch_models   # downloads what's missing
./venv/bin/python -m scripts.fetch_models --check                    # verify: presence + SHA-256
```

| Model | Location | Used by |
|---|---|---|
| `yolov8n.pt` | `MODEL_DIR` (default `backend/models`) | person detection: footfall, staff count, intrusion, overlay |
| `yolov8n-face.pt` | `MODEL_DIR` | face detection (manual download, see `backend/FACE_RECOGNITION_WIRING.md`) |
| `yolov8s.pt` | `MODEL_DIR` | person "rescue" sweep for the overlay |
| `osnet_x0_25_msmt17.pth` | `MODEL_DIR` | Re-ID for unique footfall |
| `buffalo_l/` | `INSIGHTFACE_ROOT/models/` (default `~/.insightface`) | face alignment + ArcFace embeddings |
| facial-expression model | Hugging Face cache (`HF_HOME`) | mood / expression (optional) |
| `emotion_model_v3.keras` | `MODEL_DIR` (tracked in git) | webcam demo (optional) |
| `yolo26s.pt`, `imc_general_best.pt`, `office_fixtures.pt` | `MODEL_DIR` | object detection (optional; `fetch_models` copies them from the `object-detection` branch) |

A missing required model makes `/ready` return 503 and names it. The
feature that needs it logs a clear error and keeps retrying, and the other
features keep running.

### Object detection

Detects backpacks, handbags, bottles and laptops on analytics cameras
(`backend/app/object_detection/`). It's **off by default**. Turn it on with
the switch next to **Object Detection** in the sidebar (admin). It costs
about 0.6 s of CPU per camera look.

- One worker thread serves all cameras, each at `OBJECT_DETECTION_FPS` looks
  per second (default 0.2, one look every 5 s), so it can use at most one core.
- Clients see it only if their license includes **Object Detection**, and
  only for their own cameras.
- The module's zero-shot fallback models (YOLO-World, RT-DETR, CLIP) would
  download weights at runtime, so they're off. `OBJECT_DETECTION_FALLBACK=true`
  enables them only when `MODEL_OFFLINE_MODE=false`, and then also needs
  `pip install ensemble-boxes`.

## 5. Database and first admin

Tables are created and migrated automatically on startup (additive only).
Create the first admin account (the password is prompted, never passed on
the command line):

```bash
cd backend && set -a && . /etc/deco-vision/backend.env && set +a
./venv/bin/python -m app.manage create-admin --email ops@example.com --name "Operations"
```

Other commands: `set-password`, `disable-admin`, `enable-admin`,
`list-admins`, `purge-sessions`, `retention`, `check-config`.

Existing data from an older version: stop the backend, then make the stored
image paths portable (dry run first; `--apply` backs up the database before
rewriting):

```bash
./venv/bin/python -m scripts.migrate_paths
./venv/bin/python -m scripts.migrate_paths --apply
```

## 6. Backend service

```bash
sudo cp deploy/deco-vision-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now deco-vision-backend
curl -fsS http://127.0.0.1:8811/health   # {"status":"ok"}
curl -fsS http://127.0.0.1:8811/ready    # 200 when DB, config, admin and models are OK
```

`python -m app.serve` is the only supported start command. It runs one
worker and no reload, drains open connections (live-video websockets) for
10 s, then runs the shutdown. That stops every camera read loop and its RTSP
reader process, and the workers. RTSP reader processes also exit on their
own if the backend dies abruptly, so they can't hold NVR sessions after a
crash.

## 7. Frontend and nginx (HTTPS)

```bash
cd /home/ubuntu/Deco-vision/frontend
npm ci && npm run build            # uses .env.production: API at same-origin /api
sudo cp ../deploy/nginx-deco-vision.conf /etc/nginx/sites-available/deco-vision
sudo sed -i 's/vision.example.com/<your domain>/g' /etc/nginx/sites-available/deco-vision
sudo ln -sf /etc/nginx/sites-available/deco-vision /etc/nginx/sites-enabled/deco-vision
sudo certbot --nginx -d <your domain>
sudo nginx -t && sudo systemctl reload nginx
```

nginx serves the SPA, proxies `/api` and `/ws` to the backend, sets HSTS and
security headers, and logs paths without query strings (websocket URLs carry
the session token).

## 8. CI/CD

- **`.github/workflows/ci.yml`** runs on every push and PR. It checks that no
  data, databases or secrets are tracked, then runs ruff, the backend tests,
  a production-config import check, the frontend lint and build, a check for
  hard-coded hosts in the bundle, and `npm audit`.
- **`.github/workflows/deploy.yml`** deploys **only** on a `v*` tag or a
  manual "Run workflow". It first re-runs CI for the same commit, and needs
  approval if the `production` environment has required reviewers.

  One-time setup in GitHub (Settings > Secrets and variables > Actions, and
  Settings > Environments):

  | Name | Kind | Value |
  |---|---|---|
  | `EC2_PRIVATE_KEY` | secret | deploy user's SSH private key |
  | `EC2_HOST_KEY` | secret | output of `ssh-keyscan -t ed25519 <server>`, which pins the host key |
  | `EC2_HOST` | variable | server address |
  | `EC2_USER`, `APP_DIR`, `BACKEND_PORT`, `SERVICE_NAME` | variables | optional overrides (defaults: `ubuntu`, `/home/ubuntu/Deco-vision`, `8811`, `deco-vision-backend`) |
  | `production` | environment | add required reviewers |

- **Release:** `git tag v1.0.0 && git push origin v1.0.0`, then approve.
- `scripts/deploy_remote.sh` runs on the server. It records the current
  commit, untracks old runtime files so git can't delete them, backs up the
  `.env` files, checks out the exact commit, installs and checks
  dependencies and models, builds the frontend into a new folder and swaps
  it in, restarts, and waits for `/health` and `/ready`. On any failure it
  **rolls back** to the previous commit and exits non-zero.

## 9. Operations

**Health.** `/health` is liveness and always 200 while the process serves.
`/ready` is readiness: DB, configuration, admin account and models, plus
per-camera stream state (RUNNING, CONNECTING, RECONNECTING, STALLED, FAILED,
STOPPED) and per-feature recovery state. It contains no names, hosts or
credentials. Admins get the full detail from `/api/health/details`.

**Logs.** stderr goes to journald (`journalctl -u deco-vision-backend -f`),
plus rotating files `LOG_DIR/backend.log` (20 MB x 10 by default).
Passwords, tokens and RTSP credentials are redacted.

**Retention** (hourly; `python -m app.manage retention` runs it now):

| Data | Kept for | Setting |
|---|---|---|
| unassigned review-queue captures | 30 days | `FACE_PENDING_RETENTION_DAYS` |
| unlabeled / rejected / no-embedding training captures | 60 days | `FACE_CAPTURE_RETENTION_DAYS` |
| Re-ID snapshots | 30 days | `SNAPSHOT_RETENTION_DAYS` |
| snapshots of resolved alerts | 90 days | `ALERT_SNAPSHOT_RETENTION_DAYS` |
| classifier backups | newest 5 | `MODEL_BACKUP_RETENTION_COUNT` |
| expired sessions | removed | always |

Labeled and skipped training data, enrollment photos and the active
classifier are never deleted. `RETENTION_ENABLED=false` turns the cleanup
off. When the training-capture cap (`MAX_TRAINING_CAPTURES`, 15,000) is
reached, collection pauses and a warning is logged hourly.

**Backups.** Everything stateful is in `DATA_DIR`. Take a consistent
database copy while the service runs, then copy the images:

```bash
sqlite3 /var/lib/deco-vision/app.db ".backup '/backup/app-$(date +%F).db'"
rsync -a --exclude 'app.db*' /var/lib/deco-vision/ /backup/data/
```

Store backups encrypted: they contain biometric data.

**Recovery behaviour.** A failing analytics component (face recognition,
person detection, footfall, staff count, intrusion, expression) is retried
with exponential backoff (5 s up to 5 min). After 3 failures in a row it's
rebuilt with its models reloaded, and it's never switched off permanently.
Cameras reconnect with jittered backoff (2 s up to 60 s), which protects NVR
accounts from lockout. A reader hung inside FFmpeg with no frames and no
heartbeat for 20 s is killed and restarted.

## 10. Rollback

- **Automatic:** a failed deploy rolls itself back.
- **Manual:** on the server, the previous commit is in
  `../deco-vision-deploy-backups/PREVIOUS`:

  ```bash
  cd /home/ubuntu/Deco-vision
  SHA=$(cat ../deco-vision-deploy-backups/PREVIOUS) APP_DIR=$PWD bash scripts/deploy_remote.sh
  ```

  Or run the Deploy workflow on the previous tag.
- **Database:** schema changes are additive, so older code runs on a newer
  database. Restore a backup copy only if data itself must be reverted.

## 11. Upgrading the existing EC2 server (one-time)

The old workflow deployed every push to `main`, tracked `backend/.env`, and
started the backend some other way. Before the first deploy with the new
workflow:

1. Create `/etc/deco-vision/backend.env` (section 2). Start from the
   server's current `backend/.env` and add the production settings.
2. Install the new systemd unit (section 6) and nginx config (section 7).
   Keep the port the same as today, or update `BACKEND_PORT`.
3. Set the GitHub secrets and variables listed in section 8.
4. Stop the service, run `scripts/migrate_paths.py --apply` against the
   server database, and create the admin account (section 5).
5. Deploy with a tag. The deploy script keeps the existing `backend/data`
   and `.env` even though the new commit no longer tracks them.
6. Optionally move data out of the repository to `/var/lib/deco-vision`
   and point `DATA_DIR` at it. The stored paths are relative now, so no
   database rewrite is needed.

## 12. Troubleshooting

| Symptom | Check |
|---|---|
| Service exits at start | `journalctl -u deco-vision-backend -n 50`: "Refusing to start with an unsafe production configuration" lists what to fix. |
| `/ready` is 503 | The body names the failing check (database, configuration, admin_account, models). |
| Camera stuck RECONNECTING | Detail says "failed to open stream": check host, port, stream path and credentials in Camera Management. Test with `ffprobe rtsp://...`. Too many clients on one NVR also cause this. |
| Camera STALLED repeatedly | Network loss to the NVR. The watchdog restarts the reader each time `CAMERA_FRAME_TIMEOUT` passes without a frame. |
| "face classifier ... cannot be used" | scikit-learn changed since training: retrain. |
| Login returns 429 | Too many failed attempts from that IP or for that user. Wait `Retry-After` seconds. |
| People page: "Identity enrollment service is not configured" | Set `IDENTITY_SERVICE_BASE`. |
| Windows dev: `DLL load failed ... Application Control policy` | Windows Smart App Control is blocking a compiled Python package (seen with scikit-learn and scipy). Retry, or reinstall the package; not an issue on Linux. |
