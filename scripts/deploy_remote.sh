#!/usr/bin/env bash
# Runs ON the production server (piped over SSH by .github/workflows/deploy.yml).
#
#   APP_DIR=/home/ubuntu/Deco-vision SHA=<commit> BACKEND_PORT=8811 \
#   SERVICE_NAME=deco-vision-backend bash deploy_remote.sh
#
# Deploys exactly one commit, keeps the server's own configuration and data,
# verifies liveness + readiness, and rolls back to the previous commit if
# anything fails. Safe to re-run.
set -euo pipefail

: "${APP_DIR:?}" "${SHA:?}"
BACKEND_PORT="${BACKEND_PORT:-8811}"
SERVICE_NAME="${SERVICE_NAME:-deco-vision-backend}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-120}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="${APP_DIR}/../deco-vision-deploy-backups/${STAMP}"

log() { printf '[deploy %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

cd "$APP_DIR"
PREV_SHA="$(git rev-parse HEAD)"
log "current ${PREV_SHA:0:12} -> target ${SHA:0:12}"

# --- 1. Protect server-local config and data ---------------------------------
# Older commits TRACKED backend/.env and backend/data/face_captures. Checking
# out a commit that no longer tracks them would make git DELETE them from
# this working tree. Untrack them in the local index first (files stay on
# disk), and keep a copy of the config regardless.
mkdir -p "$BACKUP_DIR"
for f in backend/.env frontend/.env; do
  [ -f "$f" ] && cp -p "$f" "$BACKUP_DIR/$(echo "$f" | tr / _)"
done
git rm -r --cached --quiet --ignore-unmatch backend/data backend/.env frontend/.env >/dev/null 2>&1 || true

# --- 2. Fetch and check out the exact commit ------------------------------------
git fetch --quiet --tags origin
git cat-file -e "${SHA}^{commit}" 2>/dev/null || git fetch --quiet origin "$SHA"
git checkout --quiet --force --detach "$SHA"
for f in backend/.env frontend/.env; do
  b="$BACKUP_DIR/$(echo "$f" | tr / _)"
  if [ -f "$b" ] && [ ! -f "$f" ]; then cp -p "$b" "$f"; log "restored $f"; fi
done

# NOTE: `set -e` does not apply inside a function used as an `if` condition,
# so every step below fails the function explicitly with `|| return 1`.
install_and_build() {
  log "system packages"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgl1 libglib2.0-0 ffmpeg >/dev/null || return 1

  log "backend dependencies"
  cd "$APP_DIR/backend" || return 1
  [ -x venv/bin/python ] || python3 -m venv venv || return 1
  ./venv/bin/pip install -q --upgrade pip || return 1
  # CPU torch first (see requirements.txt). Set TORCH_INDEX_URL for GPU hosts.
  ./venv/bin/pip install -q --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" \
    torch==2.13.0 torchvision==0.28.0 || return 1
  ./venv/bin/pip install -q -r requirements.txt || return 1
  ./venv/bin/pip check || return 1

  log "models present (offline runtime)"
  ./venv/bin/python -m scripts.fetch_models --check || return 1

  log "frontend build"
  cd "$APP_DIR/frontend" || return 1
  npm ci --silent || return 1
  rm -rf dist.new
  npx vite build --outDir dist.new --emptyOutDir >/dev/null || return 1
  # Atomic swap so nginx never serves a half-written build.
  rm -rf dist.old
  if [ -d dist ]; then mv dist dist.old || return 1; fi
  mv dist.new dist || return 1
}

restart_and_verify() {
  sudo systemctl restart "$SERVICE_NAME" || return 1
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
  until curl -fsS "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; do
    [ $SECONDS -lt $deadline ] || { log "backend /health did not come up in ${HEALTH_TIMEOUT}s"; return 1; }
    sleep 2
  done
  local body
  if ! body="$(curl -fsS "http://127.0.0.1:${BACKEND_PORT}/ready")"; then
    log "backend is not ready: $(curl -sS "http://127.0.0.1:${BACKEND_PORT}/ready" || true)"
    return 1
  fi
  sudo nginx -t -q || { log "nginx config test failed"; return 1; }
  sudo systemctl reload nginx || return 1
  curl -fsS "http://127.0.0.1/" >/dev/null || { log "nginx is not serving the frontend"; return 1; }
  log "ready: ${body:0:300}"
}

rollback() {
  log "ROLLING BACK to ${PREV_SHA:0:12}"
  cd "$APP_DIR"
  git checkout --quiet --force --detach "$PREV_SHA"
  for f in backend/.env frontend/.env; do
    b="$BACKUP_DIR/$(echo "$f" | tr / _)"
    [ -f "$b" ] && [ ! -f "$f" ] && cp -p "$b" "$f"
  done
  (cd "$APP_DIR/backend" && ./venv/bin/pip install -q -r requirements.txt) || true
  (cd "$APP_DIR/frontend" && [ -d dist.old ] && rm -rf dist && mv dist.old dist) || true
  sudo systemctl restart "$SERVICE_NAME" || true
  sudo systemctl reload nginx || true
}

if install_and_build && restart_and_verify; then
  echo "$SHA" > "$APP_DIR/../deco-vision-deploy-backups/CURRENT"
  echo "$PREV_SHA" > "$APP_DIR/../deco-vision-deploy-backups/PREVIOUS"
  log "deployed ${SHA:0:12} (previous ${PREV_SHA:0:12}; roll back with: SHA=$PREV_SHA bash scripts/deploy_remote.sh)"
else
  rollback
  log "deploy of ${SHA:0:12} FAILED and was rolled back"
  exit 1
fi
