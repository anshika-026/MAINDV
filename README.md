# Deco Vision

CCTV analytics for offices: live camera viewing plus face recognition and
attendance, person detection, unique footfall (person Re-ID across gates),
staff count at entrances, intrusion zones, desk analytics, mood/expression,
alerts, object detection (backpack, handbag, bottle, laptop), and a licensed
client portal. The backend is FastAPI with SQLite;
the frontend is React (Vite).

| Document | Contents |
|---|---|
| [DEPLOYMENT.md](DEPLOYMENT.md) | Requirements, configuration, install, models, service, nginx/HTTPS, CI/CD, operations, rollback |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | Every endpoint and websocket, with its access rule (generated from the code) |
| [PRODUCTION_READINESS_CHECKLIST.md](PRODUCTION_READINESS_CHECKLIST.md) | What's done, and what still needs a human action before go-live |
| [PRODUCTION_HARDENING_PLAN.md](PRODUCTION_HARDENING_PLAN.md) | The hardening findings and how each was addressed |
| [GIT_DATA_CLEANUP.md](GIT_DATA_CLEANUP.md) | Personal data in git history and how to purge it |
| [backend/FACE_TRAINING.md](backend/FACE_TRAINING.md), [backend/FACE_RECOGNITION_WIRING.md](backend/FACE_RECOGNITION_WIRING.md) | Face recognition pipeline, labeling and classifier training |
| [training/emotion/README.md](training/emotion/README.md) | Training, evaluating and deploying the mood (emotion) model |
| [PEOPLE_IDENTIFICATION_ARCHITECTURE.md](PEOPLE_IDENTIFICATION_ARCHITECTURE.md), [unique-footfall-export/UNIQUE_FOOTFALL.md](unique-footfall-export/UNIQUE_FOOTFALL.md) | Identity and Re-ID design |

## Layout

```
backend/            FastAPI app (app/), tests/, scripts/, models/ (weights, not in git)
  app/main.py       app, routes, lifespan (startup/shutdown)
  app/serve.py      the supported entry point:  python -m app.serve
  app/config.py     every setting (environment variables; see backend/.env.example)
frontend/           React + Vite single-page app
deploy/             systemd unit and nginx site config
training/emotion/   offline training/evaluation of the mood (emotion) model
scripts/            deploy_remote.sh (run on the server by the Deploy workflow)
.github/workflows/  ci.yml (every push/PR) and deploy.yml (tags / manual only)
```

## Local development

Backend (Python 3.12):

```bash
cd backend
python -m venv venv && . venv/bin/activate           # Windows: venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                                  # then edit; APP_ENV=development
MODEL_OFFLINE_MODE=false python -m scripts.fetch_models   # once: download model weights
python -m app.manage create-admin --email you@example.com
python -m app.serve                                   # http://127.0.0.1:8821
```

Frontend (Node 22):

```bash
cd frontend
cp .env.example .env                                  # VITE_API_BASE_URL=http://127.0.0.1:8821/api
npm ci
npm run dev                                           # http://localhost:5180
```

Log in with the admin account you created. There is no self-signup. Client
portal accounts are created by an admin in License Management.

## Checks

```bash
cd backend && ruff check . && python -m pytest -q
cd frontend && npm run lint && npm run build
```

CI runs the same checks, plus a guard that fails if runtime data, databases
or secrets are ever committed.

## Data and privacy

`backend/data/` holds the SQLite database, face captures, enrollment photos,
training images, Re-ID/alert snapshots and the trained classifier. That's
biometric personal data. It's git-ignored, never committed, and cleaned up by
the retention job (DEPLOYMENT.md section 9). Back it up with the procedure in
DEPLOYMENT.md, not with git.

## Known issues

- Windows Smart App Control can block compiled Python packages
  (scikit-learn, scipy) with "DLL load failed ... Application Control
  policy". Retry or reinstall the package. This doesn't affect Linux
  servers.
- CP Plus NVRs lock an account after repeated failed logins. The camera
  reader backs off (2 s up to 60 s) instead of retrying in a loop.
- Don't use `uvicorn --reload` with cameras streaming. Use
  `python -m app.serve`.
