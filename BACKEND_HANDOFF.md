# Deco Vision — handoff for backend work

Written for whoever (human or Claude session) picks up backend work on this repo next. Everything below reflects the actual state of the code as of this session — verified by reading the source, not by memory.

## Tech stack

**Frontend** — `frontend/`
- React 19 + Vite 8, React Router 7
- Tailwind CSS 4 (via `@tailwindcss/postcss`), plain CSS utility classes in `src/index.css` (`.card`, `.btn-primary`, `.input-field`, `.badge`, etc.)
- `lucide-react` for icons, `recharts` for charts
- `oxlint` as the linter (`npm run lint`)
- Dev server fixed to **port 5180** (not the Vite default 5173 — an unrelated project on this machine uses 5173)

**Backend** — `backend/`
- FastAPI + Uvicorn (`uvicorn app.main:app --reload`), runs on **127.0.0.1:8821**
- `opencv-python` (`cv2.VideoCapture`) to pull frames off RTSP cameras
- SQLite via the stdlib `sqlite3` module — no ORM, no migration tool. Schema is created ad hoc with `CREATE TABLE IF NOT EXISTS` in `camera_db.py`. DB file: `backend/data/app.db`.
- `python-dotenv` loads `backend/.env` (`HOST`, `PORT`, `CORS_ORIGINS`, `LIVE_STREAM_FPS`)

No auth library, no JWT, no ORM, no task queue — the backend is intentionally minimal right now (4 files: `main.py`, `camera_db.py`, `camera_stream.py`, `config.py`).

## What's actually wired to a real backend vs. what's mock

This is the single most important thing to know before touching backend code: **`frontend/src/api/client.js` is the only file the frontend talks to a backend through.** Every page calls a function from that file instead of importing mock data directly. Right now:

**Real** (hits this repo's FastAPI backend):
- `getCameras`, `addCamera`, `getSites`, `addSite` → `/api/cameras`, `/api/sites`
- `login` → `POST /api/auth/login` (see caveat below)
- The live video feed: `/ws/live/{camera_id}` websocket, opened directly from `LiveCameraTile`/`CameraViewerModal` via `src/hooks/useLiveCameraFeed.js`
- `getPeople()` (People page, Employee tab) → fetches a **separate, already-deployed** face-enrollment service at `http://13.61.58.14/api/faces` directly from the browser (see below)

**Still mock** (in-memory React state or `src/data/mockData.js`, nothing persists, nothing hits any backend): Alerts, Attendance, Workforce, Footfall, Intrusion, Dashboard's non-camera stats, Settings, and the People page's "Incoming Guests"/"Validated" tabs plus all Add/Edit/Delete person actions. Adding or editing a person, marking an alert resolved, etc. only changes local component state — refreshing the page loses it.

If you're building real endpoints for any of these, the pattern to follow is already there: open `api/client.js`, find the stub function (each one has the real `request(...)` call commented out above the mock `Promise.resolve(...)` line), uncomment it, delete the mock line.

### About that external face-enrollment service (`13.61.58.14`)

This is **not part of this repo** — it's a separately deployed instance (different backend, different DB, has fields like `admin_port`/`ai_enabled` on cameras that our local schema doesn't have). It happens to expose a public, CORS-open (`Access-Control-Allow-Origin: *`) `GET /api/faces` endpoint returning each enrolled person's name, `employee_id`, and reference photo URLs, which the People page now fetches directly from the browser as read-only data, with a fallback to mock data if that host is unreachable.

That same host also exposes `/api/users` (login accounts, including a bcrypt password hash for one account) and a POST-only `/api/people`. **Do not wire either of those into this app** — `/api/users` is out of scope/sensitive, and nothing in this repo currently posts to that service. If real backend work eventually needs to *own* people/enrollment data, treat this external host as a reference for the data shape only, not a dependency to build against.

### Auth caveat

`POST /api/auth/login` in this repo's backend does **not check a password** — it just records whichever email was submitted and returns a fixed `token: "local-session"`. There is no user table, no session store, no JWT in this repo. If real auth is in scope, this needs to be built from scratch here.

## Camera data model (this repo's backend)

SQLite `cameras` table (`backend/app/camera_db.py`): `id, name, site, cam_code, purpose, host, port, user, password, stream_path, vendor, status, live_feed_enabled`.

- `password` is stored **in plaintext**, and is stripped out of every response sent to the frontend (`_row_to_dict` pops it) — only `get_camera_connection()` (internal use, building the RTSP URL) returns it. Flag this if backend security work is in scope.
- `status` is auto-derived: `"active"` if `host` is set, else `"inactive"`.
- RTSP URL is built in `camera_stream.py`: `rtsp://{user}:{password}@{host}:{port}{stream_path}`.
- Live viewing: one background thread per camera (`CameraStream` in `camera_stream.py`) opens the RTSP stream once via OpenCV and fans JPEG frames out to every websocket subscriber watching that camera — multiple browser tabs share one RTSP connection.
- `/ws/detections/{camera_id}` is a **placeholder** — it always sends `{"faces": [], "fire_smoke": []}` on a 1s timer. No detection model is wired up. If one gets built, the frontend's `CameraViewerModal` would need bounding-box overlay UI added on top of the canvas (not built yet, since there's nothing to draw).

### Cameras currently configured (4, all verified live)

| Name | Host:Port | Path |
|---|---|---|
| Main Section 1 | 144.79.198.46:91 | /video/live?channel=1&subtype=0 |
| Main Section 2 | 144.79.198.46:90 | /video/live?channel=1&subtype=0 |
| Lift Area | 144.79.198.46:92 | /video/live?channel=1&subtype=0 |
| Tech Section | 144.79.198.46:93 | /video/live?channel=1&subtype=0 |

All: `user=admin`, `password=Admin@123`, `vendor=cp_plus`. **Heads up:** these are CP Plus NVR channels that appear to temporarily lock the account (401 Unauthorized) after repeated failed-password connection attempts — this happened once this session (self-cleared after a few minutes). Don't hammer them with retries if auth starts failing; back off instead.

## Frontend structure

- `src/pages/*` — one file per route
- `src/layouts/` — `AppShell` (page chrome), `Sidebar` (now white/light theme, was dark), `Topbar`
- `src/components/` — shared UI: `DataTable`, `Modal`, `SidePanel` (slide-in from the right, used for row-detail drill-ins), `StatusBadge`, `Avatar` (initials-on-color-circle, no real images by default), `PageHeader`, `LiveCameraTile`, `CameraViewerModal`, `FaceEnrollment` (webcam capture + multi-file upload, shared by add/edit person flows)
- `src/hooks/useLiveCameraFeed.js` — the websocket→canvas streaming logic, shared between the grid tile and the enlarged viewer modal
- `src/data/*` — mock data, split into `mockData.js` (shared shapes) and `*Extra.js` files (page-specific richer mock detail, keyed by array position against `mockData.js`)

## Recent frontend work this session (no backend changes beyond the cleanup below)

- Configured the 4 RTSP cameras above with working credentials.
- Rebuilt the People page to match a Figma reference: tabs, filters, a two-step Add/Edit person wizard, real webcam capture + multi-photo upload/gallery/delete per person, and wired the Employee tab to the real external faces API (40 real people with real employee IDs and photos, replacing 4 fake mock rows).
- Live Camera: clicking a tile now opens an enlarged modal viewer (title, close, LIVE badge, PLAYING(HD) + live clock) reusing the same stream.
- Sidebar changed from dark to white theme.

## Dead code removed this session

- Backend: unused `import time` in `main.py`.
- Frontend `api/client.js`: four API stub functions with zero callers anywhere in the app — `acknowledgeAlert`, `addPerson` (the old stub; person add/edit is currently local-state-only), `markLeave`, `getZoneAccessList`.
- `mockData.js`: `intrusionAccessList`, which was only ever read by the now-removed `getZoneAccessList`.
- Unused `ShieldAlert` import in `Dashboard.jsx`, unused `catch (err)` binding in `Login.jsx`.

Verified clean after removal: `ruff check --select F` on the backend, `oxlint` + `vite build` on the frontend — no unused-code warnings remain beyond a handful of intentionally-unused stub parameters in `api/client.js` (kept because those functions *are* called from pages with real arguments; the params just aren't used yet inside the still-mocked function bodies).

## If you're about to build real backend endpoints

1. Check `api/client.js` first — the commented-out `request(...)` line above each mock is the intended contract (path, method, body shape) the frontend already expects.
2. `backend/app/main.py` currently only has camera/site CRUD, `/api/stats` (camera counts only), `/api/alerts` (always returns `[]`), `/api/settings` (in-memory dict, resets on restart), and the two websockets. Everything else needs new routes.
3. There's no ORM — decide early whether to keep raw `sqlite3` (matches `camera_db.py`'s style) or introduce something heavier; nothing in the current code assumes either way.
