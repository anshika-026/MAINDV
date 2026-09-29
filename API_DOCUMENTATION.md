# API Documentation

Generated from the running application by `backend/scripts/gen_api_docs.py`.
Do not edit by hand; regenerate after changing routes.

## Conventions

- Base path: `/api` (served behind nginx on the same origin as the web app).
- Authentication: `Authorization: Bearer <token>`. Tokens come from
  `POST /api/auth/login` (admin: email and password) or
  `POST /api/licenses/client-login` (client portal: license username and
  password). Tokens are opaque, stored server-side, expire after
  `SESSION_TTL_HOURS` (default 12), and are revoked on logout, password
  change, admin disable, and license suspension, credential reset or deletion.
- Access levels:
  - **Public**: no token.
  - **Admin**: an admin session. A client session gets `403`.
  - **Admin or client**: any session. Client results are limited to the
    cameras assigned to their license.
  - **Admin, or client with the licensed feature**: a client also needs the
    feature enabled on their license, otherwise `403`.
- Errors are JSON `{"detail": "..."}`. Common codes: `401` missing, invalid
  or expired token; `403` authenticated but not allowed; `404` not found;
  `413` upload too large; `415` not an accepted image type; `422`
  validation failed; `429` too many failed logins (see `Retry-After`);
  `500` internal error (details only in the server log); `502`/`503` an
  external service (Identity) is unreachable or not configured.
- Uploads: JPEG, PNG or WebP only, at most `MAX_UPLOAD_MB` (default 5).

## Example

```bash
TOKEN=$(curl -s -X POST https://vision.example.com/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"ops@example.com","password":"..."}' | jq -r .token)
curl -s https://vision.example.com/api/cameras -H "Authorization: Bearer $TOKEN"
```


## Health

#### `GET /health`

- **Access:** Public
- **Status codes:** 200

#### `GET /ready`

- **Access:** Public
- **Status codes:** 200

#### `GET /api/health/details`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422


## Authentication (admin)

#### `POST /api/auth/login`

- **Access:** Public
- **Description:** Admin login: email + password checked against admin_users (auth.authenticate_admin). Failures are throttled per IP and per email (ratelimit.py) and always get the same generic message, whether the email exists or not.
- **Request body (application/json):** `email` (string, required, minLength 1, maxLength 254), `password` (string, required, minLength 1, maxLength 1024)
- **Status codes:** 200, 422

#### `GET /api/auth/me`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/auth/logout`

- **Access:** Admin or client (client limited to their cameras)
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 422


## Licensing and client portal

#### `GET /api/licenses/companies`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/licenses/companies`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `name` (string, required)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/licenses/companies/slug/{slug}`

- **Access:** Public
- **Description:** Deliberately PUBLIC (no auth) — this is the dev-mode equivalent of visiting client-<slug>.decovision.com and seeing which company's portal you're at, same as a real subdomain would reveal before any login form is even submitted. Returns only the display name; never cameras, features, or anything else. The slug never grants access — /client-login below still requires the real username/password.
- **Parameters:** `slug` (path, required)
- **Status codes:** 200, 422

#### `GET /api/licenses/features`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/licenses/analytics`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/licenses`

- **Access:** Admin
- **Parameters:** `company_id` (query), `status` (query), `search` (query), `limit` (query), `offset` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/licenses`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `company_id` (string, required), `max_cameras` (integer), `label` (string), `feature_keys` (array), `username` (string, required), `password` (string, required), `expiry_date` (string|null)
- **Status codes:** 200, 401, 403, 422

#### `PUT /api/licenses/{license_id}/credentials`

- **Access:** Admin
- **Description:** Admin resets a client's portal username/password (e.g. they forgot it, or it needs rotating) — there's no self-service "forgot password" flow since there's no email-sending infrastructure in this app.
- **Parameters:** `license_id` (path, required), `authorization` (header)
- **Request body (application/json):** `username` (string, required), `password` (string, required)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/licenses/client-login`

- **Access:** Public
- **Description:** Client-portal login — separate from the existing admin /api/auth/login. A license's username/password lets that client sign in from any browser/device, unlike a device-bound QR/key. On success, issues a real server-side session token (auth.create_client_session) — every subsequent request re-validates this license's status/expiry fresh from the DB (see auth.load_active_client_license), so a suspen
- **Request body (application/json):** `username` (string, required, minLength 1, maxLength 128), `password` (string, required, minLength 1, maxLength 1024)
- **Status codes:** 200, 422

#### `POST /api/licenses/client-logout`

- **Access:** Admin or client (client limited to their cameras)
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 422

#### `GET /api/licenses/client-me`

- **Access:** Client session
- **Description:** Lets the frontend re-check (and refresh cameras/features for) an already-logged-in client session — polled periodically by AuthContext so a license change made by an admin (suspended, cameras/features changed) reaches an already-open client tab without requiring them to log out and back in.
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 422

#### `GET /api/licenses/{license_id}`

- **Access:** Admin
- **Parameters:** `license_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `PUT /api/licenses/{license_id}`

- **Access:** Admin
- **Parameters:** `license_id` (path, required), `authorization` (header)
- **Request body (application/json):** `label` (string|null), `max_cameras` (integer|null), `expiry_date` (string|null)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/licenses/{license_id}/status`

- **Access:** Admin
- **Parameters:** `license_id` (path, required), `authorization` (header)
- **Request body (application/json):** `status` (string, required)
- **Status codes:** 200, 401, 403, 422

#### `DELETE /api/licenses/{license_id}`

- **Access:** Admin
- **Parameters:** `license_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/licenses/{license_id}/qr`

- **Access:** Admin
- **Parameters:** `license_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/licenses/{license_id}/cameras`

- **Access:** Admin
- **Parameters:** `license_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/licenses/{license_id}/cameras`

- **Access:** Admin
- **Parameters:** `license_id` (path, required), `authorization` (header)
- **Request body (application/json):** `camera_ids` (array, required)
- **Status codes:** 200, 401, 403, 422

#### `DELETE /api/licenses/{license_id}/cameras`

- **Access:** Admin
- **Parameters:** `license_id` (path, required), `authorization` (header)
- **Request body (application/json):** `camera_ids` (array, required)
- **Status codes:** 200, 401, 403, 422

#### `PUT /api/licenses/{license_id}/features`

- **Access:** Admin
- **Parameters:** `license_id` (path, required), `authorization` (header)
- **Request body (application/json):** `feature_keys` (array, required)
- **Status codes:** 200, 401, 403, 422

#### `PUT /api/licenses/{license_id}/cameras/{camera_id}/features`

- **Access:** Admin
- **Parameters:** `license_id` (path, required), `camera_id` (path, required), `authorization` (header)
- **Request body (application/json):** `feature_keys` (array, required)
- **Status codes:** 200, 401, 403, 422


## Cameras and sites

#### `GET /api/cameras/{camera_id}/frame`

- **Access:** Admin
- **Description:** Latest still from any camera. If it isn't streaming yet, starts it for a few seconds to get one.
- **Parameters:** `camera_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/cameras/health`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/cameras`

- **Access:** Admin or client (client limited to their cameras)
- **Description:** Tenant isolation lives here, not in the React UI: an admin principal sees every camera (unchanged), but a client principal only ever sees the cameras their own license has been assigned — enforced fresh against the DB on every call, not from anything the browser claims about itself. Previously this endpoint had no authentication at all and returned every camera to anyone who called it.
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 422

#### `POST /api/cameras`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `name` (string, required), `site` (string, required), `cam_code` (string), `purpose` (string), `host` (string), `port` (integer), `user` (string), `password` (string|null), `stream_path` (string), `vendor` (string), `live_feed_enabled` (boolean), `attendance_tracking` (boolean)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/cameras/test-stream`

- **Access:** Admin
- **Description:** Connects to an RTSP link once and returns a single still (JPEG), so the Add Camera form can show the link works before it's saved. Runs in FastAPI's threadpool (plain def), so a slow camera never blocks others.
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `rtsp_url` (string, required)
- **Status codes:** 200, 401, 403, 422

#### `PUT /api/cameras/{camera_id}`

- **Access:** Admin
- **Parameters:** `camera_id` (path, required), `authorization` (header)
- **Request body (application/json):** `name` (string|null), `site` (string|null), `cam_code` (string|null), `purpose` (string|null), `host` (string|null), `port` (integer|null), `user` (string|null), `password` (string|null), `stream_path` (string|null), `vendor` (string|null), `live_feed_enabled` (boolean|null), `attendance_tracking` (boolean|null)
- **Status codes:** 200, 401, 403, 422

#### `DELETE /api/cameras/{camera_id}`

- **Access:** Admin
- **Parameters:** `camera_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/sites`

- **Access:** Admin or client (client limited to their cameras)
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 422

#### `POST /api/sites`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `name` (string, required), `description` (string)
- **Status codes:** 200, 401, 403, 422

#### `PUT /api/sites/{site_id}`

- **Access:** Admin
- **Parameters:** `site_id` (path, required), `authorization` (header)
- **Request body (application/json):** `name` (string|null), `description` (string|null)
- **Status codes:** 200, 401, 403, 422

#### `DELETE /api/sites/{site_id}`

- **Access:** Admin
- **Parameters:** `site_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422


## Face training

#### `GET /api/faces/training/next`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/training/stats`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/training/image/{capture_id}`

- **Access:** Admin
- **Parameters:** `capture_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/training/label`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `capture_id` (integer, required), `employee_id` (string, required)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/training/skip`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `capture_id` (integer, required)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/training/relabel`

- **Access:** Admin
- **Description:** Corrects the employee_id on a capture that was already labeled or skipped — the fix for 'I typed the wrong ID'. Unlike /label, this accepts a capture that isn't currently 'unlabeled'.
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `capture_id` (integer, required), `employee_id` (string, required)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/training/unlabel`

- **Access:** Admin
- **Description:** Plain undo: sends a labeled/skipped capture back into the unlabeled queue. No employee_id guessed or assumed — the next /next call will serve it again for a human to label properly.
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `capture_id` (integer, required)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/training/recent-labels`

- **Access:** Admin
- **Description:** Powers the labeling page's 'just labeled' correction list — recent labeled captures only (not skipped), newest first.
- **Parameters:** `limit` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/training/employees`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/training/employees`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `employee_id` (string, required), `name` (string, required)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/training/employees/sync`

- **Access:** Admin
- **Description:** Pull from the external face-enrollment service to refresh the local roster — the only place that host is ever touched by the training pipeline. Called explicitly (this route) and automatically whenever the /face-training page loads (see FaceTraining.jsx), so an ID or name edited on that external service shows up here without a manual step.
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/training/train`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/training/model-status`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/training/auto-train/status`

- **Access:** Admin
- **Description:** Background periodic-retraining scheduler state — see face_training_scheduler.py. Also folded into /collection/status for a single combined monitoring call.
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/training/auto-train/toggle`

- **Access:** Admin
- **Description:** Pause/resume automatic retraining without a restart. Manual POST /train is unaffected either way — this only controls the background trigger.
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `enabled` (boolean, required)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/training/training-history`

- **Access:** Admin
- **Description:** Newest-first record of every completed training run (see face_training.train_classifier -> face_db.add_training_run) — real measured numbers only, nothing estimated. Powers the 'Face Model Training' history panel in FaceTraining.jsx.
- **Parameters:** `limit` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/training/collection/start`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `camera_ids` (array|null), `days` (number|null)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/training/collection/stop`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `camera_id` (integer|null)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/training/collection/status`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422


## Face management (review queue, enrollment, Identity)

#### `GET /api/faces/pending`

- **Access:** Admin
- **Parameters:** `hours` (query), `status` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/assign`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `pending_id` (integer, required), `person_id` (string, required)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/ignore`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `pending_id` (integer, required)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/enroll`

- **Access:** Admin
- **Description:** Direct enrollment path (e.g. from the People page's existing photo upload UI) — bypasses the review queue since the human is already confirming identity by uploading it against a specific person_id.
- **Parameters:** `authorization` (header)
- **Request body (multipart/form-data):** `person_id` (string, required), `photo` (string, required)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/gallery/{person_id}/count`

- **Access:** Admin
- **Parameters:** `person_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/behavior/analyze`

- **Access:** Admin
- **Description:** One webcam frame in, current detection state out.
- **Parameters:** `authorization` (header)
- **Request body (multipart/form-data):** `frame` (string, required)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/people`

- **Access:** Admin
- **Description:** Create or update one hand-entered person. Committed to the database before this returns, so a success response means the record is durable — the caller can treat a thrown error as "nothing was saved" and must not report success on its own.
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `employee_id` (string, required), `name` (string, required), `department` (string|null), `person_type` (string|null)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/people`

- **Access:** Admin
- **Description:** Every hand-entered person, read straight from the database — this is what makes them reappear after a refresh, a backend restart or a reboot, rather than living only in the page's React state.
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/people/photo/{embedding_id}`

- **Access:** Admin
- **Description:** Serves back a saved enrollment image so the Identity page can still show a person's face samples after a refresh. Reads the file written by POST /enroll; 404 rather than an error if that file is gone, since the embedding itself (the part recognition actually uses) is still valid.
- **Parameters:** `embedding_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/identity/roster`

- **Access:** Admin
- **Description:** The external roster, proxied: [{name, employee_id, sample_count, photo_urls: [path, ...]}]. Photos are then fetched through /identity/photo below, never directly by the browser.
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/identity/photo`

- **Access:** Admin
- **Parameters:** `path` (query, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/gallery/sync-from-identity`

- **Access:** Admin
- **Description:** Turns the people enrolled on the Identity page into local face embeddings, so they can be recognised live even when they have too few labelled camera captures to be one of the trained classifier's classes (see face_pipeline._identify_for_overlay, which consults this gallery whenever the classifier isn't confident).
- **Parameters:** `force` (query), `max_photos_per_person` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/faces/people-id-overrides`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/faces/people-id-overrides`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `name` (string, required), `employee_id` (string, required)
- **Status codes:** 200, 401, 403, 422


## Attendance, workforce, dashboard

#### `GET /api/attendance`

- **Access:** Admin
- **Description:** Every employee's attendance for `date` (YYYY-MM-DD, default today).
- **Parameters:** `date` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/attendance/{employee_id}/history`

- **Access:** Admin
- **Parameters:** `employee_id` (path, required), `days` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/attendance/leave`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `employee_id` (string, required), `day_from` (string, required), `day_to` (string, required), `reason` (string)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/dashboard/summary`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/workforce/overview`

- **Access:** Admin
- **Parameters:** `date` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422


## Desk analytics

#### `GET /api/desk-zones`

- **Access:** Admin
- **Parameters:** `camera_id` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/desk-zones`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `camera_id` (integer, required), `polygon` (array, required), `label` (string|null)
- **Status codes:** 200, 401, 403, 422

#### `DELETE /api/desk-zones/{zone_id}`

- **Access:** Admin
- **Parameters:** `zone_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/desk-analytics/report`

- **Access:** Admin
- **Parameters:** `date` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422


## Footfall

#### `GET /api/footfall/summary`

- **Access:** Admin
- **Description:** Today's unique people across every gate (each person once, whichever gate(s) they used), plus the per-gate breakdown, hourly arrivals vs yesterday, and today's visitor list.
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/footfall/people`

- **Access:** Admin
- **Description:** Every unique person currently counted, newest first, with snapshot ids.
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/footfall/snapshots/{snapshot_id}`

- **Access:** Admin
- **Parameters:** `snapshot_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/footfall/reset`

- **Access:** Admin
- **Description:** UAT: restart the unique count from zero. Deletes every Re-ID identity and snapshot; cameras, faces and attendance are untouched.
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/footfall/cameras/{camera_id}/frame`

- **Access:** Admin
- **Description:** Latest still from a gate camera, for drawing its counting zone on.
- **Parameters:** `camera_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `PUT /api/footfall/cameras/{camera_id}/zone`

- **Access:** Admin
- **Description:** Only people whose body centre is inside this zone are counted at this gate — draw it over the doorway to keep seating areas out.
- **Parameters:** `camera_id` (path, required), `authorization` (header)
- **Request body (application/json):** `roi` (array|null)
- **Status codes:** 200, 401, 403, 422


## Staff count

#### `GET /api/staff/count`

- **Access:** Admin, or client with the licensed feature (limited to their cameras)
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/staff/present`

- **Access:** Admin, or client with the licensed feature (limited to their cameras)
- **Description:** Everyone inside now (named employees and anonymous people), plus employees who were in earlier today and have left.
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/staff/events`

- **Access:** Admin, or client with the licensed feature (limited to their cameras)
- **Parameters:** `event_type` (query), `limit` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/staff/entries`

- **Access:** Admin, or client with the licensed feature (limited to their cameras)
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/staff/exits`

- **Access:** Admin, or client with the licensed feature (limited to their cameras)
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/staff/status`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/staff/cameras`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `PUT /api/staff/cameras/{camera_id}/config`

- **Access:** Admin
- **Parameters:** `camera_id` (path, required), `authorization` (header)
- **Request body (application/json):** `enabled` (boolean), `line` (array|null), `inside_sign` (integer), `roi` (array|null)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/staff/reset`

- **Access:** Admin
- **Description:** Mark everyone as having left (logged as AUTO_EXIT), e.g. before a test run.
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `reason` (string)
- **Status codes:** 200, 401, 403, 422


## Intrusion

#### `GET /api/intrusion/zones`

- **Access:** Admin
- **Parameters:** `camera_id` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/intrusion/zones`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `camera_id` (integer, required), `name` (string, required), `polygon` (array, required), `active_from` (string|null), `active_to` (string|null)
- **Status codes:** 200, 401, 403, 422

#### `PATCH /api/intrusion/zones/{zone_id}`

- **Access:** Admin
- **Parameters:** `zone_id` (path, required), `authorization` (header)
- **Request body (application/json):** `name` (string|null), `enabled` (boolean|null), `active_from` (string|null), `active_to` (string|null)
- **Status codes:** 200, 401, 403, 422

#### `DELETE /api/intrusion/zones/{zone_id}`

- **Access:** Admin
- **Parameters:** `zone_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/intrusion/stats`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422


## Alerts

#### `GET /api/alerts`

- **Access:** Admin
- **Parameters:** `range` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/alerts/summary`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/alerts/{alert_id}/snapshot`

- **Access:** Admin
- **Parameters:** `alert_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/alerts/{alert_id}/acknowledge`

- **Access:** Admin
- **Parameters:** `alert_id` (path, required), `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `POST /api/alerts/{alert_id}/resolve`

- **Access:** Admin
- **Parameters:** `alert_id` (path, required), `authorization` (header)
- **Request body (application/json):** `reason` (string, required)
- **Status codes:** 200, 401, 403, 422


## Analytics switches and settings

#### `GET /api/analytics/settings`

- **Access:** Admin or client (client limited to their cameras)
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 422

#### `PUT /api/analytics/settings/{feature}`

- **Access:** Admin
- **Parameters:** `feature` (path, required), `authorization` (header)
- **Request body (application/json):** `on` (boolean, required)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/stats`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `GET /api/settings`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Status codes:** 200, 401, 403, 422

#### `PUT /api/settings`

- **Access:** Admin
- **Parameters:** `authorization` (header)
- **Request body (application/json):** `detection_fps` (number, required, minimum 0.1, maximum 30.0)
- **Status codes:** 200, 401, 403, 422


## Audit log

#### `GET /api/audit`

- **Access:** Admin
- **Parameters:** `limit` (query), `authorization` (header)
- **Status codes:** 200, 401, 403, 422


## WebSockets

Browsers can't send an `Authorization` header on a WebSocket handshake, so these take the session token as `?token=`. The server re-checks the session every 30 s and closes with **4401** (not authenticated, or the session ended) or **4403** (not allowed: feed switched off, or an admin-only stream).

#### `WS /ws/staff`

- Admin, or client with the attendance feature (counts limited to their cameras). JSON staff count every 2 s.

#### `WS /ws/staff/debug/{camera_id}`

- Admin only. Raw tracker state for one entrance camera, 4 times a second.

#### `WS /ws/live/{camera_id}`

- Admin, or client whose license includes the camera. Binary JPEG frames. Query: `token`, `plain=1` (no overlay detection), `w` (scale width, 320-3840).

#### `WS /ws/detections/{camera_id}`

- Same access as `/ws/live`. JSON `{people: [{track_id, bbox, employee_id, name, color, confidence, identity_source, expression, expression_confidence}], fire_smoke: [], frame_w, frame_h}` about 3 times a second.
