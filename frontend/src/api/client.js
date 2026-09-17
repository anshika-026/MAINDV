// ---------------------------------------------------------------------------
// API CLIENT
// This is the ONE file you need to edit to connect your backend.
// Every page imports its data through the functions below instead of
// importing mockData.js directly, so swapping mock -> real API is a
// one-line change per function (remove the mock line, uncomment the fetch).
//
// Wired to the real wellmont FastAPI backend: auth, cameras, sites, and the
// camera-count portion of the dashboard. Everything else (alerts, people,
// attendance, workforce, footfall, intrusion, settings) has no backend yet
// and still reads mockData — there's no detection/analytics pipeline behind
// this app to serve real data for those.
// ---------------------------------------------------------------------------
import * as mock from "../data/mockData";

// Set this in a .env file as VITE_API_BASE_URL=https://your-api.example.com/api
export const BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

// Derived for the live-view websocket (camera_stream.py) — same host as
// BASE_URL, minus the /api suffix and http(s) swapped for ws(s).
const WS_ROOT = BASE_URL.replace(/\/api\/?$/, "");
export const WS_HOST = WS_ROOT.replace(/^https?:\/\//, "");
export const WS_PROTOCOL = WS_ROOT.startsWith("https") ? "wss" : "ws";

// Called wherever a request comes back 401/403 while this browser THOUGHT
// it had a valid session (deco_token was set) — that combination means
// the session was revoked/expired server-side, or (for a client login)
// the underlying license was just suspended/deactivated/expired (see
// backend/app/auth.py's load_active_client_license, re-checked on every
// request). Clearing storage and bouncing to the right login page is the
// visible side of "access stops immediately", not just a 403 toast on
// whatever page happened to be open. Does nothing for a failed LOGIN
// attempt itself (no token was set yet), so it never interferes with the
// error message a login form needs to show.
function handleAuthFailure() {
  const hadToken = !!localStorage.getItem("deco_token");
  if (!hadToken) return;
  let wasClient = false;
  try {
    wasClient = JSON.parse(localStorage.getItem("deco_user") || "null")?.role === "client";
  } catch {
    // corrupt localStorage — fall through with wasClient=false
  }
  localStorage.removeItem("deco_token");
  localStorage.removeItem("deco_user");
  const path = window.location.pathname;
  if (!path.startsWith("/login") && !path.startsWith("/client-login") && !path.startsWith("/client/")) {
    window.location.href = wasClient ? "/client-login" : "/login";
  }
}

function authHeaders() {
  const token = localStorage.getItem("deco_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(options.headers || {}),
    },
  });
  if (res.status === 401 || res.status === 403) handleAuthFailure();
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

// ---- Auth ------------------------------------------------------------
export async function login(email, _password) {
  // Backend doesn't verify a password yet (no real auth) — it just records
  // who signed in, same as the previous frontend's login. What IS real
  // now: the token it returns is a genuine server-side session (see
  // backend/app/auth.py) required on every subsequent admin request —
  // previously this was a hardcoded client-side string nothing ever checked.
  const data = await request("/auth/login", { method: "POST", body: JSON.stringify({ email }) });
  return { token: data.token, user: { ...mock.currentUser, name: data.name, email: data.email } };
}
export async function logoutAdmin() {
  try {
    await request("/auth/logout", { method: "POST" });
  } catch {
    // best-effort — the local session is cleared either way by AuthContext.logout()
  }
}

export async function signup(payload) {
  // No backend signup endpoint (no accounts/roles system built yet) — still mocked.
  return Promise.resolve({ token: "demo-token", user: { ...mock.currentUser, ...payload } });
}

// ---- Dashboard ---------------------------------------------------------
export async function getDashboardStats() {
  // Only the camera count is real; the rest (people/footfall/alerts/AI
  // insights) has no backend pipeline yet, so it stays mock.
  const cameras = await getCameras();
  const online = cameras.filter((c) => c.status === "Active").length;
  return {
    ...mock.dashboardStats,
    admin: {
      ...mock.dashboardStats.admin,
      camerasOnline: { value: `${online} / ${cameras.length}`, sub: "" },
    },
  };
}

// ---- Alerts & Events -----------------------------------------------------
export async function getAlerts() {
  // return request("/alerts");
  return Promise.resolve(mock.alerts);
}
export async function getAlertsSummary() {
  // return request("/alerts/summary");
  return Promise.resolve(mock.alertsSummary);
}
export async function resolveAlert(id, reason) {
  // return request(`/alerts/${id}/resolve`, { method: "POST", body: JSON.stringify({ reason }) });
  return Promise.resolve({ ok: true });
}
// ---- Cameras / Sites -----------------------------------------------------

// Builds a display-only rtsp:// URL from a camera's non-secret connection
// fields — mirrors backend/app/camera_stream.py's build_rtsp_url(). The
// backend never sends the real password to the frontend (see
// camera_db.py's _row_to_dict), so `password` here is always blank —
// editCamera()'s caller decides whether an edited URL's (also-blank)
// password means "unchanged" or "cleared", see parseRtspUrl below.
export function buildRtspUrl({ user, password, host, port, streamPath }) {
  if (!host) return "";
  const auth = user || password ? `${user || ""}:${password || ""}@` : "";
  const path = streamPath ? (streamPath.startsWith("/") ? streamPath : `/${streamPath}`) : "/";
  return `rtsp://${auth}${host}:${port || 554}${path}`;
}

// Inverse of the above. Deliberately splits on the LAST "@" before the
// first "/" — camera passwords here are known to contain "@" themselves
// (e.g. "Admin@123"), and a host/IP never does, so the last "@" is
// unambiguously the credentials/host boundary.
export function parseRtspUrl(raw) {
  const withoutProto = String(raw || "").replace(/^rtsp:\/\//i, "");
  const firstSlash = withoutProto.indexOf("/");
  const beforePath = firstSlash === -1 ? withoutProto : withoutProto.slice(0, firstSlash);
  const path = firstSlash === -1 ? "" : withoutProto.slice(firstSlash);
  const lastAt = beforePath.lastIndexOf("@");
  const credentials = lastAt === -1 ? "" : beforePath.slice(0, lastAt);
  const hostPort = lastAt === -1 ? beforePath : beforePath.slice(lastAt + 1);
  const colonIdx = credentials.indexOf(":");
  const user = colonIdx === -1 ? credentials : credentials.slice(0, colonIdx);
  const password = colonIdx === -1 ? "" : credentials.slice(colonIdx + 1);
  const [host, portStr] = hostPort.split(":");
  return { user, password, host: host || "", port: portStr ? Number(portStr) : 554, streamPath: path || "/" };
}

function mapCamera(c) {
  return {
    id: c.id,
    code: c.cam_code || `CAM-${c.id}`,
    label: c.name,
    site: c.site,
    purpose: c.purpose,
    status: c.status === "active" ? "Active" : "Inactive",
    live: c.live_feed_enabled ? "On" : "Off",
    isConfigured: c.is_configured,
    // Raw fields needed to reconstruct a (password-blank) stream URL and
    // populate the Edit Camera form — password itself is never sent here.
    host: c.host || "",
    port: c.port || 554,
    user: c.user || "",
    streamPath: c.stream_path || "",
    attendanceTracking: c.attendance_tracking !== 0,
  };
}

export async function getCameras() {
  const cameras = await request("/cameras");
  return cameras.map(mapCamera);
}
export async function addCamera(payload) {
  return request("/cameras", {
    method: "POST",
    body: JSON.stringify({
      name: payload.driveName,
      site: payload.site,
      cam_code: payload.code,
      purpose: payload.purpose || "GENERAL",
    }),
  });
}
export async function updateCamera(id, payload) {
  return request(`/cameras/${id}`, { method: "PUT", body: JSON.stringify(payload) });
}
export async function deleteCamera(id) {
  return request(`/cameras/${id}`, { method: "DELETE" });
}
export async function getSites() {
  const sites = await request("/sites");
  return sites.map((s) => ({
    id: s.id,
    name: s.name,
    description: s.description || "",
    cameras: s.cameras.length,
    cameraList: s.cameras.map((c) => ({ id: c.id, label: c.name })),
    wgs: "-",
    status: s.active_count > 0 ? "Active" : "Inactive",
  }));
}
export async function addSite(payload) {
  return request("/sites", { method: "POST", body: JSON.stringify({ name: payload.name }) });
}
export async function updateSite(id, payload) {
  return request(`/sites/${id}`, { method: "PUT", body: JSON.stringify(payload) });
}
export async function deleteSite(id) {
  return request(`/sites/${id}`, { method: "DELETE" });
}

// ---- People ----------------------------------------------------------
// Real enrolled-face data lives on the separately deployed face-enrollment
// service (not this repo's backend) — /api/faces returns each enrolled
// person's name, unique employee ID and reference sample photos.
const FACES_API_BASE = "http://13.61.58.14";

export async function getPeople() {
  try {
    const res = await fetch(`${FACES_API_BASE}/api/faces`);
    if (!res.ok) throw new Error(`Faces API error ${res.status}`);
    const rows = await res.json();
    // The external service has no way to save an edited employee ID (see
    // setPersonEmployeeId) — its own employee_id field is often null/stale.
    // Our own backend's override, keyed by this same `name`, wins whenever
    // one exists so an edit actually survives a refresh.
    let overrides = {};
    try {
      overrides = await facesRequest("/people-id-overrides");
    } catch {
      // Best-effort — People page still works with the external service's
      // own (possibly stale) IDs if this backend is briefly unreachable.
    }
    return rows.map((r) => ({
      name: r.name,
      employeeId: overrides[r.name] || r.employee_id || "-",
      designs: r.sample_count,
      faceEnrolled: r.sample_count > 0,
      enrollment: r.sample_count > 0 ? "Enrolled" : "Not enrolled",
      photos: (r.photo_urls || []).map((path, i) => ({ id: `${r.name}-${i}`, url: `${FACES_API_BASE}${path}` })),
    }));
  } catch {
    return mock.people;
  }
}
// Persists an edited employee ID for a person from the People page — see
// face_db.py's people_employee_id_overrides table comment for why this is
// needed (the external roster service has no save/update API of its own).
export function setPersonEmployeeId(name, employeeId) {
  return facesRequest("/people-id-overrides", {
    method: "POST",
    body: JSON.stringify({ name, employee_id: employeeId }),
  });
}
export async function getValidatedPeople() {
  // return request("/people/validated");
  return Promise.resolve(mock.validatedPeople);
}
// ---- Attendance --------------------------------------------------------
export async function getAttendance() {
  // return request("/attendance");
  return Promise.resolve(mock.attendance);
}
export async function getAttendanceStats() {
  // return request("/attendance/stats");
  return Promise.resolve(mock.attendanceStats);
}
// ---- Workforce -----------------------------------------------------------
export async function getWorkforceStats() {
  // return request("/workforce/stats");
  return Promise.resolve(mock.workforceStats);
}
export async function getWorkforcePeopleAnalytics() {
  // return request("/workforce/people-analytics");
  return Promise.resolve(mock.workforcePeopleAnalytics);
}
export async function getDeskAnalytics() {
  // return request("/workforce/desk-analytics");
  return Promise.resolve(mock.deskAnalytics);
}

// ---- Footfall ------------------------------------------------------------
export async function getFootfallStats() {
  // return request("/footfall/stats");
  return Promise.resolve(mock.footfallStats);
}
export async function getFootfallVisitors() {
  // return request("/footfall/visitors");
  return Promise.resolve(mock.footfallVisitors);
}

// ---- Intrusion -----------------------------------------------------------
export async function getIntrusionZones() {
  // return request("/intrusion/zones");
  return Promise.resolve(mock.intrusionZones);
}
export async function addZone(payload) {
  // return request("/intrusion/zones", { method: "POST", body: JSON.stringify(payload) });
  return Promise.resolve({ ok: true });
}

// ---- Settings ------------------------------------------------------------
// No backend endpoint exists for user profiles yet, so this persists to the
// same "deco_user" localStorage key AuthContext already writes on login —
// otherwise an edit here would appear to save (toast + updated UI) but
// silently revert on the next page load/navigation, since getProfile() would
// keep returning the pristine mock object. AuthContext.updateUser() keeps
// the Topbar/sidebar in sync with these edits within the same session.
export async function getProfile() {
  // return request("/me");
  try {
    const saved = localStorage.getItem("deco_user");
    if (saved) return Promise.resolve({ ...mock.currentUser, ...JSON.parse(saved) });
  } catch {
    // corrupt/unavailable localStorage — fall through to the default profile
  }
  return Promise.resolve(mock.currentUser);
}
export async function updateProfile(payload) {
  // return request("/me", { method: "PATCH", body: JSON.stringify(payload) });
  try {
    const saved = localStorage.getItem("deco_user");
    const merged = { ...(saved ? JSON.parse(saved) : mock.currentUser), ...payload };
    localStorage.setItem("deco_user", JSON.stringify(merged));
  } catch {
    // localStorage unavailable (e.g. private browsing) — edit still applies
    // for the rest of this session via React state, just won't survive a reload
  }
  return Promise.resolve({ ok: true });
}

// ---- Face training (manual labeling + classifier training) ---------------
// Real backend, unlike most of this file — see backend/FACE_TRAINING.md.
// Errors surface FastAPI's `detail` message directly (e.g. "Unknown
// employee_id '999'") instead of request()'s generic wrapped text, since
// the labeling page shows this string straight to the person typing IDs.
async function trainingRequest(path, options = {}) {
  const res = await fetch(`${BASE_URL}/faces/training${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...authHeaders(), ...(options.headers || {}) },
  });
  if (res.status === 401 || res.status === 403) handleAuthFailure();
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

// Same shape as trainingRequest but for the (non-training) /api/faces/*
// routes in face_routes.py, e.g. people-id-overrides.
async function facesRequest(path, options = {}) {
  const res = await fetch(`${BASE_URL}/faces${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...authHeaders(), ...(options.headers || {}) },
  });
  if (res.status === 401 || res.status === 403) handleAuthFailure();
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export function getNextTrainingCapture() {
  return trainingRequest("/next");
}
export function getTrainingStats() {
  return trainingRequest("/stats");
}
export function getTrainingEmployees() {
  return trainingRequest("/employees");
}
// Best-effort: the labeling page calls this before loading the roster so an
// ID/name edited on the external face-enrollment service shows up here
// without anyone having to remember to sync manually. Never blocks the page
// if that external service is briefly unreachable.
export function syncTrainingEmployees() {
  return trainingRequest("/employees/sync", { method: "POST" });
}
// Deliberately NOT a plain URL for an <img src> — /faces/training/image/{id}
// requires an admin session (see face_training_routes.py's router-level
// auth), and a browser <img> request can't attach an Authorization header
// the way fetch() can (the same reason the camera websocket takes its token
// as a query param instead). Fetching the bytes ourselves and handing back
// an object URL keeps the endpoint under normal Bearer-token auth instead of
// putting a token in a URL (which can leak via referrers/logs).
export async function fetchTrainingImageObjectUrl(captureId) {
  const res = await fetch(`${BASE_URL}/faces/training/image/${captureId}`, {
    headers: { ...authHeaders() },
  });
  if (res.status === 401 || res.status === 403) handleAuthFailure();
  if (!res.ok) throw new Error(`Could not load image (${res.status})`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}
export function labelTrainingCapture(captureId, employeeId) {
  return trainingRequest("/label", {
    method: "POST",
    body: JSON.stringify({ capture_id: captureId, employee_id: employeeId }),
  });
}
export function skipTrainingCapture(captureId) {
  return trainingRequest("/skip", { method: "POST", body: JSON.stringify({ capture_id: captureId }) });
}
// Corrects an already-labeled capture's employee ID (e.g. a mistyped ID) —
// unlike labelTrainingCapture, works on a capture that isn't 'unlabeled'.
export function relabelTrainingCapture(captureId, employeeId) {
  return trainingRequest("/relabel", {
    method: "POST",
    body: JSON.stringify({ capture_id: captureId, employee_id: employeeId }),
  });
}
// Plain undo: sends a labeled capture back into the unlabeled queue.
export function unlabelTrainingCapture(captureId) {
  return trainingRequest("/unlabel", { method: "POST", body: JSON.stringify({ capture_id: captureId }) });
}
export function getRecentTrainingLabels(limit = 8) {
  return trainingRequest(`/recent-labels?limit=${limit}`);
}

// ---- License management ---------------------------------------------
// Real backend (backend/app/license_routes.py) — companies, licenses,
// per-license camera assignment, and license/camera feature toggles. Uses
// its own error-surfacing helper (like trainingRequest above) since the
// UI shows FastAPI's `detail` message directly (e.g. a max_cameras cap
// error, or an unknown feature key).
async function licenseRequest(path, options = {}) {
  const res = await fetch(`${BASE_URL}/licenses${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...authHeaders(), ...(options.headers || {}) },
  });
  // Not for /client-login itself — a failed login attempt has no token
  // set yet, so handleAuthFailure() is a no-op there (see its own comment).
  if (res.status === 401 || res.status === 403) handleAuthFailure();
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export function getLicenseAnalytics() {
  return licenseRequest("/analytics");
}
export function getLicenseFeatureCatalog() {
  return licenseRequest("/features");
}
export function getCompanies() {
  return licenseRequest("/companies");
}
export function createCompany(name) {
  return licenseRequest("/companies", { method: "POST", body: JSON.stringify({ name }) });
}
export function getLicenses(params = {}) {
  const qs = new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== ""))
  ).toString();
  return licenseRequest(`${qs ? `?${qs}` : ""}`);
}
export function createLicense(payload) {
  return licenseRequest("", { method: "POST", body: JSON.stringify(payload) });
}
export function updateLicense(id, payload) {
  return licenseRequest(`/${id}`, { method: "PUT", body: JSON.stringify(payload) });
}
export function setLicenseCredentials(id, username, password) {
  return licenseRequest(`/${id}/credentials`, { method: "PUT", body: JSON.stringify({ username, password }) });
}
// Client-portal login (separate from the admin login at /login) — a
// license's username/password, so a client can sign in from any browser
// instead of a device-bound key/QR. See AuthContext.loginAsClient().
export function clientLogin(username, password) {
  return licenseRequest("/client-login", { method: "POST", body: JSON.stringify({ username, password }) });
}
export async function clientLogout() {
  try {
    await licenseRequest("/client-logout", { method: "POST" });
  } catch {
    // best-effort — the local session is cleared either way by AuthContext.logout()
  }
}
// Re-checks (and refreshes) an already-logged-in client session — see
// AuthContext's periodic refresh, which is how a suspended/edited license
// reaches an already-open client tab without them having to log out first.
export function clientMe() {
  return licenseRequest("/client-me");
}
// Public — no login required. Used by the dev-mode per-client portal
// route (/client/:slug/login) to show which company's portal this is,
// the local equivalent of what a real client-<slug>.decovision.com
// subdomain would reveal before any credentials are entered. The slug
// itself never grants access.
export function getCompanyBySlug(slug) {
  return licenseRequest(`/companies/slug/${encodeURIComponent(slug)}`);
}
export function deleteLicense(id) {
  return licenseRequest(`/${id}`, { method: "DELETE" });
}
export function setLicenseStatus(id, status) {
  return licenseRequest(`/${id}/status`, { method: "POST", body: JSON.stringify({ status }) });
}
export function getLicenseCameras(id) {
  return licenseRequest(`/${id}/cameras`);
}
export function assignLicenseCameras(id, cameraIds) {
  return licenseRequest(`/${id}/cameras`, { method: "POST", body: JSON.stringify({ camera_ids: cameraIds }) });
}
export function unassignLicenseCameras(id, cameraIds) {
  return licenseRequest(`/${id}/cameras`, { method: "DELETE", body: JSON.stringify({ camera_ids: cameraIds }) });
}
export function setLicenseFeatures(id, featureKeys) {
  return licenseRequest(`/${id}/features`, { method: "PUT", body: JSON.stringify({ feature_keys: featureKeys }) });
}
export function setCameraFeatures(id, cameraId, featureKeys) {
  return licenseRequest(`/${id}/cameras/${cameraId}/features`, {
    method: "PUT",
    body: JSON.stringify({ feature_keys: featureKeys }),
  });
}
export function licenseQrUrl(id) {
  return `${BASE_URL}/licenses/${id}/qr`;
}
