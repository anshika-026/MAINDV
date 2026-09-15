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

async function request(path, options = {}) {
  const token = localStorage.getItem("deco_token");
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

// ---- Auth ------------------------------------------------------------
export async function login(email, _password) {
  // Backend doesn't verify a password yet (no real auth) — it just records
  // who signed in, same as the previous frontend's login.
  const data = await request("/auth/login", { method: "POST", body: JSON.stringify({ email }) });
  return { token: "local-session", user: { ...mock.currentUser, name: data.name, email: data.email } };
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
export async function getSites() {
  const sites = await request("/sites");
  return sites.map((s) => ({
    id: s.id,
    name: s.name,
    cameras: s.cameras.length,
    wgs: "-",
    status: s.active_count > 0 ? "Active" : "Inactive",
  }));
}
export async function addSite(payload) {
  return request("/sites", { method: "POST", body: JSON.stringify({ name: payload.name }) });
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
    return rows.map((r) => ({
      name: r.name,
      employeeId: r.employee_id || "-",
      designs: r.sample_count,
      faceEnrolled: r.sample_count > 0,
      enrollment: r.sample_count > 0 ? "Enrolled" : "Not enrolled",
      photos: (r.photo_urls || []).map((path, i) => ({ id: `${r.name}-${i}`, url: `${FACES_API_BASE}${path}` })),
    }));
  } catch {
    return mock.people;
  }
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
export async function getProfile() {
  // return request("/me");
  return Promise.resolve(mock.currentUser);
}
export async function updateProfile(payload) {
  // return request("/me", { method: "PATCH", body: JSON.stringify(payload) });
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
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
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
export function trainingImageUrl(captureId) {
  return `${BASE_URL}/faces/training/image/${captureId}`;
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
