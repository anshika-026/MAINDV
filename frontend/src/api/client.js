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
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

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
export async function acknowledgeAlert(id) {
  // return request(`/alerts/${id}/acknowledge`, { method: "POST" });
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
export async function getPeople() {
  // return request("/people");
  return Promise.resolve(mock.people);
}
export async function getValidatedPeople() {
  // return request("/people/validated");
  return Promise.resolve(mock.validatedPeople);
}
export async function addPerson(payload) {
  // return request("/people", { method: "POST", body: JSON.stringify(payload) });
  return Promise.resolve({ ok: true });
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
export async function markLeave(employeeId, payload) {
  // return request(`/attendance/${employeeId}/leave`, { method: "POST", body: JSON.stringify(payload) });
  return Promise.resolve({ ok: true });
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
export async function getZoneAccessList(zoneName) {
  // return request(`/intrusion/zones/${encodeURIComponent(zoneName)}/access`);
  return Promise.resolve(mock.intrusionAccessList);
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
