// ---------------------------------------------------------------------------
// API CLIENT
// Every page reads and writes backend data through the functions below.
// There is no silent mock fallback: if the backend fails, the caller gets an
// error and the page shows it, instead of plausible-looking fake data.
// ---------------------------------------------------------------------------
import * as mock from "../data/mockData";

// Base URL of the backend API. In production this is normally the relative
// "/api" (nginx serves the SPA and proxies /api and /ws to the backend on the
// same origin), set via frontend/.env.production. A missing value is a build
// configuration error, not something to guess around.
export const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
if (!BASE_URL) {
  // eslint-disable-next-line no-console
  console.error("VITE_API_BASE_URL is not set; API calls will fail. See frontend/.env.example.");
}

// Development-only sample data (People page "validated guests", default
// profile fields). Off unless explicitly enabled with VITE_DEMO_MODE=true.
export const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "true";

// Websocket host/protocol: same origin as the API. A relative BASE_URL
// ("/api") resolves against the page's own host, and https pages get wss.
const API_URL = new URL(BASE_URL || "/api", window.location.origin);
export const WS_HOST = API_URL.host;
export const WS_PROTOCOL = API_URL.protocol === "https:" ? "wss" : "ws";

// Called when a request comes back 401 while this browser THOUGHT it had a
// valid session (deco_token was set): the session expired or was revoked
// server-side (logout elsewhere, password change, license suspended). A 403
// is different — the session is fine, this one action just isn't allowed
// (e.g. a feature the license doesn't include) — so it never logs you out.
// Does nothing for a failed LOGIN attempt (no token set yet), so it never
// interferes with the error message a login form needs to show.
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
  if (res.status === 401) handleAuthFailure();
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const err = new Error(body?.detail || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// ---- Auth ------------------------------------------------------------
// Admin login: email + password, verified server-side (backend/app/auth.py).
// Throws with the backend's message ("Invalid email or password", or the
// rate-limit message after too many failures).
export async function login(email, password) {
  const res = await fetch(`${BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : "Couldn't log in. Check your email and password.";
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  const data = await res.json();
  return { token: data.token, user: { role: "admin", name: data.name, email: data.email } };
}
export async function logoutAdmin() {
  try {
    await request("/auth/logout", { method: "POST" });
  } catch {
    // best-effort — the local session is cleared either way by AuthContext.logout()
  }
}

// ---- Dashboard ---------------------------------------------------------
// Real: /dashboard/summary (backend/app/insights.py) + footfall. Guests and
// Guests have no backend yet, so that card says so instead of showing
// sample numbers.
export async function getDashboardStats() {
  const [summary, footfall, alertSummary, staff] = await Promise.all([
    request("/dashboard/summary"),
    getFootfallSummary().catch(() => null),
    getAlertsSummary().catch(() => null),
    request("/staff/count").catch(() => null),
  ]);
  const gates = footfall?.gates.length ?? 0;
  const pp = summary.people_present;
  const diff = pp.value - pp.yesterday;
  return {
    admin: {
      peoplePresent: { value: pp.value, of: pp.of, sub: `${diff >= 0 ? "+" : ""}${diff} vs yesterday` },
      footfallToday: footfall
        ? {
            value: footfall.unique_today,
            sub: gates ? `across ${gates} gate${gates === 1 ? "" : "s"} · avg ${footfall.avg_per_gate}/gate` : "no gate cameras set",
          }
        : { value: "—", sub: "footfall unavailable" },
      unknownVisitors: { value: summary.unknown_faces, sub: "faces not recognised today" },
      camerasOnline: {
        value: `${summary.cameras.online} / ${summary.cameras.total}`,
        sub: summary.cameras.offline ? `${summary.cameras.offline} offline` : "",
      },
      currentStaff: staff
        ? { value: staff.current_staff_count, sub: `${staff.total_entries_today} in · ${staff.total_exits_today} out today` }
        : { value: "—", sub: "staff count unavailable" },
      activeAlerts: alertSummary
        ? { value: alertSummary.active, sub: alertSummary.acknowledged ? `${alertSummary.acknowledged} acknowledged` : "" }
        : { value: "—", sub: "alerts unavailable" },
    },
    needsAttention: summary.needs_attention,
    aiInsights: summary.insights,
  };
}

// ---- Alerts & Events -----------------------------------------------------
// Real alerts (backend/app/alerts.py): unknown people at gates, cameras
// offline, footfall stopped, late arrivals. range: today | yesterday | week | all
function mapAlert(a) {
  const d = new Date(a.ts * 1000);
  return {
    id: a.id,
    ts: a.ts,
    date: d.toLocaleDateString(undefined, { day: "numeric", month: "short" }),
    time: d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    event: a.event,
    severity: a.severity,
    status: a.status,
    camera: a.camera_name,
    location: a.location,
    message: a.message,
    confidence: a.confidence,
    occurrences: a.occurrences,
    lastSeen: new Date(a.last_seen * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    hasSnapshot: a.has_snapshot,
    acknowledgedBy: a.acknowledged_by,
    resolvedBy: a.resolved_by,
    resolutionReason: a.resolution_reason,
  };
}
export async function getAlerts(range = "today") {
  return (await request(`/alerts?range=${range}`)).map(mapAlert);
}
// On/off switch per analytics feature + live CPU (backend/app/analytics_settings.py).
export async function getAnalyticsSettings() {
  return request("/analytics/settings");
}
export async function setAnalyticsFeature(feature, on) {
  return request(`/analytics/settings/${feature}`, { method: "PUT", body: JSON.stringify({ on }) });
}
export async function getAlertsSummary() {
  return request("/alerts/summary");
}
export async function acknowledgeAlert(id) {
  return request(`/alerts/${id}/acknowledge`, { method: "POST" });
}
export async function resolveAlert(id, reason) {
  return request(`/alerts/${id}/resolve`, { method: "POST", body: JSON.stringify({ reason }) });
}
export async function fetchAlertSnapshotObjectUrl(id) {
  const res = await fetch(`${BASE_URL}/alerts/${id}/snapshot`, { headers: { ...authHeaders() } });
  if (res.status === 401) handleAuthFailure();
  if (!res.ok) throw new Error("No snapshot");
  return URL.createObjectURL(await res.blob());
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
    feedOn: !!c.live,
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
  // payload.streamUrl is a full rtsp:// link; split into the fields the
  // backend stores (password included — it's only ever sent, never read back).
  const s = payload.streamUrl ? parseRtspUrl(payload.streamUrl) : null;
  return request("/cameras", {
    method: "POST",
    body: JSON.stringify({
      name: payload.driveName,
      site: payload.site,
      cam_code: payload.code || "",
      purpose: payload.purpose || "GENERAL",
      ...(s && { host: s.host, port: s.port, user: s.user, password: s.password, stream_path: s.streamPath }),
      attendance_tracking: payload.attendanceTracking ?? true,
      live_feed_enabled: true,
    }),
  });
}
// Connects to an RTSP link once and returns an object URL of one still, or
// throws with the backend's explanation (bad address, login, stream path).
export async function testCameraStream(rtspUrl) {
  const res = await fetch(`${BASE_URL}/cameras/test-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ rtsp_url: rtspUrl }),
  });
  if (res.status === 401) handleAuthFailure();
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.detail || `Couldn't reach the camera (${res.status})`);
  }
  return URL.createObjectURL(await res.blob());
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
// ("Identity") service. The browser never calls it directly: the backend
// proxies it (/faces/identity/*, admin auth, configured IDENTITY_SERVICE_BASE),
// which keeps this page working over HTTPS and keeps that service private.
async function fetchIdentityRosterPhotoObjectUrl(path) {
  const res = await fetch(`${BASE_URL}/faces/identity/photo?path=${encodeURIComponent(path)}`, {
    headers: { ...authHeaders() },
  });
  if (res.status === 401) handleAuthFailure();
  if (!res.ok) throw new Error(`Could not load photo (${res.status})`);
  return URL.createObjectURL(await res.blob());
}

// ---- Manually enrolled Identity people ----------------------------------
// People added by hand on the Identity page are saved in OUR backend's
// database (face_db's employees table, manually_added=1) rather than the
// external roster service, which has no write API. These four functions are
// the whole persistence path: save the record, upload each face photo
// (stored on disk + embedded into face_embeddings by /faces/enroll), read
// them back after a refresh, and fetch a saved photo for display.

export function saveIdentityPerson({ employeeId, name, department, personType }) {
  return facesRequest("/people", {
    method: "POST",
    body: JSON.stringify({
      employee_id: employeeId,
      name,
      department: department || null,
      person_type: personType || null,
    }),
  });
}

export function getIdentityPeople() {
  return facesRequest("/people");
}

// Behavior Analytics: posts one laptop-webcam frame and gets back what the
// EXISTING detectors saw in it (person/face counts from the same YOLO models
// the live cameras use, plus the existing expression service). Nothing about
// the RTSP camera pipeline is involved. Same multipart shape as
// enrollFacePhoto — no Content-Type header, so the browser sets the boundary.
export async function analyzeBehaviorFrame(blob) {
  const body = new FormData();
  body.append("frame", blob, "frame.jpg");
  const res = await fetch(`${BASE_URL}/faces/behavior/analyze`, {
    method: "POST",
    body,
    headers: { ...authHeaders() },
  });
  if (res.status === 401) handleAuthFailure();
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.detail || `Analyze failed (${res.status})`);
  }
  return res.json();
}

// multipart/form-data — deliberately does NOT set Content-Type, so the
// browser adds the multipart boundary itself. Surfaces the backend's real
// `detail` (e.g. "No face detected in photo") so the UI can show why an
// upload failed instead of reporting a success it didn't get.
export async function enrollFacePhoto(employeeId, blob, filename = "face.jpg") {
  const body = new FormData();
  body.append("person_id", employeeId);
  body.append("photo", blob, filename);
  const res = await fetch(`${BASE_URL}/faces/enroll`, {
    method: "POST",
    body,
    headers: { ...authHeaders() },
  });
  if (res.status === 401) handleAuthFailure();
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.detail || `Photo upload failed (${res.status})`);
  }
  return res.json();
}

// Same reasoning as fetchTrainingImageObjectUrl — this endpoint needs an
// admin Bearer token, which a plain <img src> cannot send.
export async function fetchIdentityPhotoObjectUrl(embeddingId) {
  const res = await fetch(`${BASE_URL}/faces/people/photo/${embeddingId}`, {
    headers: { ...authHeaders() },
  });
  if (res.status === 401) handleAuthFailure();
  if (!res.ok) throw new Error(`Could not load photo (${res.status})`);
  return URL.createObjectURL(await res.blob());
}

// Reads the hand-entered people back out of our database and shapes them
// like the external roster rows, so getPeople() can present one merged
// list. Photo object URLs are resolved here so a saved face sample still
// renders after a refresh.
async function getLocalIdentityRows() {
  const people = await getIdentityPeople();
  return Promise.all(
    people.map(async (p) => ({
      name: p.name,
      employeeId: p.employee_id,
      department: p.department || undefined,
      type: p.person_type || "Employee",
      designs: p.embedding_count,
      faceEnrolled: p.embedding_count > 0,
      enrollment: p.embedding_count > 0 ? "Enrolled" : "Not enrolled",
      manuallyAdded: true,
      photos: await Promise.all(
        (p.photos || []).map(async (ph) => ({
          id: `local-${ph.id}`,
          url: await fetchIdentityPhotoObjectUrl(ph.id).catch(() => null),
        }))
      ),
    }))
  );
}

export async function getPeople() {
  // Hand-entered people come from our own database and must show up even
  // if the external roster service is unreachable — they're independent
  // sources, so a failure of one must not hide the other. If BOTH fail the
  // error propagates so the page can say so (never fake people).
  let localError = null;
  const localRows = await getLocalIdentityRows().catch((e) => {
    localError = e;
    return [];
  });
  try {
    const rows = await facesRequest("/identity/roster");
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
    const externalRows = await Promise.all(
      rows.map(async (r) => ({
        name: r.name,
        employeeId: overrides[r.name] || r.employee_id || "-",
        designs: r.sample_count,
        faceEnrolled: r.sample_count > 0,
        enrollment: r.sample_count > 0 ? "Enrolled" : "Not enrolled",
        photos: await Promise.all(
          (r.photo_urls || []).slice(0, 5).map(async (path, i) => ({
            id: `${r.name}-${i}`,
            url: await fetchIdentityRosterPhotoObjectUrl(path).catch(() => null),
          }))
        ),
      }))
    );
    // Merge by employee ID rather than dropping either side. When both
    // sources describe the same ID it's the same person — someone filled
    // in details locally for somebody the roster service also knows — so
    // the hand-entered values win (they were typed deliberately, and the
    // external service has no write API to push them back to), while the
    // roster's reference photos are kept if the local record has none.
    // Discarding the local row here instead is exactly what made a saved
    // person look like it "didn't save": the record was in the database
    // but never reached the list.
    const byExternalId = new Map(externalRows.map((r) => [r.employeeId, r]));
    const mergedLocal = localRows.map((local) => {
      const external = byExternalId.get(local.employeeId);
      if (!external) return local;
      const photos = local.photos?.length ? local.photos : external.photos;
      return {
        ...external,
        ...local,
        photos,
        designs: local.designs || external.designs,
        enrollment: (local.designs || external.designs) > 0 ? "Enrolled" : "Not enrolled",
      };
    });
    // Locally-saved people first — a just-added person should be visible
    // without scrolling.
    const localIds = new Set(localRows.map((r) => r.employeeId));
    return [...mergedLocal, ...externalRows.filter((r) => !localIds.has(r.employeeId))];
  } catch (e) {
    // External roster unreachable — still show what we have saved locally.
    if (localRows.length || !localError) return localRows;
    throw e;
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
  // No backend for validated guests exists yet. Sample rows only in an
  // explicit demo build; otherwise an honest empty list.
  return DEMO_MODE ? mock.validatedPeople : [];
}
// ---- Attendance --------------------------------------------------------
// Marked from face recognition on cameras with it switched on
// (backend/app/attendance.py). `date` is YYYY-MM-DD.
export async function getAttendanceDay(date) {
  return request(`/attendance?date=${encodeURIComponent(date)}`);
}
export async function getAttendanceHistory(employeeId, days = 14) {
  return request(`/attendance/${encodeURIComponent(employeeId)}/history?days=${days}`);
}
export async function markLeave({ employeeId, from, to, reason }) {
  return request("/attendance/leave", {
    method: "POST",
    body: JSON.stringify({ employee_id: employeeId, day_from: from, day_to: to, reason }),
  });
}
// ---- Workforce -----------------------------------------------------------
// Real: attendance, trend and face-enrolment split (backend/app/insights.py).
export async function getWorkforceOverview(date) {
  return request(`/workforce/overview${date ? `?date=${date}` : ""}`);
}
// ---- Desk analytics (real: backend/app/desks.py) --------------------------
export async function listDeskZones(cameraId) {
  return request(`/desk-zones${cameraId ? `?camera_id=${cameraId}` : ""}`);
}
export async function createDeskZone(cameraId, polygon, label) {
  return request("/desk-zones", { method: "POST", body: JSON.stringify({ camera_id: cameraId, polygon, label: label || null }) });
}
export async function deleteDeskZone(zoneId) {
  return request(`/desk-zones/${zoneId}`, { method: "DELETE" });
}
export async function getDeskReport(date) {
  return request(`/desk-analytics/report${date ? `?date=${date}` : ""}`);
}
// Latest still from any camera (starts it briefly if it isn't streaming).
export async function fetchCameraFrameObjectUrl(cameraId) {
  const res = await fetch(`${BASE_URL}/cameras/${cameraId}/frame`, { headers: { ...authHeaders() } });
  if (res.status === 401) handleAuthFailure();
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.detail || `No picture from this camera (${res.status})`);
  }
  return URL.createObjectURL(await res.blob());
}

// ---- Footfall ------------------------------------------------------------
// Unique people across every entry gate — each person counted once no
// matter which gate(s) they used (backend/app/footfall.py). Gate cameras
// are the ones with purpose "Entry/Exit" in Camera Management.
export async function getFootfallSummary() {
  return request("/footfall/summary");
}
// UAT panel: every unique person currently counted, and a one-click restart.
export async function getFootfallPeople() {
  return request("/footfall/people");
}
export async function resetFootfall() {
  return request("/footfall/reset", { method: "POST" });
}
// Counting zone per gate: roi is [[x, y], ...] in 0..1 frame fractions, or
// null to count the whole frame.
export async function setFootfallZone(cameraId, roi) {
  return request(`/footfall/cameras/${cameraId}/zone`, { method: "PUT", body: JSON.stringify({ roi }) });
}
export async function fetchFootfallFrameObjectUrl(cameraId) {
  const res = await fetch(`${BASE_URL}/footfall/cameras/${cameraId}/frame`, { headers: { ...authHeaders() } });
  if (res.status === 401) handleAuthFailure();
  if (!res.ok) throw new Error(`No frame from this camera yet (${res.status})`);
  return URL.createObjectURL(await res.blob());
}
// Needs the admin Bearer token, which a plain <img src> cannot send — same
// reasoning as fetchTrainingImageObjectUrl.
export async function fetchFootfallSnapshotObjectUrl(snapshotId) {
  const res = await fetch(`${BASE_URL}/footfall/snapshots/${snapshotId}`, { headers: { ...authHeaders() } });
  if (res.status === 401) handleAuthFailure();
  if (!res.ok) throw new Error(`Could not load snapshot (${res.status})`);
  return URL.createObjectURL(await res.blob());
}

// ---- Intrusion -----------------------------------------------------------
// Real restricted-area zones (backend/app/intrusion.py). Detections arrive as
// "Intrusion Detected" alerts.
export async function getIntrusionZones(cameraId) {
  return request(`/intrusion/zones${cameraId ? `?camera_id=${cameraId}` : ""}`);
}
export async function createIntrusionZone({ cameraId, name, polygon, from, to }) {
  return request("/intrusion/zones", {
    method: "POST",
    body: JSON.stringify({ camera_id: cameraId, name, polygon, active_from: from || null, active_to: to || null }),
  });
}
export async function updateIntrusionZone(id, fields) {
  return request(`/intrusion/zones/${id}`, { method: "PATCH", body: JSON.stringify(fields) });
}
export async function deleteIntrusionZone(id) {
  return request(`/intrusion/zones/${id}`, { method: "DELETE" });
}
export async function getIntrusionStats() {
  return request("/intrusion/stats");
}

// ---- Object detection (backend/app/object_detection/) ----------------------
// Latest backpack/handbag/bottle/laptop detections per camera.
export async function getObjectDetections() {
  return request("/objects/latest");
}

// ---- Staff Count (backend/app/staff/) -------------------------------------
// Occupancy from entry/exit line crossings at entrance cameras; the count is
// the backend's occupancy state, never the number of people in a frame.
export async function getStaffCount() {
  return request("/staff/count");
}
export async function getStaffPresent() {
  return request("/staff/present");
}
export async function getStaffEvents(limit = 100) {
  return request(`/staff/events?limit=${limit}`);
}
export async function getStaffCameras() {
  return request("/staff/cameras");
}
export async function getStaffStatus() {
  return request("/staff/status");
}
export async function setStaffCameraConfig(cameraId, { enabled, line, insideSign, roi }) {
  return request(`/staff/cameras/${cameraId}/config`, {
    method: "PUT",
    body: JSON.stringify({ enabled, line, inside_sign: insideSign, roi: roi && roi.length ? roi : null }),
  });
}
export async function resetStaffOccupancy() {
  return request("/staff/reset", { method: "POST", body: JSON.stringify({ reason: "manual reset" }) });
}
export function staffSocketUrl(path) {
  const token = localStorage.getItem("deco_token") || "";
  return `${WS_PROTOCOL}://${WS_HOST}${path}?token=${encodeURIComponent(token)}`;
}

// ---- Settings ------------------------------------------------------------
// No backend endpoint exists for user profiles yet, so this persists to the
// same "deco_user" localStorage key AuthContext already writes on login —
// otherwise an edit here would appear to save (toast + updated UI) but
// silently revert on the next page load/navigation, since getProfile() would
// keep returning the pristine mock object. AuthContext.updateUser() keeps
// the Topbar/sidebar in sync with these edits within the same session.
// Profile display fields only (name, avatar, phone) — kept in this browser.
// Identity and permissions always come from the server session, never from
// these fields.
const EMPTY_PROFILE = { name: "", email: "", phone: "", role: "" };
export async function getProfile() {
  const base = DEMO_MODE ? mock.currentUser : EMPTY_PROFILE;
  try {
    const saved = localStorage.getItem("deco_user");
    if (saved) return { ...base, ...JSON.parse(saved) };
  } catch {
    // corrupt/unavailable localStorage — fall through to the default profile
  }
  return base;
}
export async function updateProfile(payload) {
  try {
    const saved = localStorage.getItem("deco_user");
    const merged = { ...(saved ? JSON.parse(saved) : EMPTY_PROFILE), ...payload };
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
  if (res.status === 401) handleAuthFailure();
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
  if (res.status === 401) handleAuthFailure();
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export function getNextTrainingCapture() {
  return trainingRequest("/next");
}
export function getTrainingEmployees() {
  return trainingRequest("/employees");
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
  if (res.status === 401) handleAuthFailure();
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
export function getTrainingHistory(limit = 10) {
  return trainingRequest(`/training-history?limit=${limit}`);
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
  if (res.status === 401) handleAuthFailure();
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
export async function fetchLicenseQrObjectUrl(id) {
  const res = await fetch(`${BASE_URL}/licenses/${encodeURIComponent(id)}/qr`, { headers: { ...authHeaders() } });
  if (res.status === 401) handleAuthFailure();
  if (!res.ok) throw new Error(`Could not load QR code (${res.status})`);
  return URL.createObjectURL(await res.blob());
}
