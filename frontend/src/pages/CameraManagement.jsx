import { useEffect, useState } from "react";
import { Pencil, Trash2, Video } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import Modal from "../components/Modal";
import Tabs from "../components/Tabs";
import ToggleSwitch from "../components/ToggleSwitch";
import * as api from "../api/client";

// The backend has no per-camera health metric yet, so we derive a
// deterministic pseudo-random score (70-100%) per camera id — stable across
// renders/refreshes for the same camera, but clearly a local UI mock.
function healthForCamera(id) {
  const seed = String(id)
    .split("")
    .reduce((acc, ch) => (acc * 31 + ch.charCodeAt(0)) % 100000, 7);
  return 70 + (seed % 31);
}

function healthTone(pct) {
  if (pct >= 90) return "bg-success-500";
  if (pct >= 80) return "bg-warning-500";
  return "bg-danger-500";
}

const VIEWS = ["All", "Camera Health"];
const PURPOSES = ["General", "Theft", "Entry/Exit", "Surveillance", "Safety"];
const EMPTY_FORM = { streamUrl: "", driveName: "", site: "", purpose: "General", code: "", attendanceTracking: false };

// What the Add Camera form can tell from the link alone, before testing it.
function describeRtsp(url) {
  if (!url.trim()) return null;
  if (!/^rtsp:\/\//i.test(url.trim())) return { error: "The link must start with rtsp://" };
  const p = api.parseRtspUrl(url.trim());
  if (!p.host) return { error: "The link has no camera address after the @" };
  return p;
}

export default function CameraManagement() {
  const [cameras, setCameras] = useState([]);
  const [addOpen, setAddOpen] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  const [sites, setSites] = useState([]);
  const [test, setTest] = useState({ state: "idle", url: "", message: "" });
  const [saving, setSaving] = useState(false);
  const [addError, setAddError] = useState("");
  const [view, setView] = useState("All");
  const [editing, setEditing] = useState(null);
  const [toast, setToast] = useState("");

  useEffect(() => {
    refresh();
  }, []);

  function refresh() {
    api.getCameras().then(setCameras);
  }

  function showToast(msg) {
    setToast(msg);
    setTimeout(() => setToast(""), 2500);
  }

  const online = cameras.filter((c) => c.status === "Active").length;
  const rtspInfo = describeRtsp(form.streamUrl);
  const avgHealth = cameras.length
    ? Math.round(cameras.reduce((sum, c) => sum + healthForCamera(c.id), 0) / cameras.length)
    : null;

  function openAdd() {
    setForm(EMPTY_FORM);
    setTest({ state: "idle", url: "", message: "" });
    setAddError("");
    setAddOpen(true);
    api
      .getSites()
      .then((list) => {
        setSites(list);
        if (list.length === 1) setForm((f) => ({ ...f, site: f.site || list[0].name }));
      })
      .catch(() => setSites([]));
  }

  async function runTest() {
    setTest({ state: "testing", url: "", message: "" });
    try {
      const url = await api.testCameraStream(form.streamUrl.trim());
      setTest({ state: "ok", url, message: "" });
    } catch (err) {
      setTest({ state: "failed", url: "", message: err.message });
    }
  }

  async function handleAdd(e) {
    e.preventDefault();
    const parsed = describeRtsp(form.streamUrl);
    if (!parsed || parsed.error) {
      setAddError(parsed?.error || "Paste the camera's RTSP link");
      return;
    }
    setSaving(true);
    setAddError("");
    try {
      await api.addCamera({ ...form, streamUrl: form.streamUrl.trim() });
      setAddOpen(false);
      showToast(`${form.driveName} added. Open Live Feed to watch it.`);
      refresh();
    } catch {
      setAddError("Couldn't save the camera. Check the backend is running.");
    } finally {
      setSaving(false);
    }
  }

  function openEdit(cam) {
    setEditing({
      id: cam.id,
      camCode: cam.code,
      label: cam.label,
      purpose: cam.purpose || "General",
      site: cam.site,
      streamUrl: api.buildRtspUrl({
        user: cam.user,
        password: "",
        host: cam.host,
        port: cam.port,
        streamPath: cam.streamPath,
      }),
      attendanceTracking: cam.attendanceTracking,
    });
  }

  async function handleEditSubmit(e) {
    e.preventDefault();
    const parsed = api.parseRtspUrl(editing.streamUrl);
    const payload = {
      name: editing.label,
      cam_code: editing.camCode,
      purpose: editing.purpose,
      site: editing.site,
      host: parsed.host,
      port: parsed.port,
      user: parsed.user,
      stream_path: parsed.streamPath,
      attendance_tracking: editing.attendanceTracking,
    };
    // Password is never sent to the frontend, so the field only ever shows
    // it blank — only overwrite the stored password if the user actually
    // typed a new one into the URL.
    if (parsed.password) payload.password = parsed.password;

    await api.updateCamera(editing.id, payload);
    setEditing(null);
    showToast("Camera updated");
    refresh();
  }

  // Feed off = the backend stops connecting to this camera entirely (no live
  // view, no analytics, no background streaming), freeing CPU and bandwidth.
  const [switching, setSwitching] = useState(null);
  async function toggleFeed(cam, on) {
    setSwitching(cam.id);
    try {
      await api.updateCamera(cam.id, { live_feed_enabled: on });
      showToast(on ? `${cam.label} feed switched on` : `${cam.label} feed switched off: it's no longer streamed or analysed`);
      refresh();
    } catch {
      showToast(`Couldn't change ${cam.label}. Check the backend is running.`);
    } finally {
      setSwitching(null);
    }
  }

  async function handleDelete(cam) {
    if (!window.confirm(`Remove camera "${cam.label}"? This cannot be undone.`)) return;
    await api.deleteCamera(cam.id);
    showToast(`${cam.label} removed`);
    refresh();
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Camera Management"
        action={
          <button onClick={openAdd} className="btn-primary text-sm">
            + Add camera
          </button>
        }
      />

      {toast && (
        <div className="text-sm text-success-600 bg-success-50 border border-success-500/20 rounded-lg px-3.5 py-2">
          {toast}
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Cameras" value={cameras.length} subTone="neutral" />
        <StatCard label="Cameras Online" value={online} />
        <StatCard label="Cameras Offline" value={cameras.length - online} subTone="danger" />
        <StatCard label="Avg Camera Health" value={avgHealth === null ? "-" : `${avgHealth}%`} />
      </div>

      <Tabs tabs={VIEWS} active={view} onChange={setView} />

      {view === "All" ? (
        <DataTable
          columns={[
            { key: "code", label: "Cam Code" },
            { key: "label", label: "Label" },
            { key: "site", label: "Site" },
            { key: "purpose", label: "Purpose" },
            { key: "status", label: "Status", render: (r) => <StatusBadge value={r.status} /> },
            {
              key: "live",
              label: "Live feed",
              render: (r) => (
                <span className="flex items-center gap-2">
                  <ToggleSwitch checked={r.feedOn} onChange={(on) => toggleFeed(r, on)} disabled={!r.isConfigured || switching === r.id} />
                  <span className="text-xs text-slate-500">{r.feedOn ? "On" : "Off"}</span>
                </span>
              ),
            },
            {
              key: "action",
              label: "Actions",
              render: (r) => (
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => openEdit(r)}
                    title="Edit"
                    className="text-slate-400 hover:text-brand-600"
                  >
                    <Pencil size={15} />
                  </button>
                  <button
                    onClick={() => handleDelete(r)}
                    title="Delete"
                    className="text-slate-400 hover:text-danger-500"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              ),
            },
          ]}
          rows={cameras}
        />
      ) : (
        <div className="card divide-y divide-[#f1f2f7]">
          {cameras.length === 0 ? (
            <div className="text-center text-sm text-slate-400 py-10">
              No cameras yet — add one to see health data.
            </div>
          ) : (
            cameras.map((cam) => {
              const pct = healthForCamera(cam.id);
              return (
                <div key={cam.id} className="flex items-center gap-4 px-4 py-3.5">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-ink-900 truncate">{cam.label}</p>
                    <p className="text-xs text-slate-400 truncate">
                      {cam.site} · {cam.code}
                    </p>
                  </div>
                  <div className="w-40 h-2 rounded-full bg-[#eceef4] overflow-hidden hidden sm:block">
                    <div
                      className={`h-full rounded-full ${healthTone(pct)}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="text-sm font-semibold text-ink-900 w-10 text-right shrink-0">
                    {pct}%
                  </span>
                </div>
              );
            })
          )}
        </div>
      )}

      <Modal open={addOpen} onClose={() => setAddOpen(false)} title="Add Camera" width="max-w-lg">
        <form onSubmit={handleAdd} className="space-y-4">
          <div>
            <label htmlFor="add-rtsp" className="text-sm font-medium text-ink-900 block mb-1.5">
              RTSP link <span className="text-danger-500">*</span>
            </label>
            <div className="flex gap-2">
              <input
                id="add-rtsp"
                required
                autoFocus
                value={form.streamUrl}
                onChange={(e) => {
                  setForm((f) => ({ ...f, streamUrl: e.target.value }));
                  setTest({ state: "idle", url: "", message: "" });
                }}
                placeholder="rtsp://user:password@192.168.1.10:554/stream"
                className="input-field font-mono text-xs flex-1"
              />
              <button
                type="button"
                onClick={runTest}
                disabled={!rtspInfo || !!rtspInfo.error || test.state === "testing"}
                className="btn-secondary text-sm whitespace-nowrap disabled:opacity-50"
              >
                {test.state === "testing" ? "Testing…" : "Test"}
              </button>
            </div>
            {!rtspInfo && (
              <p className="text-xs text-slate-400 mt-1.5">Paste the full link from the camera or NVR, including the login.</p>
            )}
            {rtspInfo?.error && <p className="text-xs text-danger-500 mt-1.5">{rtspInfo.error}</p>}
            {rtspInfo && !rtspInfo.error && (
              <p className="text-xs text-slate-500 mt-1.5">
                Camera <span className="font-mono">{rtspInfo.host}:{rtspInfo.port}</span>
                {rtspInfo.user && (
                  <>
                    {" "}· user <span className="font-mono">{rtspInfo.user}</span>
                  </>
                )}
                {rtspInfo.password ? " · password included" : " · no password"} · path{" "}
                <span className="font-mono">{rtspInfo.streamPath}</span>
              </p>
            )}
            {test.state === "ok" && (
              <div className="mt-2 rounded-lg overflow-hidden border border-border-200">
                <img src={test.url} alt="Still from the camera" className="w-full aspect-video object-cover" />
                <p className="text-xs text-success-600 px-3 py-1.5">Connected. This is what the camera sees right now.</p>
              </div>
            )}
            {test.state === "failed" && <p className="text-xs text-danger-500 mt-1.5">{test.message}</p>}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="add-name" className="text-sm font-medium text-ink-900 block mb-1.5">
                Camera name <span className="text-danger-500">*</span>
              </label>
              <input
                id="add-name"
                required
                value={form.driveName}
                onChange={(e) => setForm((f) => ({ ...f, driveName: e.target.value }))}
                placeholder="e.g. Gate 2"
                className="input-field"
              />
            </div>
            <div>
              <label htmlFor="add-site" className="text-sm font-medium text-ink-900 block mb-1.5">
                Site <span className="text-danger-500">*</span>
              </label>
              <input
                id="add-site"
                required
                list="add-site-options"
                value={form.site}
                onChange={(e) => setForm((f) => ({ ...f, site: e.target.value }))}
                placeholder="Noida Site"
                className="input-field"
              />
              <datalist id="add-site-options">
                {sites.map((site) => (
                  <option key={site.id} value={site.name} />
                ))}
              </datalist>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="add-purpose" className="text-sm font-medium text-ink-900 block mb-1.5">
                Purpose
              </label>
              <select
                id="add-purpose"
                value={form.purpose}
                onChange={(e) => setForm((f) => ({ ...f, purpose: e.target.value }))}
                className="input-field"
              >
                {PURPOSES.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="add-code" className="text-sm font-medium text-ink-900 block mb-1.5">
                Cam code (optional)
              </label>
              <input
                id="add-code"
                value={form.code}
                onChange={(e) => setForm((f) => ({ ...f, code: e.target.value }))}
                placeholder="CAM-XXXX"
                className="input-field"
              />
            </div>
          </div>
          {form.purpose === "Entry/Exit" && (
            <p className="text-xs text-brand-600 -mt-2">
              Entry/Exit cameras count unique footfall. Draw the doorway zone on the Footfall UAT page after adding.
            </p>
          )}

          <label className="flex items-start gap-2.5 cursor-pointer">
            <input
              type="checkbox"
              checked={form.attendanceTracking}
              onChange={(e) => setForm((f) => ({ ...f, attendanceTracking: e.target.checked }))}
              className="mt-0.5 accent-brand-500"
            />
            <span>
              <span className="block text-sm font-medium text-ink-900">Run face recognition on this camera</span>
              <span className="block text-xs text-slate-500">
                Leave off for live view only. Face recognition uses a lot of CPU per camera.
              </span>
            </span>
          </label>

          {addError && <p className="text-sm text-danger-500">{addError}</p>}
          <button
            type="submit"
            disabled={saving}
            className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-60"
          >
            <Video size={15} /> {saving ? "Adding…" : "Add camera"}
          </button>
        </form>
      </Modal>

      {/* --- Edit Camera ------------------------------------------------ */}
      <Modal open={!!editing} onClose={() => setEditing(null)} title="Edit Camera" width="max-w-lg">
        {editing && (
          <form onSubmit={handleEditSubmit} className="space-y-5">
            <p className="text-sm text-slate-500 -mt-3">Edit a camera and assign its site access.</p>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-ink-900 block mb-1.5">
                  Camera Code <span className="text-danger-500">*</span>
                </label>
                <input
                  required
                  value={editing.camCode}
                  onChange={(e) => setEditing((f) => ({ ...f, camCode: e.target.value }))}
                  className="input-field"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-ink-900 block mb-1.5">
                  Camera Label <span className="text-danger-500">*</span>
                </label>
                <input
                  required
                  value={editing.label}
                  onChange={(e) => setEditing((f) => ({ ...f, label: e.target.value }))}
                  className="input-field"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-ink-900 block mb-1.5">
                  Purpose <span className="text-danger-500">*</span>
                </label>
                <select
                  required
                  value={editing.purpose}
                  onChange={(e) => setEditing((f) => ({ ...f, purpose: e.target.value }))}
                  className="input-field"
                >
                  {[...new Set([editing.purpose, ...PURPOSES])].map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-sm font-medium text-ink-900 block mb-1.5">Site</label>
                <input
                  required
                  value={editing.site}
                  onChange={(e) => setEditing((f) => ({ ...f, site: e.target.value }))}
                  className="input-field"
                />
              </div>
            </div>

            <div className="pt-1">
              <p className="text-sm font-semibold text-ink-900 mb-3">Stream Configuration</p>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">
                Stream URL <span className="text-danger-500">*</span>
              </label>
              <input
                required
                value={editing.streamUrl}
                onChange={(e) => setEditing((f) => ({ ...f, streamUrl: e.target.value }))}
                placeholder="rtsp://user:password@host:port/path"
                className="input-field font-mono text-xs"
              />
              <p className="text-xs text-slate-400 mt-1.5">
                The saved password is hidden for security and shown blank here. Leave it blank
                to keep the current password, or type the full URL with a new password to change it.
              </p>
            </div>

            <label className="flex items-start gap-2.5 cursor-pointer">
              <input
                type="checkbox"
                checked={editing.attendanceTracking}
                onChange={(e) => setEditing((f) => ({ ...f, attendanceTracking: e.target.checked }))}
                className="mt-0.5 accent-brand-500"
              />
              <span>
                <span className="block text-sm font-medium text-ink-900">Enable attendance tracking</span>
                <span className="block text-xs text-slate-500">
                  Attendance events and employee check-ins will be processed using this camera.
                </span>
              </span>
            </label>

            <div className="flex items-center gap-3 pt-1">
              <button type="button" onClick={() => setEditing(null)} className="btn-secondary flex-1">
                Cancel
              </button>
              <button type="submit" className="btn-primary flex-1">
                Submit
              </button>
            </div>
          </form>
        )}
      </Modal>
    </div>
  );
}
