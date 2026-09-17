import { useEffect, useState } from "react";
import { Pencil, Trash2 } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import Modal from "../components/Modal";
import Tabs from "../components/Tabs";
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

export default function CameraManagement() {
  const [cameras, setCameras] = useState([]);
  const [addOpen, setAddOpen] = useState(false);
  const [form, setForm] = useState({ code: "", driveName: "", purpose: "", site: "" });
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
  const avgHealth = cameras.length
    ? Math.round(cameras.reduce((sum, c) => sum + healthForCamera(c.id), 0) / cameras.length)
    : null;

  async function handleAdd(e) {
    e.preventDefault();
    await api.addCamera(form);
    setAddOpen(false);
    setForm({ code: "", driveName: "", purpose: "", site: "" });
    refresh();
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
          <button onClick={() => setAddOpen(true)} className="btn-primary text-sm">
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
            { key: "live", label: "Live feed" },
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

      <Modal open={addOpen} onClose={() => setAddOpen(false)} title="Add Camera">
        <form onSubmit={handleAdd} className="space-y-4">
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Cam Code</label>
            <input
              required
              value={form.code}
              onChange={(e) => setForm((f) => ({ ...f, code: e.target.value }))}
              placeholder="CAM-XXXX-XXXX-XXXX"
              className="input-field"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Give name to camera</label>
            <input
              required
              value={form.driveName}
              onChange={(e) => setForm((f) => ({ ...f, driveName: e.target.value }))}
              placeholder="e.g. Entry / Exit"
              className="input-field"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Purpose (optional)</label>
            <input
              value={form.purpose}
              onChange={(e) => setForm((f) => ({ ...f, purpose: e.target.value }))}
              placeholder="Theft, General"
              className="input-field"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Site</label>
            <input
              required
              value={form.site}
              onChange={(e) => setForm((f) => ({ ...f, site: e.target.value }))}
              placeholder="Noida"
              className="input-field"
            />
          </div>
          <button type="submit" className="btn-primary w-full">
            Add
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
