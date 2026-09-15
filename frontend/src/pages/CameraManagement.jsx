import { useEffect, useState } from "react";
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

export default function CameraManagement() {
  const [cameras, setCameras] = useState([]);
  const [addOpen, setAddOpen] = useState(false);
  const [form, setForm] = useState({ code: "", driveName: "", purpose: "", site: "" });
  const [view, setView] = useState("All");

  useEffect(() => {
    api.getCameras().then(setCameras);
  }, []);

  const online = cameras.filter((c) => c.status === "Active").length;
  const avgHealth = cameras.length
    ? Math.round(cameras.reduce((sum, c) => sum + healthForCamera(c.id), 0) / cameras.length)
    : null;

  async function handleAdd(e) {
    e.preventDefault();
    await api.addCamera(form);
    setAddOpen(false);
    setForm({ code: "", driveName: "", purpose: "", site: "" });
    api.getCameras().then(setCameras);
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
              label: "",
              render: () => (
                <div className="flex items-center gap-3 text-sm">
                  <button className="text-brand-600 font-medium">Edit</button>
                  <button className="text-danger-500 font-medium">Remove</button>
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
    </div>
  );
}
