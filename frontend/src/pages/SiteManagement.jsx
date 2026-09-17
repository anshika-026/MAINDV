import { useEffect, useState } from "react";
import { Pencil, Trash2, X } from "lucide-react";
import PageHeader from "../components/PageHeader";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import Modal from "../components/Modal";
import * as api from "../api/client";

const SITE_SUGGESTIONS = ["Noida", "Mumbai", "Bangalore", "Delhi"];
const CREATE_NEW = "__create_new__";

export default function SiteManagement() {
  const [sites, setSites] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [addOpen, setAddOpen] = useState(false);
  const [selected, setSelected] = useState(SITE_SUGGESTIONS[0]);
  const [customName, setCustomName] = useState("");
  const [editing, setEditing] = useState(null);
  const [toast, setToast] = useState("");

  useEffect(() => {
    refresh();
    api.getCameras().then(setCameras);
  }, []);

  function refresh() {
    api.getSites().then(setSites);
  }

  function showToast(msg) {
    setToast(msg);
    setTimeout(() => setToast(""), 2500);
  }

  const isCustom = selected === CREATE_NEW;
  const name = isCustom ? customName.trim() : selected;

  function resetForm() {
    setSelected(SITE_SUGGESTIONS[0]);
    setCustomName("");
  }

  function closeModal() {
    setAddOpen(false);
    resetForm();
  }

  async function handleAdd(e) {
    e.preventDefault();
    if (!name) return;
    await api.addSite({ name });
    closeModal();
    refresh();
  }

  function openEdit(site) {
    setEditing({
      id: site.id,
      name: site.name,
      description: site.description,
      cameraIds: site.cameraList.map((c) => c.id),
    });
  }

  function toggleCamera(id) {
    setEditing((f) => ({
      ...f,
      cameraIds: f.cameraIds.includes(id) ? f.cameraIds.filter((c) => c !== id) : [...f.cameraIds, id],
    }));
  }

  async function handleEditSubmit(e) {
    e.preventDefault();
    const original = sites.find((s) => s.id === editing.id);
    const originalIds = original ? original.cameraList.map((c) => c.id) : [];

    await api.updateSite(editing.id, { name: editing.name, description: editing.description });

    const added = editing.cameraIds.filter((id) => !originalIds.includes(id));
    const removed = originalIds.filter((id) => !editing.cameraIds.includes(id));
    await Promise.all([
      ...added.map((id) => api.updateCamera(id, { site: editing.name })),
      ...removed.map((id) => api.updateCamera(id, { site: "" })),
    ]);

    setEditing(null);
    showToast("Site updated");
    refresh();
  }

  async function handleDelete(site) {
    if (!window.confirm(`Remove site "${site.name}"? This cannot be undone.`)) return;
    await api.deleteSite(site.id);
    showToast(`${site.name} removed`);
    refresh();
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Site Management"
        action={
          <button onClick={() => setAddOpen(true)} className="btn-primary text-sm">
            + Add Site
          </button>
        }
      />

      {toast && (
        <div className="text-sm text-success-600 bg-success-50 border border-success-500/20 rounded-lg px-3.5 py-2">
          {toast}
        </div>
      )}

      <DataTable
        columns={[
          { key: "name", label: "Site" },
          { key: "cameras", label: "Cameras" },
          { key: "wgs", label: "Wgs" },
          { key: "status", label: "Status", render: (r) => <StatusBadge value={r.status} /> },
          {
            key: "action",
            label: "Actions",
            render: (r) => (
              <div className="flex items-center gap-3">
                <button onClick={() => openEdit(r)} title="Edit" className="text-slate-400 hover:text-brand-600">
                  <Pencil size={15} />
                </button>
                <button onClick={() => handleDelete(r)} title="Delete" className="text-slate-400 hover:text-danger-500">
                  <Trash2 size={15} />
                </button>
              </div>
            ),
          },
        ]}
        rows={sites}
      />

      <Modal open={addOpen} onClose={closeModal} title="Add site">
        <form onSubmit={handleAdd} className="space-y-4">
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Site name</label>
            <select
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              className="input-field"
            >
              {SITE_SUGGESTIONS.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
              <option value={CREATE_NEW}>+ Create new site</option>
            </select>
          </div>

          {isCustom && (
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">New site name</label>
              <input
                required
                autoFocus
                value={customName}
                onChange={(e) => setCustomName(e.target.value)}
                placeholder="e.g. Pune"
                className="input-field"
              />
            </div>
          )}

          <div className="flex items-center gap-3 pt-2">
            <button type="button" onClick={closeModal} className="btn-secondary flex-1">
              Cancel
            </button>
            <button type="submit" disabled={!name} className="btn-primary flex-1">
              Add
            </button>
          </div>
        </form>
      </Modal>

      {/* --- Edit Site ---------------------------------------------------- */}
      <Modal open={!!editing} onClose={() => setEditing(null)} title="Edit Site" width="max-w-lg">
        {editing && (
          <form onSubmit={handleEditSubmit} className="space-y-5">
            <p className="text-sm text-slate-500 -mt-3">Edit site details and assign cameras and users</p>

            <div>
              <p className="text-sm font-semibold text-ink-900 mb-3">Basic Information</p>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">
                Site Name <span className="text-danger-500">*</span>
              </label>
              <input
                required
                value={editing.name}
                onChange={(e) => setEditing((f) => ({ ...f, name: e.target.value }))}
                className="input-field"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Description</label>
              <textarea
                rows={3}
                value={editing.description}
                onChange={(e) => setEditing((f) => ({ ...f, description: e.target.value }))}
                className="input-field resize-y"
              />
            </div>

            <div>
              <p className="text-sm font-semibold text-ink-900 mb-3">Resources</p>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Cameras</label>
              <div className="flex flex-wrap gap-2 rounded-xl border border-border-300 p-2.5 min-h-[46px]">
                {editing.cameraIds.length === 0 && (
                  <span className="text-sm text-slate-400 px-1 py-0.5">No cameras assigned</span>
                )}
                {editing.cameraIds.map((id) => {
                  const cam = cameras.find((c) => c.id === id);
                  return (
                    <span
                      key={id}
                      className="inline-flex items-center gap-1 bg-[#f0f2f9] text-ink-900 text-sm rounded-full pl-3 pr-1.5 py-1"
                    >
                      {cam ? cam.label : `Camera ${id}`}
                      <button
                        type="button"
                        onClick={() => toggleCamera(id)}
                        className="text-slate-400 hover:text-danger-500"
                      >
                        <X size={13} />
                      </button>
                    </span>
                  );
                })}
              </div>
              <select
                value=""
                onChange={(e) => e.target.value && toggleCamera(Number(e.target.value))}
                className="input-field mt-2"
              >
                <option value="">+ Add a camera to this site</option>
                {cameras
                  .filter((c) => !editing.cameraIds.includes(c.id))
                  .map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.label}
                    </option>
                  ))}
              </select>
              <p className="text-xs text-slate-400 mt-1.5">Assign cameras to this site for monitoring and access control.</p>
            </div>

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
