import { useEffect, useState } from "react";
import PageHeader from "../components/PageHeader";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import Modal from "../components/Modal";
import * as api from "../api/client";

const SITE_SUGGESTIONS = ["Noida", "Mumbai", "Bangalore", "Delhi"];
const CREATE_NEW = "__create_new__";

export default function SiteManagement() {
  const [sites, setSites] = useState([]);
  const [addOpen, setAddOpen] = useState(false);
  const [selected, setSelected] = useState(SITE_SUGGESTIONS[0]);
  const [customName, setCustomName] = useState("");

  useEffect(() => {
    api.getSites().then(setSites);
  }, []);

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
    api.getSites().then(setSites);
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

      <DataTable
        columns={[
          { key: "name", label: "Site" },
          { key: "cameras", label: "Cameras" },
          { key: "wgs", label: "Wgs" },
          { key: "status", label: "Status", render: (r) => <StatusBadge value={r.status} /> },
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
    </div>
  );
}
