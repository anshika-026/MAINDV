import { useState } from "react";
import DataTable from "../../components/DataTable";
import Modal from "../../components/Modal";

const INITIAL = [
  { reason: "Holiday Calendar", date: "28/08/2026", day: "Wednesday", appliesTo: "All employees" },
];

export default function RulesPolicy() {
  const [rules, setRules] = useState(INITIAL);
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ reason: "", date: "", appliesTo: "" });

  function handleAdd(e) {
    e.preventDefault();
    const day = form.date
      ? new Date(form.date).toLocaleDateString(undefined, { weekday: "long" })
      : "";
    setRules((r) => [...r, { ...form, day }]);
    setOpen(false);
    setForm({ reason: "", date: "", appliesTo: "" });
  }

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-base font-semibold text-ink-900">Rules &amp; Policy</h2>
        <button onClick={() => setOpen(true)} className="btn-primary text-sm">
          Add Holiday
        </button>
      </div>

      <DataTable
        columns={[
          { key: "reason", label: "Reason" },
          { key: "date", label: "Date" },
          { key: "day", label: "Day" },
          { key: "appliesTo", label: "Applies to" },
          { key: "action", label: "", render: () => <button className="text-danger-500 text-sm font-medium">Delete</button> },
        ]}
        rows={rules}
      />

      <Modal open={open} onClose={() => setOpen(false)} title="Add Holiday">
        <form onSubmit={handleAdd} className="space-y-4">
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Reason</label>
            <input
              required
              value={form.reason}
              onChange={(e) => setForm((f) => ({ ...f, reason: e.target.value }))}
              className="input-field"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Select date</label>
            <input
              type="date"
              required
              value={form.date}
              onChange={(e) => setForm((f) => ({ ...f, date: e.target.value }))}
              className="input-field"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Applies to</label>
            <select
              value={form.appliesTo}
              onChange={(e) => setForm((f) => ({ ...f, appliesTo: e.target.value }))}
              className="input-field"
            >
              <option value="All employees">All employees</option>
              <option value="Specific site">Specific site</option>
            </select>
          </div>
          <div className="flex items-center gap-3 pt-2">
            <button type="button" onClick={() => setOpen(false)} className="btn-secondary flex-1">
              Cancel
            </button>
            <button type="submit" className="btn-primary flex-1">
              Add holiday
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
