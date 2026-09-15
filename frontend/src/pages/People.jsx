import { useEffect, useState } from "react";
import { Camera, Check, Download, Pencil, Search, Trash2, UserPlus } from "lucide-react";
import PageHeader from "../components/PageHeader";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import Modal from "../components/Modal";
import Avatar from "../components/Avatar";
import * as api from "../api/client";
import { employeeMeta, incomingGuests, personTypes, validatedMeta } from "../data/peopleExtra";

const TABS = ["Employee", "Incoming Guests", "Validated"];

const EMPTY_FORM = { type: "Employee", firstName: "", lastName: "", employeeId: "" };

export default function People() {
  const [tab, setTab] = useState("Employee");
  const [employees, setEmployees] = useState([]);
  const [guests, setGuests] = useState(incomingGuests);
  const [validated, setValidated] = useState([]);
  const [search, setSearch] = useState("");
  const [toast, setToast] = useState("");

  // Add-person wizard
  const [addOpen, setAddOpen] = useState(false);
  const [step, setStep] = useState(1);
  const [form, setForm] = useState(EMPTY_FORM);
  const [captured, setCaptured] = useState(false);
  const [newPerson, setNewPerson] = useState(null);

  // Edit modal
  const [editing, setEditing] = useState(null);

  useEffect(() => {
    api.getPeople().then((rows) =>
      setEmployees(rows.map((r, i) => ({ ...r, ...(employeeMeta[i] || {}), type: "Employee" })))
    );
    api.getValidatedPeople().then((rows) =>
      setValidated(rows.map((r, i) => ({ ...r, ...(validatedMeta[i] || {}) })))
    );
  }, []);

  function showToast(msg) {
    setToast(msg);
    setTimeout(() => setToast(""), 2500);
  }

  function resetWizard() {
    setStep(1);
    setForm(EMPTY_FORM);
    setCaptured(false);
    setNewPerson(null);
  }

  function openAdd() {
    resetWizard();
    setAddOpen(true);
  }

  function closeAdd() {
    setAddOpen(false);
    resetWizard();
  }

  function handleCapture() {
    setCaptured(true);
  }

  function finishEnrollment() {
    const name = `${form.firstName} ${form.lastName}`.trim();
    const record = {
      name,
      faceEnrolled: captured,
      designs: 0,
      date: "Sep 15",
      enrollment: captured ? "Enrolled" : "Not enrolled",
      employeeId: form.type === "Employee" ? form.employeeId || "-" : "-",
      syncStatus: "Pending sync",
      guestOf: form.type === "Guest" ? "Front desk" : undefined,
      type: form.type,
    };
    if (form.type === "Employee") {
      setEmployees((prev) => [record, ...prev]);
    } else {
      setGuests((prev) => [record, ...prev]);
    }
    setNewPerson(record);
    setStep(3);
  }

  function deletePerson(list, setList, row) {
    if (!window.confirm(`Remove ${row.name} from People?`)) return;
    setList(list.filter((r) => r !== row));
    showToast(`${row.name} removed`);
  }

  function saveEdit(e) {
    e.preventDefault();
    const isEmployee = editing.type !== "Guest";
    const setList = isEmployee ? setEmployees : setGuests;
    setList((prev) => prev.map((r) => (r === editing.original ? { ...r, ...editing } : r)));
    showToast("Details updated");
    setEditing(null);
  }

  function validateGuest(row) {
    setValidated((prev) => prev.map((r) => (r === row ? { ...r, enrollment: "Validated" } : r)));
  }

  const employeeRows = employees.filter((r) => r.name.toLowerCase().includes(search.toLowerCase()));
  const guestRows = guests.filter((r) => r.name.toLowerCase().includes(search.toLowerCase()));
  const validatedRows = validated.filter((r) => r.name.toLowerCase().includes(search.toLowerCase()));

  const personColumns = (setList, list) => [
    {
      key: "name",
      label: "Photo",
      render: (r) => <Avatar name={r.name} />,
    },
    { key: "name2", label: "Name", render: (r) => <span className="font-medium text-ink-900">{r.name}</span> },
    { key: "enrollment", label: "Face enrolled", render: (r) => <StatusBadge value={r.enrollment} /> },
    { key: "designs", label: "Designs" },
    { key: "date", label: "Date" },
    {
      key: "syncStatus",
      label: "Enrollment status",
      render: (r) => <StatusBadge value={r.syncStatus === "Active" ? "Enrolled" : "Acknowledged"} />,
    },
    {
      key: "action",
      label: "Actions",
      render: (r) => (
        <div className="flex items-center gap-3">
          <button
            onClick={() => setEditing({ ...r, original: r })}
            title="Edit"
            className="text-slate-400 hover:text-brand-600"
          >
            <Pencil size={15} />
          </button>
          <button
            onClick={() => deletePerson(list, setList, r)}
            title="Delete"
            className="text-slate-400 hover:text-danger-500"
          >
            <Trash2 size={15} />
          </button>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-5">
      <PageHeader
        title="People"
        action={
          <div className="flex items-center gap-3">
            <button
              onClick={() => showToast("Export started — file will download shortly")}
              className="text-sm font-medium text-slate-500 hover:text-ink-900 flex items-center gap-1.5"
            >
              <Download size={15} /> Export
            </button>
            <button onClick={openAdd} className="btn-primary flex items-center gap-2 text-sm">
              <UserPlus size={16} /> Add person
            </button>
          </div>
        }
      />

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex items-center gap-1 rounded-full bg-white border border-border-200 p-1">
          {TABS.map((t) => {
            const count = t === "Employee" ? employeeRows.length : t === "Incoming Guests" ? guestRows.length : validatedRows.length;
            return (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
                  tab === t ? "bg-brand-500 text-white" : "text-slate-500 hover:text-ink-900"
                }`}
              >
                {t} ({count})
              </button>
            );
          })}
        </div>
        <div className="relative w-64 max-w-full">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name"
            className="input-field pl-9"
          />
        </div>
      </div>

      {toast && (
        <div className="text-sm text-success-600 bg-success-50 border border-success-500/20 rounded-lg px-3.5 py-2">
          {toast}
        </div>
      )}

      {tab === "Employee" && (
        <DataTable columns={personColumns(setEmployees, employees)} rows={employeeRows} />
      )}

      {tab === "Incoming Guests" && (
        <DataTable
          columns={[
            { key: "photo", label: "Photo", render: (r) => <Avatar name={r.name} /> },
            { key: "name", label: "Name", render: (r) => <span className="font-medium text-ink-900">{r.name}</span> },
            { key: "guestOf", label: "Guest of" },
            { key: "enrollment", label: "Face enrolled", render: (r) => <StatusBadge value={r.enrollment} /> },
            { key: "date", label: "Date" },
            {
              key: "action",
              label: "Actions",
              render: (r) => (
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setEditing({ ...r, original: r, type: "Guest" })}
                    title="Edit"
                    className="text-slate-400 hover:text-brand-600"
                  >
                    <Pencil size={15} />
                  </button>
                  <button
                    onClick={() => deletePerson(guests, setGuests, r)}
                    title="Delete"
                    className="text-slate-400 hover:text-danger-500"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              ),
            },
          ]}
          rows={guestRows}
        />
      )}

      {tab === "Validated" && (
        <DataTable
          columns={[
            { key: "photo", label: "Photo", render: (r) => <Avatar name={r.name} /> },
            { key: "name", label: "Name", render: (r) => <span className="font-medium text-ink-900">{r.name}</span> },
            { key: "guestOf", label: "Guest of" },
            { key: "date", label: "Date" },
            { key: "enrollment", label: "Guest enrollment", render: (r) => <StatusBadge value={r.enrollment} /> },
            {
              key: "action",
              label: "",
              render: (r) =>
                r.enrollment !== "Validated" ? (
                  <button onClick={() => validateGuest(r)} className="text-brand-600 text-sm font-medium">
                    Validate
                  </button>
                ) : null,
            },
          ]}
          rows={validatedRows}
        />
      )}

      {/* --- Add person wizard --------------------------------------------- */}
      <Modal
        open={addOpen}
        onClose={closeAdd}
        title={step === 1 ? "Add person" : step === 2 ? "Face enrollment" : undefined}
        width="max-w-lg"
      >
        {step === 1 && (
          <div className="space-y-5">
            <div className="grid grid-cols-2 gap-3">
              {personTypes.map((pt) => (
                <button
                  key={pt.key}
                  type="button"
                  onClick={() => setForm((f) => ({ ...f, type: pt.key }))}
                  className={`text-left rounded-xl border p-4 transition-colors ${
                    form.type === pt.key ? "border-brand-500 bg-brand-50" : "border-border-200 hover:bg-[#f8f9fc]"
                  }`}
                >
                  <p className="text-sm font-semibold text-ink-900">{pt.label}</p>
                  <p className="text-xs text-slate-500 mt-1">{pt.desc}</p>
                </button>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-sm font-medium text-ink-900 block mb-1.5">First name</label>
                <input
                  required
                  value={form.firstName}
                  onChange={(e) => setForm((f) => ({ ...f, firstName: e.target.value }))}
                  className="input-field"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-ink-900 block mb-1.5">Last name</label>
                <input
                  required
                  value={form.lastName}
                  onChange={(e) => setForm((f) => ({ ...f, lastName: e.target.value }))}
                  className="input-field"
                />
              </div>
            </div>

            {form.type === "Employee" && (
              <div>
                <label className="text-sm font-medium text-ink-900 block mb-1.5">Employee ID</label>
                <input
                  required
                  value={form.employeeId}
                  onChange={(e) => setForm((f) => ({ ...f, employeeId: e.target.value }))}
                  placeholder="EMP-2205"
                  className="input-field"
                />
              </div>
            )}

            <button
              type="button"
              disabled={
                !form.firstName || !form.lastName || (form.type === "Employee" && !form.employeeId)
              }
              onClick={() => setStep(2)}
              className="btn-primary w-full"
            >
              Continue to face enrollment
            </button>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-5">
            <div className="mx-auto w-48 h-48 rounded-2xl border-2 border-dashed border-border-300 bg-[#f8f9fc] flex flex-col items-center justify-center gap-2">
              {captured ? (
                <>
                  <span className="w-14 h-14 rounded-full bg-success-500 text-white flex items-center justify-center">
                    <Check size={26} />
                  </span>
                  <p className="text-xs font-medium text-success-600">Photo captured</p>
                </>
              ) : (
                <>
                  <Camera size={32} className="text-slate-400" />
                  <p className="text-xs text-slate-400 px-6 text-center">Position the face inside the frame</p>
                </>
              )}
            </div>
            <button
              type="button"
              onClick={handleCapture}
              className={captured ? "btn-secondary w-full" : "btn-primary w-full"}
            >
              {captured ? "Retake photo" : "Capture photo"}
            </button>
            <div className="flex items-center gap-3">
              <button type="button" onClick={() => setStep(1)} className="btn-secondary flex-1">
                Back
              </button>
              <button type="button" onClick={finishEnrollment} className="btn-primary flex-1">
                {captured ? "Finish" : "Skip for now"}
              </button>
            </div>
          </div>
        )}

        {step === 3 && newPerson && (
          <div className="text-center space-y-4 py-2">
            <Avatar name={newPerson.name} size={64} className="mx-auto text-lg" />
            <div>
              <p className="text-base font-semibold text-ink-900">Person added successfully</p>
              <p className="text-sm text-slate-500 mt-1">{newPerson.name}</p>
            </div>
            <p className="text-xs text-slate-400 bg-[#f8f9fc] border border-border-200 rounded-xl px-4 py-3">
              You cannot add or delete this person from People once enrollment syncs — manage
              enrollment status from the kiosk instead.
            </p>
            <div className="flex items-center gap-3 pt-1">
              <button type="button" onClick={openAdd} className="btn-secondary flex-1">
                + Add another person
              </button>
              <button type="button" onClick={closeAdd} className="btn-primary flex-1">
                View person details
              </button>
            </div>
          </div>
        )}
      </Modal>

      {/* --- Edit modal ----------------------------------------------------- */}
      <Modal open={!!editing} onClose={() => setEditing(null)} title="Edit person">
        {editing && (
          <form onSubmit={saveEdit} className="space-y-4">
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Name</label>
              <input
                value={editing.name}
                onChange={(e) => setEditing((f) => ({ ...f, name: e.target.value }))}
                className="input-field"
              />
            </div>
            {editing.employeeId !== undefined && (
              <div>
                <label className="text-sm font-medium text-ink-900 block mb-1.5">Employee ID</label>
                <input
                  value={editing.employeeId}
                  onChange={(e) => setEditing((f) => ({ ...f, employeeId: e.target.value }))}
                  className="input-field"
                />
              </div>
            )}
            <div className="flex items-center gap-3 pt-2">
              <button type="button" onClick={() => setEditing(null)} className="btn-secondary flex-1">
                Cancel
              </button>
              <button type="submit" className="btn-primary flex-1">
                Save changes
              </button>
            </div>
          </form>
        )}
      </Modal>
    </div>
  );
}
