import { useEffect, useMemo, useState } from "react";
import { Calendar, ChevronRight, Download, MapPin, Pencil, Search, Trash2, UserPlus, X } from "lucide-react";
import PageHeader from "../components/PageHeader";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import Modal from "../components/Modal";
import SidePanel from "../components/SidePanel";
import Avatar from "../components/Avatar";
import FaceEnrollment from "../components/FaceEnrollment";
import * as api from "../api/client";
import {
  departments,
  employeeMeta,
  enrollmentFilters,
  incomingGuests,
  personTypes,
  validatedMeta,
} from "../data/peopleExtra";

const TABS = ["Employee", "Incoming Guests", "Validated"];

const EMPTY_FORM = {
  type: "Employee",
  firstName: "",
  lastName: "",
  department: departments[0],
  host: "",
  from: "",
  to: "",
  photos: [],
};

// Existing mock rows carry only a sample *count* (no real files behind
// them) — this fills in placeholder tiles so the gallery still has
// something to show/delete for people who were "enrolled" before this
// session started.
function placeholderPhotos(prefix, count) {
  return Array.from({ length: count || 0 }, (_, i) => ({ id: `${prefix}-${i}`, url: null }));
}

export default function People() {
  const [tab, setTab] = useState("Employee");
  const [employees, setEmployees] = useState([]);
  const [guests, setGuests] = useState(
    incomingGuests.map((g, i) => ({ ...g, type: "Guest", photos: placeholderPhotos(`guest-${i}`, g.designs) }))
  );
  const [validated, setValidated] = useState([]);
  const [search, setSearch] = useState("");
  const [enrollmentFilter, setEnrollmentFilter] = useState(enrollmentFilters[0]);
  const [hostFilter, setHostFilter] = useState("All hosts");
  const [dateOpen, setDateOpen] = useState(false);
  const [date, setDate] = useState("2026-08-28");
  const [toast, setToast] = useState("");

  // Add-person wizard
  const [addOpen, setAddOpen] = useState(false);
  const [step, setStep] = useState(1);
  const [form, setForm] = useState(EMPTY_FORM);
  const [newPerson, setNewPerson] = useState(null);

  // Edit modal — mirrors the add-person wizard (details, then face enrollment)
  const [editing, setEditing] = useState(null);
  const [editStep, setEditStep] = useState(1);

  // Row detail side panel
  const [selected, setSelected] = useState(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  // Sample-photo gallery (click a row's avatar to open)
  const [gallery, setGallery] = useState(null); // { person, setList }
  const [galleryPreview, setGalleryPreview] = useState(null);

  useEffect(() => {
    api.getPeople().then((rows) =>
      setEmployees(
        rows.map((r, i) => {
          // employeeMeta only supplies demo defaults (zone/camera/...) for the
          // mock fallback rows — real API fields (employeeId, photos, designs,
          // enrollment) always win when present.
          const merged = { ...(employeeMeta[i] || {}), ...r, type: "Employee" };
          return {
            ...merged,
            photos: merged.photos && merged.photos.length ? merged.photos : placeholderPhotos(`emp-${i}`, merged.designs),
          };
        })
      )
    );
    api.getValidatedPeople().then((rows) =>
      setValidated(
        rows.map((r, i) => {
          const merged = { ...r, ...(validatedMeta[i] || {}), type: "Guest" };
          return { ...merged, photos: placeholderPhotos(`val-${i}`, merged.designs) };
        })
      )
    );
  }, []);

  function showToast(msg) {
    setToast(msg);
    setTimeout(() => setToast(""), 2500);
  }

  function resetWizard() {
    setStep(1);
    setForm(EMPTY_FORM);
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

  function addFormPhoto(url) {
    setForm((f) => ({ ...f, photos: [...f.photos, { id: crypto.randomUUID(), url }] }));
  }

  function removeFormPhoto(id) {
    setForm((f) => ({ ...f, photos: f.photos.filter((p) => p.id !== id) }));
  }

  const canContinue =
    form.firstName.trim() && form.lastName.trim() && (form.type === "Employee" ? form.department : form.host.trim());

  function finishEnrollment() {
    const name = `${form.firstName} ${form.lastName}`.trim();
    const enrolled = form.photos.length > 0;
    const record = {
      name,
      photos: form.photos,
      faceEnrolled: enrolled,
      designs: form.photos.length,
      date: "Sep 15",
      enrollment: enrolled ? "Enrolled" : "Not enrolled",
      employeeId: form.type === "Employee" ? `EMP-${Math.floor(2200 + Math.random() * 90)}` : "-",
      syncStatus: "Pending sync",
      guestOf: form.type === "Guest" ? form.host : undefined,
      department: form.type === "Employee" ? form.department : undefined,
      status: "Away",
      zone: "-",
      camera: "-",
      lastSeenDesk: "-",
      confidence: 0,
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

  function splitName(name) {
    const parts = String(name || "").trim().split(/\s+/);
    return { firstName: parts[0] || "", lastName: parts.slice(1).join(" ") || "" };
  }

  function openEdit(row, type, setList) {
    const { firstName, lastName } = splitName(row.name);
    setEditing({
      original: row,
      type: type || row.type || "Employee",
      firstName,
      lastName,
      employeeId: row.employeeId || "",
      department: row.department || departments[0],
      host: row.guestOf || "",
      photos: row.photos || [],
      setList,
    });
    setEditStep(1);
  }

  function closeEdit() {
    setEditing(null);
    setEditStep(1);
  }

  function addEditPhoto(url) {
    setEditing((f) => ({ ...f, photos: [...f.photos, { id: crypto.randomUUID(), url }] }));
  }

  function removeEditPhoto(id) {
    setEditing((f) => ({ ...f, photos: f.photos.filter((p) => p.id !== id) }));
  }

  const canContinueEdit =
    editing && editing.firstName.trim() && editing.lastName.trim() &&
    (editing.type === "Employee" ? editing.department : editing.host.trim());

  function saveEdit() {
    const isEmployee = editing.type !== "Guest";
    const name = `${editing.firstName} ${editing.lastName}`.trim();
    const enrolled = editing.photos.length > 0;
    const updated = {
      name,
      photos: editing.photos,
      employeeId: isEmployee ? editing.employeeId : editing.original.employeeId,
      department: isEmployee ? editing.department : editing.original.department,
      guestOf: !isEmployee ? editing.host : editing.original.guestOf,
      enrollment: enrolled ? "Enrolled" : "Not enrolled",
      designs: editing.photos.length,
    };
    editing.setList((prev) => prev.map((r) => (r === editing.original ? { ...r, ...updated } : r)));
    showToast("Details updated");
    closeEdit();
  }

  function validateGuest(row) {
    setValidated((prev) => prev.map((r) => (r === row ? { ...r, enrollment: "Validated" } : r)));
  }

  function openGallery(person, setList) {
    setGallery({ person, setList });
  }

  function removeGalleryPhoto(photoId) {
    const { person, setList } = gallery;
    setList((prev) =>
      prev.map((r) => {
        if (r !== person) return r;
        const photos = (r.photos || []).filter((p) => p.id !== photoId);
        return { ...r, photos, designs: photos.length, enrollment: photos.length ? "Enrolled" : "Not enrolled" };
      })
    );
    setGallery((g) => g && { ...g, person: { ...g.person, photos: g.person.photos.filter((p) => p.id !== photoId) } });
    showToast("Sample removed");
  }

  const hostNames = useMemo(
    () => [...new Set(guests.map((g) => g.guestOf).filter(Boolean))],
    [guests]
  );

  const employeeRows = useMemo(
    () =>
      employees.filter((r) => {
        if (search && !`${r.name} ${r.employeeId || ""}`.toLowerCase().includes(search.toLowerCase())) return false;
        if (enrollmentFilter !== enrollmentFilters[0] && r.enrollment !== enrollmentFilter) return false;
        return true;
      }),
    [employees, search, enrollmentFilter]
  );

  const guestRows = useMemo(
    () =>
      guests.filter((r) => {
        if (search && !r.name.toLowerCase().includes(search.toLowerCase())) return false;
        if (enrollmentFilter !== enrollmentFilters[0] && r.enrollment !== enrollmentFilter) return false;
        if (hostFilter !== "All hosts" && r.guestOf !== hostFilter) return false;
        return true;
      }),
    [guests, search, enrollmentFilter, hostFilter]
  );

  const validatedRows = useMemo(
    () => validated.filter((r) => (search ? r.name.toLowerCase().includes(search.toLowerCase()) : true)),
    [validated, search]
  );

  const displayDate = new Date(`${date}T00:00:00`).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });

  function openDetail(row) {
    setDetailsOpen(false);
    setSelected(row);
  }

  function photoButton(row, setList) {
    return (
      <button
        onClick={(e) => {
          e.stopPropagation();
          openGallery(row, setList);
        }}
        title="View face samples"
        className="block rounded-full ring-offset-2 hover:ring-2 hover:ring-brand-200"
      >
        <Avatar name={row.name} />
      </button>
    );
  }

  const employeeColumns = [
    { key: "photo", label: "Photo", render: (r) => photoButton(r, setEmployees) },
    { key: "name", label: "Name", render: (r) => <span className="font-medium text-ink-900">{r.name}</span> },
    { key: "enrollment", label: "Face enrollment", render: (r) => <StatusBadge value={r.enrollment} /> },
    { key: "designs", label: "Samples" },
    {
      key: "action",
      label: "Actions",
      render: (r) => (
        <div className="flex items-center gap-3">
          <button
            onClick={(e) => {
              e.stopPropagation();
              openEdit(r, "Employee", setEmployees);
            }}
            title="Edit"
            className="text-slate-400 hover:text-brand-600"
          >
            <Pencil size={15} />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              deletePerson(employees, setEmployees, r);
            }}
            title="Delete"
            className="text-slate-400 hover:text-danger-500"
          >
            <Trash2 size={15} />
          </button>
        </div>
      ),
    },
  ];

  const guestColumns = [
    { key: "date", label: "Date" },
    { key: "photo", label: "Photo", render: (r) => photoButton(r, setGuests) },
    { key: "name", label: "Name", render: (r) => <span className="font-medium text-ink-900">{r.name}</span> },
    { key: "guestOf", label: "Hosted by" },
    { key: "enrollment", label: "Face enrollment", render: (r) => <StatusBadge value={r.enrollment} /> },
    { key: "designs", label: "Images" },
    {
      key: "action",
      label: "Actions",
      render: (r) => (
        <div className="flex items-center gap-3">
          <button
            onClick={(e) => {
              e.stopPropagation();
              openEdit(r, "Guest", setGuests);
            }}
            title="Edit"
            className="text-slate-400 hover:text-brand-600"
          >
            <Pencil size={15} />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              deletePerson(guests, setGuests, r);
            }}
            title="Delete"
            className="text-slate-400 hover:text-danger-500"
          >
            <Trash2 size={15} />
          </button>
        </div>
      ),
    },
  ];

  const validatedColumns = [
    { key: "date", label: "Date" },
    { key: "photo", label: "Photo", render: (r) => photoButton(r, setValidated) },
    { key: "name", label: "Name", render: (r) => <span className="font-medium text-ink-900">{r.name}</span> },
    { key: "enrollment", label: "Status", render: (r) => <StatusBadge value={r.enrollment} /> },
    {
      key: "action",
      label: "Actions",
      render: (r) =>
        r.enrollment !== "Validated" ? (
          <button
            onClick={(e) => {
              e.stopPropagation();
              validateGuest(r);
            }}
            className="text-brand-600 text-sm font-medium"
          >
            Validate
          </button>
        ) : null,
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

      <div className="inline-flex items-center gap-1 rounded-full bg-white border border-border-200 p-1 w-fit">
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

      <div className="flex flex-wrap items-center gap-3">
        <select
          value={enrollmentFilter}
          onChange={(e) => setEnrollmentFilter(e.target.value)}
          className="input-field w-auto"
        >
          {enrollmentFilters.map((f) => (
            <option key={f}>{f}</option>
          ))}
        </select>

        {tab === "Incoming Guests" && (
          <select value={hostFilter} onChange={(e) => setHostFilter(e.target.value)} className="input-field w-auto">
            <option>All hosts</option>
            {hostNames.map((h) => (
              <option key={h}>{h}</option>
            ))}
          </select>
        )}

        <div className="relative">
          <button onClick={() => setDateOpen((o) => !o)} className="btn-secondary text-sm flex items-center gap-2">
            <Calendar size={15} /> {displayDate}
          </button>
          {dateOpen && (
            <div className="absolute z-10 top-full mt-2 card p-3">
              <input type="date" value={date} onChange={(e) => setDate(e.target.value)} className="input-field" />
              <button onClick={() => setDateOpen(false)} className="btn-primary w-full mt-2 text-sm">
                Apply
              </button>
            </div>
          )}
        </div>

        <div className="relative flex-1 min-w-[180px] max-w-xs">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={tab === "Employee" ? "Search by employee name, Emp ID" : "Search by name"}
            className="input-field pl-9"
          />
        </div>
      </div>

      {toast && (
        <div className="text-sm text-success-600 bg-success-50 border border-success-500/20 rounded-lg px-3.5 py-2">
          {toast}
        </div>
      )}

      {tab === "Employee" && <DataTable columns={employeeColumns} rows={employeeRows} onRowClick={openDetail} />}

      {tab === "Incoming Guests" && <DataTable columns={guestColumns} rows={guestRows} onRowClick={openDetail} />}

      {tab === "Validated" && <DataTable columns={validatedColumns} rows={validatedRows} onRowClick={openDetail} />}

      {/* --- Add person wizard --------------------------------------------- */}
      <Modal
        open={addOpen}
        onClose={closeAdd}
        title={step === 1 ? "Add person" : step === 2 ? "Face enrollment" : undefined}
        width="max-w-lg"
      >
        {step === 1 && (
          <div className="space-y-5">
            <div className="space-y-2">
              {personTypes.map((pt) => (
                <label
                  key={pt.key}
                  className={`flex items-start gap-3 rounded-xl border p-3.5 cursor-pointer transition-colors ${
                    form.type === pt.key ? "border-brand-500 bg-brand-50" : "border-border-200 hover:bg-[#f8f9fc]"
                  }`}
                >
                  <input
                    type="radio"
                    name="personType"
                    checked={form.type === pt.key}
                    onChange={() => setForm((f) => ({ ...f, type: pt.key }))}
                    className="mt-1 accent-brand-500"
                  />
                  <span>
                    <span className="block text-sm font-semibold text-ink-900">{pt.label}</span>
                    <span className="block text-xs text-slate-500 mt-0.5">{pt.desc}</span>
                  </span>
                </label>
              ))}
            </div>

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

            {form.type === "Employee" ? (
              <div>
                <label className="text-sm font-medium text-ink-900 block mb-1.5">Department</label>
                <select
                  value={form.department}
                  onChange={(e) => setForm((f) => ({ ...f, department: e.target.value }))}
                  className="input-field"
                >
                  {departments.map((d) => (
                    <option key={d}>{d}</option>
                  ))}
                </select>
              </div>
            ) : (
              <>
                <div>
                  <label className="text-sm font-medium text-ink-900 block mb-1.5">Host</label>
                  <input
                    list="host-options"
                    value={form.host}
                    onChange={(e) => setForm((f) => ({ ...f, host: e.target.value }))}
                    placeholder="Search person name"
                    className="input-field"
                  />
                  <datalist id="host-options">
                    {employees.map((e) => (
                      <option key={e.name} value={e.name} />
                    ))}
                  </datalist>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-sm font-medium text-ink-900 block mb-1.5">From</label>
                    <input
                      type="date"
                      value={form.from}
                      onChange={(e) => setForm((f) => ({ ...f, from: e.target.value }))}
                      className="input-field"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium text-ink-900 block mb-1.5">To</label>
                    <input
                      type="date"
                      value={form.to}
                      onChange={(e) => setForm((f) => ({ ...f, to: e.target.value }))}
                      className="input-field"
                    />
                  </div>
                </div>
              </>
            )}

            <button type="button" disabled={!canContinue} onClick={() => setStep(2)} className="btn-primary w-full">
              Continue to face enrollment
            </button>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-5">
            <FaceEnrollment photos={form.photos} onAddPhoto={addFormPhoto} onRemovePhoto={removeFormPhoto} />
            <div className="flex items-center gap-3">
              <button type="button" onClick={() => setStep(1)} className="btn-secondary flex-1">
                Back
              </button>
              <button type="button" onClick={finishEnrollment} className="btn-primary flex-1">
                {form.photos.length > 0 ? "Finish" : "Skip for now"}
              </button>
            </div>
          </div>
        )}

        {step === 3 && newPerson && (
          <div className="text-center space-y-4 py-2">
            {newPerson.photos?.[0]?.url ? (
              <img
                src={newPerson.photos[0].url}
                alt={newPerson.name}
                className="w-16 h-16 rounded-full object-cover mx-auto"
              />
            ) : (
              <Avatar name={newPerson.name} size={64} className="mx-auto text-lg" />
            )}
            <div>
              <p className="text-base font-semibold text-ink-900">Person added successfully</p>
              <p className="text-sm text-slate-500 mt-1">{newPerson.name}</p>
            </div>
            <div className="flex items-center justify-center gap-2">
              <StatusBadge value={newPerson.enrollment} />
              <span className="badge badge-neutral">
                {newPerson.designs} sample{newPerson.designs === 1 ? "" : "s"}
              </span>
            </div>
            <p className="text-xs text-slate-400 bg-[#f8f9fc] border border-border-200 rounded-xl px-4 py-3">
              You can edit and delete this person from People.
            </p>
            <div className="flex items-center gap-3 pt-1">
              <button type="button" onClick={openAdd} className="btn-secondary flex-1">
                + Add another person
              </button>
              <button
                type="button"
                onClick={() => {
                  setAddOpen(false);
                  openDetail(newPerson);
                }}
                className="btn-primary flex-1"
              >
                View person details
              </button>
            </div>
          </div>
        )}
      </Modal>

      {/* --- Edit modal ------------------------------------------------------ */}
      <Modal
        open={!!editing}
        onClose={closeEdit}
        title={editStep === 1 ? "Edit person" : "Face enrollment"}
        width="max-w-lg"
      >
        {editing && editStep === 1 && (
          <div className="space-y-5">
            <span className={`badge ${editing.type === "Guest" ? "badge-neutral" : "badge-success"}`}>
              {editing.type}
            </span>

            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">First name</label>
              <input
                required
                value={editing.firstName}
                onChange={(e) => setEditing((f) => ({ ...f, firstName: e.target.value }))}
                className="input-field"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Last name</label>
              <input
                required
                value={editing.lastName}
                onChange={(e) => setEditing((f) => ({ ...f, lastName: e.target.value }))}
                className="input-field"
              />
            </div>

            {editing.type === "Employee" ? (
              <>
                <div>
                  <label className="text-sm font-medium text-ink-900 block mb-1.5">Employee ID</label>
                  <input
                    value={editing.employeeId}
                    onChange={(e) => setEditing((f) => ({ ...f, employeeId: e.target.value }))}
                    className="input-field"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-ink-900 block mb-1.5">Department</label>
                  <select
                    value={editing.department}
                    onChange={(e) => setEditing((f) => ({ ...f, department: e.target.value }))}
                    className="input-field"
                  >
                    {departments.map((d) => (
                      <option key={d}>{d}</option>
                    ))}
                  </select>
                </div>
              </>
            ) : (
              <div>
                <label className="text-sm font-medium text-ink-900 block mb-1.5">Host</label>
                <input
                  list="edit-host-options"
                  value={editing.host}
                  onChange={(e) => setEditing((f) => ({ ...f, host: e.target.value }))}
                  placeholder="Search person name"
                  className="input-field"
                />
                <datalist id="edit-host-options">
                  {employees.map((e) => (
                    <option key={e.name} value={e.name} />
                  ))}
                </datalist>
              </div>
            )}

            <div className="flex items-center gap-3 pt-1">
              <button type="button" onClick={closeEdit} className="btn-secondary flex-1">
                Cancel
              </button>
              <button type="button" disabled={!canContinueEdit} onClick={() => setEditStep(2)} className="btn-primary flex-1">
                Continue to face enrollment
              </button>
            </div>
          </div>
        )}

        {editing && editStep === 2 && (
          <div className="space-y-5">
            <FaceEnrollment photos={editing.photos} onAddPhoto={addEditPhoto} onRemovePhoto={removeEditPhoto} />
            <div className="flex items-center gap-3">
              <button type="button" onClick={() => setEditStep(1)} className="btn-secondary flex-1">
                Back
              </button>
              <button type="button" onClick={saveEdit} className="btn-primary flex-1">
                Save changes
              </button>
            </div>
          </div>
        )}
      </Modal>

      {/* --- Face sample gallery ---------------------------------------------- */}
      <Modal
        open={!!gallery}
        onClose={() => setGallery(null)}
        title={gallery ? `${gallery.person.name} — Face samples` : undefined}
        width="max-w-lg"
      >
        {gallery && (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <Avatar name={gallery.person.name} size={40} />
              <div>
                <p className="font-medium text-ink-900">{gallery.person.name}</p>
                <p className="text-xs text-slate-500">
                  {gallery.person.photos.length} sample{gallery.person.photos.length === 1 ? "" : "s"}
                </p>
              </div>
            </div>

            {gallery.person.photos.length === 0 ? (
              <p className="text-sm text-slate-400 text-center py-8">No face samples enrolled yet.</p>
            ) : (
              <div className="grid grid-cols-4 gap-3">
                {gallery.person.photos.map((p) => (
                  <div
                    key={p.id}
                    className="relative aspect-square rounded-lg overflow-hidden border border-border-200 bg-[#f8f9fc]"
                  >
                    <button
                      type="button"
                      onClick={() => setGalleryPreview(p)}
                      className="w-full h-full flex items-center justify-center"
                    >
                      {p.url ? (
                        <img src={p.url} alt="Sample" className="w-full h-full object-cover" />
                      ) : (
                        <span className="text-[10px] text-slate-400 text-center px-1">No preview</span>
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={() => removeGalleryPhoto(p.id)}
                      className="absolute top-1 right-1 w-5 h-5 rounded-full bg-black/60 text-white flex items-center justify-center hover:bg-danger-500"
                      title="Remove photo"
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <button
              type="button"
              onClick={() => {
                const { person, setList } = gallery;
                setGallery(null);
                openEdit(person, person.type, setList);
                setEditStep(2);
              }}
              className="btn-secondary w-full text-sm"
            >
              + Add more samples
            </button>
          </div>
        )}
      </Modal>

      <Modal open={!!galleryPreview} onClose={() => setGalleryPreview(null)} title="Photo preview">
        {galleryPreview &&
          (galleryPreview.url ? (
            <img src={galleryPreview.url} alt="Sample preview" className="w-full rounded-xl" />
          ) : (
            <div className="w-full aspect-square rounded-xl bg-[#f8f9fc] flex items-center justify-center text-sm text-slate-400 text-center px-6">
              No preview available for this sample — it was enrolled before file storage was wired up.
            </div>
          ))}
      </Modal>

      {/* --- Person detail side panel ---------------------------------------- */}
      <SidePanel open={!!selected} onClose={() => { setSelected(null); setDetailsOpen(false); }} title="Person detail">
        {selected && (
          <div className="space-y-5">
            <div className="flex items-center gap-3">
              <Avatar name={selected.name} size={48} />
              <div>
                <p className="font-semibold text-ink-900">{selected.name}</p>
                <span className={`badge mt-1 ${selected.status === "Away" ? "badge-warning" : "badge-success"}`}>
                  {selected.status || (selected.type === "Employee" ? "Employee" : "Guest")}
                </span>
              </div>
            </div>

            <div className="space-y-2.5 text-sm">
              <p className="flex items-center justify-between">
                <span className="text-slate-400">Current location</span>
                <span className="font-medium text-ink-900">{selected.zone || "-"}</span>
              </p>
              <p className="flex items-center justify-between">
                <span className="text-slate-400">Camera</span>
                <span className="font-medium text-ink-900">{selected.camera || "-"}</span>
              </p>
              <p className="flex items-center justify-between">
                <span className="text-slate-400">Last seen on desk</span>
                <span className="font-medium text-ink-900">{selected.lastSeenDesk || "-"}</span>
              </p>
              <p className="flex items-center justify-between">
                <span className="text-slate-400">Confidence</span>
                <span className="font-medium text-ink-900">{selected.confidence ? `${selected.confidence}%` : "-"}</span>
              </p>
            </div>

            <button className="text-brand-600 text-sm font-medium flex items-center gap-1.5">
              <MapPin size={14} /> Live location <ChevronRight size={14} />
            </button>

            <button onClick={() => setDetailsOpen((o) => !o)} className="btn-secondary w-full text-sm">
              {detailsOpen ? "Hide details" : "Check details →"}
            </button>

            {detailsOpen && (
              <div className="space-y-2.5 text-sm border-t border-border-100 pt-4">
                <p className="flex items-center justify-between">
                  <span className="text-slate-400">{selected.type === "Guest" ? "Hosted by" : "Employee ID"}</span>
                  <span className="font-medium text-ink-900">
                    {selected.type === "Guest" ? selected.guestOf || "-" : selected.employeeId || "-"}
                  </span>
                </p>
                <p className="flex items-center justify-between">
                  <span className="text-slate-400">Face enrollment</span>
                  <StatusBadge value={selected.enrollment} />
                </p>
                <p className="flex items-center justify-between">
                  <span className="text-slate-400">Samples</span>
                  <span className="font-medium text-ink-900">{selected.designs ?? 0}</span>
                </p>
              </div>
            )}
          </div>
        )}
      </SidePanel>
    </div>
  );
}
