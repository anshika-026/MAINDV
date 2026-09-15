import { useEffect, useMemo, useState } from "react";
import { Calendar, MapPin, Search } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import Modal from "../components/Modal";
import SidePanel from "../components/SidePanel";
import Avatar from "../components/Avatar";
import * as api from "../api/client";
import { attendanceMeta, departments, statusFilters } from "../data/attendanceExtra";

export default function Attendance() {
  const [rows, setRows] = useState([]);
  const [stats, setStats] = useState(null);
  const [department, setDepartment] = useState(departments[0]);
  const [status, setStatus] = useState(statusFilters[0]);
  const [search, setSearch] = useState("");
  const [dateOpen, setDateOpen] = useState(false);
  const [date, setDate] = useState("2026-08-30");

  const [selected, setSelected] = useState(null);
  const [historyOpen, setHistoryOpen] = useState(false);

  const [leaveOpen, setLeaveOpen] = useState(false);
  const [leaveForm, setLeaveForm] = useState({ employee: "", from: "", to: "", reason: "" });
  const [toast, setToast] = useState("");

  useEffect(() => {
    api.getAttendance().then((data) =>
      setRows(data.map((r, i) => ({ ...r, ...(attendanceMeta[i] || {}), id: i })))
    );
    api.getAttendanceStats().then(setStats);
  }, []);

  function showToast(msg) {
    setToast(msg);
    setTimeout(() => setToast(""), 2500);
  }

  const filtered = useMemo(
    () =>
      rows.filter((r) => {
        if (department !== departments[0] && r.department !== department) return false;
        if (status !== statusFilters[0] && r.status !== status) return false;
        if (search && !r.employee.toLowerCase().includes(search.toLowerCase())) return false;
        return true;
      }),
    [rows, department, status, search]
  );

  const displayDate = new Date(`${date}T00:00:00`).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });

  // Per product notes: HR approves and marks leave externally — this modal
  // just records that decision here so Attendance reflects it directly,
  // instead of that being tracked in a separate spreadsheet.
  function handleMarkLeave(e) {
    e.preventDefault();
    setRows((prev) =>
      prev.map((r) =>
        r.employee === leaveForm.employee
          ? { ...r, status: "On Leave", timeIn: "-", timeOut: "-", timeStay: "-", arrival: "-" }
          : r
      )
    );
    showToast(`Leave marked for ${leaveForm.employee}`);
    setLeaveOpen(false);
    setLeaveForm({ employee: "", from: "", to: "", reason: "" });
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Attendance"
        action={
          <button onClick={() => setLeaveOpen(true)} className="btn-secondary text-sm">
            Mark Leave
          </button>
        }
      />

      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="People Present" value={`${stats.present} of ${stats.presentOf}`} sub="+2 vs yesterday" />
          <StatCard label="People Absent" value={stats.absent} subTone="danger" sub="-4 vs yesterday" />
          <StatCard label="Attendance Percentage" value={stats.attendancePct} sub="+2% vs yesterday" />
          <StatCard label="Late Arrivals" value={stats.lateArrivals} subTone="danger" sub="-2 vs yesterday" />
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <select value={department} onChange={(e) => setDepartment(e.target.value)} className="input-field w-auto">
          {departments.map((d) => (
            <option key={d}>{d}</option>
          ))}
        </select>
        <select value={status} onChange={(e) => setStatus(e.target.value)} className="input-field w-auto">
          {statusFilters.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>

        <div className="relative">
          <button onClick={() => setDateOpen((o) => !o)} className="btn-secondary text-sm flex items-center gap-2">
            <Calendar size={15} /> {displayDate}
          </button>
          {dateOpen && (
            <div className="absolute z-10 top-full mt-2 card p-3">
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="input-field"
              />
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
            placeholder="Search employee"
            className="input-field pl-9"
          />
        </div>
      </div>

      {toast && (
        <div className="text-sm text-success-600 bg-success-50 border border-success-500/20 rounded-lg px-3.5 py-2">
          {toast}
        </div>
      )}

      <DataTable
        columns={[
          { key: "date", label: "Date" },
          {
            key: "employee",
            label: "Employee",
            render: (r) => (
              <div className="flex items-center gap-2">
                <Avatar name={r.employee} size={26} />
                <span className="font-medium text-ink-900">{r.employee}</span>
              </div>
            ),
          },
          { key: "empId", label: "Emp ID" },
          { key: "timeIn", label: "Time in" },
          { key: "timeOut", label: "Time out" },
          { key: "timeStay", label: "Time stay" },
          { key: "arrival", label: "Arrival", render: (r) => <StatusBadge value={r.arrival} /> },
          { key: "status", label: "Status", render: (r) => <StatusBadge value={r.status} /> },
        ]}
        rows={filtered}
        onRowClick={setSelected}
      />

      <SidePanel open={!!selected} onClose={() => { setSelected(null); setHistoryOpen(false); }} title="Employee detail">
        {selected && (
          <div className="space-y-5">
            <div className="flex items-center gap-3">
              <Avatar name={selected.employee} size={48} />
              <div>
                <p className="font-semibold text-ink-900">{selected.employee}</p>
                <span className="badge badge-neutral mt-1">{selected.role || "Employee"}</span>
              </div>
            </div>

            <div className="space-y-2.5 text-sm">
              <p className="flex items-center justify-between">
                <span className="text-slate-400">Current location</span>
                <span className="font-medium text-ink-900">{selected.zone}</span>
              </p>
              <p className="flex items-center justify-between">
                <span className="text-slate-400">Camera</span>
                <span className="font-medium text-ink-900">{selected.camera}</span>
              </p>
              <p className="flex items-center justify-between">
                <span className="text-slate-400">Last seen on desk</span>
                <span className="font-medium text-ink-900">{selected.lastSeenDesk}</span>
              </p>
              <p className="flex items-center justify-between">
                <span className="text-slate-400">Confidence</span>
                <span className="font-medium text-ink-900">{selected.confidence}%</span>
              </p>
            </div>

            <button className="text-brand-600 text-sm font-medium flex items-center gap-1.5">
              <MapPin size={14} /> Live location →
            </button>

            <button
              onClick={() => setHistoryOpen((o) => !o)}
              className="btn-secondary w-full text-sm"
            >
              {historyOpen ? "Hide details" : "Check details →"}
            </button>

            {historyOpen && (
              <div>
                <p className="text-sm font-semibold text-ink-900 mb-2">Attendance history</p>
                <div className="card overflow-hidden">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(selected.history || []).map((h, i) => (
                        <tr key={i}>
                          <td>{h.date}</td>
                          <td><StatusBadge value={h.status} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </SidePanel>

      <Modal open={leaveOpen} onClose={() => setLeaveOpen(false)} title="Mark leave">
        <form onSubmit={handleMarkLeave} className="space-y-4">
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Employee</label>
            <input
              required
              list="attendance-employees"
              value={leaveForm.employee}
              onChange={(e) => setLeaveForm((f) => ({ ...f, employee: e.target.value }))}
              placeholder="Search employee by name"
              className="input-field"
            />
            <datalist id="attendance-employees">
              {rows.map((r) => (
                <option key={r.employee} value={r.employee} />
              ))}
            </datalist>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">From</label>
              <input
                type="date"
                required
                value={leaveForm.from}
                onChange={(e) => setLeaveForm((f) => ({ ...f, from: e.target.value }))}
                className="input-field"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">To</label>
              <input
                type="date"
                required
                value={leaveForm.to}
                onChange={(e) => setLeaveForm((f) => ({ ...f, to: e.target.value }))}
                className="input-field"
              />
            </div>
          </div>
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Reason</label>
            <textarea
              required
              rows={3}
              value={leaveForm.reason}
              onChange={(e) => setLeaveForm((f) => ({ ...f, reason: e.target.value }))}
              className="input-field resize-none"
              placeholder="e.g. Approved sick leave"
            />
          </div>
          <div className="flex items-center gap-3 pt-2">
            <button type="button" onClick={() => setLeaveOpen(false)} className="btn-secondary flex-1">
              Cancel
            </button>
            <button type="submit" className="btn-primary flex-1">
              Add
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
