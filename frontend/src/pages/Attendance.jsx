import { useCallback, useEffect, useMemo, useState } from "react";
import { Calendar, Search } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import Modal from "../components/Modal";
import SidePanel from "../components/SidePanel";
import Avatar from "../components/Avatar";
import * as api from "../api/client";
// Real attendance, marked from face recognition (backend/app/attendance.py).
const ALL_COMPANIES = "All companies";
const statusFilters = ["All statuses", "Present", "On site", "Absent", "On Leave"];
const REFRESH_MS = 30000;

function todayIso() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function displayDateOf(iso) {
  return new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

export default function Attendance() {
  const [rows, setRows] = useState([]);
  const [stats, setStats] = useState(null);
  const [department, setDepartment] = useState(ALL_COMPANIES);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState(statusFilters[0]);
  const [search, setSearch] = useState("");
  const [dateOpen, setDateOpen] = useState(false);
  const [date, setDate] = useState(todayIso);

  const [selected, setSelected] = useState(null);
  const [historyOpen, setHistoryOpen] = useState(false);

  const [leaveOpen, setLeaveOpen] = useState(false);
  const [leaveForm, setLeaveForm] = useState({ employee: "", from: "", to: "", reason: "" });
  const [toast, setToast] = useState("");

  const load = useCallback(() => {
    api
      .getAttendanceDay(date)
      .then((data) => {
        setRows(
          data.rows.map((r) => ({
            id: r.employee_id,
            date: displayDateOf(data.date),
            employee: r.name,
            empId: r.employee_id,
            department: r.company,
            timeIn: r.time_in,
            timeOut: r.time_out,
            timeStay: r.time_stay,
            arrival: r.arrival,
            status: r.status,
            camera: r.camera,
            lastSeen: r.last_seen,
            confidence: r.confidence,
          }))
        );
        setStats(data.stats);
        setError("");
      })
      .catch(() => setError("Couldn't load attendance. Check the backend is running."));
  }, [date]);

  useEffect(() => {
    load();
    const t = setInterval(load, REFRESH_MS);
    return () => clearInterval(t);
  }, [load]);

  useEffect(() => {
    if (!selected) return;
    setHistory([]);
    api.getAttendanceHistory(selected.empId).then(setHistory).catch(() => setHistory([]));
  }, [selected]);

  const companies = useMemo(() => [ALL_COMPANIES, ...new Set(rows.map((r) => r.department))], [rows]);

  function showToast(msg) {
    setToast(msg);
    setTimeout(() => setToast(""), 2500);
  }

  const filtered = useMemo(
    () =>
      rows.filter((r) => {
        if (department !== ALL_COMPANIES && r.department !== department) return false;
        if (status !== statusFilters[0] && r.status !== status) return false;
        if (search && !r.employee.toLowerCase().includes(search.toLowerCase())) return false;
        return true;
      }),
    [rows, department, status, search]
  );

  const displayDate = displayDateOf(date);

  // Per product notes: HR approves and marks leave externally — this modal
  // just records that decision here so Attendance reflects it directly,
  // instead of that being tracked in a separate spreadsheet.
  async function handleMarkLeave(e) {
    e.preventDefault();
    const emp = rows.find((r) => r.employee === leaveForm.employee);
    if (!emp) {
      showToast("Pick an employee from the list");
      return;
    }
    try {
      await api.markLeave({ employeeId: emp.empId, from: leaveForm.from, to: leaveForm.to, reason: leaveForm.reason });
      showToast(`Leave marked for ${leaveForm.employee}`);
      setLeaveOpen(false);
      setLeaveForm({ employee: "", from: "", to: "", reason: "" });
      load();
    } catch (err) {
      showToast(err.message.includes("422") ? "The leave dates don't make sense. Check From and To." : "Couldn't save the leave.");
    }
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Presence"
        action={
          <button onClick={() => setLeaveOpen(true)} className="btn-secondary text-sm">
            Mark Leave
          </button>
        }
      />

      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="People Present" value={`${stats.present} of ${stats.total}`} sub="seen by face recognition" subTone="neutral" />
          <StatCard label="People Absent" value={stats.absent} subTone="danger" sub={stats.on_leave ? `${stats.on_leave} on leave` : ""} />
          <StatCard label="Attendance Percentage" value={`${stats.attendance_pct}%`} sub="" />
          <StatCard label="Late Arrivals" value={stats.late} subTone="danger" sub="" />
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <select value={department} onChange={(e) => setDepartment(e.target.value)} className="input-field w-auto">
          {companies.map((d) => (
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

      {error && <p className="text-sm text-danger-500">{error}</p>}

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
                <span className="badge badge-neutral mt-1">{selected.department}</span>
              </div>
            </div>

            <div className="space-y-2.5 text-sm">
              <p className="flex items-center justify-between">
                <span className="text-slate-400">Status</span>
                <StatusBadge value={selected.status} />
              </p>
              <p className="flex items-center justify-between">
                <span className="text-slate-400">Last seen by</span>
                <span className="font-medium text-ink-900">{selected.camera}</span>
              </p>
              <p className="flex items-center justify-between">
                <span className="text-slate-400">Last seen at</span>
                <span className="font-medium text-ink-900">{selected.lastSeen}</span>
              </p>
              <p className="flex items-center justify-between">
                <span className="text-slate-400">Confidence</span>
                <span className="font-medium text-ink-900">{selected.confidence}%</span>
              </p>
            </div>

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
                      {history.map((h) => (
                        <tr key={h.date}>
                          <td>{displayDateOf(h.date)}</td>
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
