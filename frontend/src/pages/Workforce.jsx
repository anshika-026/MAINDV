import { useCallback, useEffect, useMemo, useState } from "react";
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip, PieChart, Pie, Cell } from "recharts";
import { Search } from "lucide-react";
import PageHeader from "../components/PageHeader";
import Tabs from "../components/Tabs";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import Modal from "../components/Modal";
import Avatar from "../components/Avatar";
import * as api from "../api/client";
import DeskZoneEditor from "../components/DeskZoneEditor";
import { setVisibleInterval, clearVisibleInterval } from "../lib/visibleInterval";

function hms(seconds) {
  if (!seconds) return "0m";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}m`;
}

function clock(ts) {
  return ts ? new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "-";
}

export default function Workforce() {
  const [tab, setTab] = useState("Overview");
  const [stats, setStats] = useState(null);
  const [peopleAnalytics, setPeopleAnalytics] = useState([]);
  const [deskAnalytics, setDeskAnalytics] = useState([]);

  const [paSearch, setPaSearch] = useState("");

  const [zoneModalOpen, setZoneModalOpen] = useState(false);
  const [cameras, setCameras] = useState([]);
  const [deskError, setDeskError] = useState("");

  useEffect(() => {
    api.getCameras().then(setCameras).catch(() => setCameras([]));
  }, []);

  // Overview + People Analytics: real attendance data (backend/app/insights.py, attendance.py).
  const [paDate, setPaDate] = useState(() => new Date().toISOString().slice(0, 10));
  useEffect(() => {
    api.getWorkforceOverview().then(setStats).catch(() => setStats(null));
  }, []);
  useEffect(() => {
    api
      .getAttendanceDay(paDate)
      .then((d) =>
        setPeopleAnalytics(
          d.rows
            .filter((r) => r.sightings > 0)
            .map((r) => ({
              id: r.employee_id,
              name: r.name,
              company: r.company,
              lastSeen: r.camera,
              lastSeenAt: r.last_seen,
              firstSeen: r.time_in,
              status: r.status,
              clips: r.sightings,
            }))
        )
      )
      .catch(() => setPeopleAnalytics([]));
  }, [paDate]);

  // Real desk analytics (backend/app/desks.py), refreshed every 30 s.
  const loadDesks = useCallback(() => {
    api
      .getDeskReport()
      .then((r) => {
        setDeskAnalytics(
          r.employees.map((e) => ({
            id: e.employee_id,
            person: e.name,
            firstSeen: clock(e.first_session),
            lastSeen: clock(e.last_session),
            deskTime: hms(e.desk_seconds),
            awayTime: hms(e.away_seconds),
            currentDesk: e.current_desk,
            status: e.status,
            movements: e.movements,
          }))
        );
        setDeskError("");
      })
      .catch(() => setDeskError("Couldn't load desk analytics. Check the backend is running."));
  }, []);

  useEffect(() => {
    loadDesks();
    const t = setVisibleInterval(loadDesks, 30000);
    return () => clearVisibleInterval(t);
  }, [loadDesks]);

  const enrollmentSplit = useMemo(
    () => [
      { name: "Recognisable", value: stats?.enrollment.enrolled ?? 0, color: "var(--color-brand-500)" },
      { name: "No face photos yet", value: stats?.enrollment.not_enrolled ?? 0, color: "var(--color-border-300)" },
    ],
    [stats]
  );

  const filteredPeopleAnalytics = peopleAnalytics.filter((r) =>
    r.name.toLowerCase().includes(paSearch.toLowerCase())
  );

  function openZoneModal() {
    setZoneModalOpen(true);
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Workforce Insights" />
      <Tabs tabs={["Overview", "People Analytics", "Desk Analytics"]} active={tab} onChange={setTab} />

      {tab === "Overview" && stats && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Employees Present" value={`${stats.present} of ${stats.total_employees}`} sub={`${stats.on_site} on site now`} subTone="neutral" />
            <StatCard label="Left for Now" value={stats.exited} subTone="neutral" sub="seen today, not in the last 15 min" />
            <StatCard
              label="Attendance Percentage"
              value={`${stats.attendance_pct}%`}
              sub={`${stats.attendance_pct - stats.attendance_pct_yesterday >= 0 ? "+" : ""}${(stats.attendance_pct - stats.attendance_pct_yesterday).toFixed(1)}% vs yesterday`}
            />
            <StatCard label="Not Detected Today" value={stats.not_detected} subTone="danger" sub={stats.on_leave ? `${stats.on_leave} on leave` : ""} />
          </div>

          <div className="grid lg:grid-cols-2 gap-5">
            <div className="card p-5">
              <h3 className="font-semibold text-ink-900 mb-4">Employees present, last 7 days</h3>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={stats.trend}>
                    <XAxis dataKey="name" tick={{ fontSize: 12, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                    <YAxis hide allowDecimals={false} />
                    <Tooltip />
                    <Line type="monotone" dataKey="value" name="Present" stroke="var(--color-brand-500)" strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card p-5">
              <h3 className="font-semibold text-ink-900 mb-4">Face recognition coverage</h3>
              <div className="h-56 flex items-center">
                <div className="w-1/2 h-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={enrollmentSplit}
                        dataKey="value"
                        nameKey="name"
                        innerRadius={45}
                        outerRadius={70}
                        paddingAngle={3}
                        stroke="none"
                      >
                        {enrollmentSplit.map((s, i) => (
                          <Cell key={i} fill={s.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="w-1/2 space-y-3">
                  {enrollmentSplit.map((s) => (
                    <div key={s.name} className="flex items-center gap-2 text-sm">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ background: s.color }} />
                      <span className="text-slate-500 flex-1">{s.name}</span>
                      <span className="font-semibold text-ink-900">{s.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === "People Analytics" && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative flex-1 min-w-[200px] max-w-xs">
              <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                value={paSearch}
                onChange={(e) => setPaSearch(e.target.value)}
                placeholder="Search by name"
                className="input-field pl-9"
              />
            </div>
            <input type="date" value={paDate} onChange={(e) => setPaDate(e.target.value)} className="input-field w-auto" />
          </div>

          <DataTable
            columns={[
              { key: "photo", label: "Photo", render: (r) => <Avatar name={r.name} /> },
              { key: "name", label: "Name" },
              { key: "company", label: "Company" },
              { key: "firstSeen", label: "First seen" },
              { key: "lastSeen", label: "Last seen by" },
              { key: "lastSeenAt", label: "Last seen at" },
              { key: "clips", label: "Sightings" },
              { key: "status", label: "Status", render: (r) => <StatusBadge value={r.status} /> },
            ]}
            rows={filteredPeopleAnalytics}
            emptyLabel="Nobody recognised on this date."
          />
        </div>
      )}

      {tab === "Desk Analytics" && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-slate-500">
              Today, from face recognition at the desks you've marked. Someone not seen at a desk for 90 seconds counts as
              away.
            </p>
            <button onClick={openZoneModal} className="btn-secondary text-sm">
              Mark desks on camera
            </button>
          </div>
          {deskError && <p className="text-sm text-danger-500">{deskError}</p>}
          <DataTable
            columns={[
              { key: "person", label: "Person" },
              { key: "firstSeen", label: "First seen" },
              { key: "lastSeen", label: "Last seen" },
              { key: "deskTime", label: "Desk Time" },
              { key: "awayTime", label: "Away Time" },
              { key: "currentDesk", label: "Current desk" },
              { key: "status", label: "Status", render: (r) => <StatusBadge value={r.status} /> },
              { key: "movements", label: "Movements" },
            ]}
            rows={deskAnalytics}
            emptyLabel="Nobody tracked at a desk yet today. Mark desks on a camera, then check back in a few minutes."
          />
        </div>
      )}


      <Modal open={zoneModalOpen} onClose={() => setZoneModalOpen(false)} title="Mark desks on camera" width="max-w-3xl">
        {zoneModalOpen && <DeskZoneEditor cameras={cameras} onChanged={loadDesks} />}
      </Modal>
    </div>
  );
}
