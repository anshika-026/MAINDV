import { useEffect, useMemo, useRef, useState } from "react";
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip, PieChart, Pie, Cell } from "recharts";
import { Search } from "lucide-react";
import PageHeader from "../components/PageHeader";
import Tabs from "../components/Tabs";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import Modal from "../components/Modal";
import SidePanel from "../components/SidePanel";
import Avatar from "../components/Avatar";
import * as api from "../api/client";
import {
  deskAnalyticsExtra,
  initialDeskZones,
  peopleAnalyticsExtra,
  weeklyAttendanceTrend,
} from "../data/workforceExtra";
import * as mock from "../data/mockData";

// Bright, non-brand accent used ONLY for the transient rectangle the user is
// actively drawing on the camera-frame placeholder — a deliberate one-off so
// the in-progress selection reads clearly against the dark frame; it's not
// part of the reusable design-system palette.
const DRAW_ACCENT = "#ec4899";

function DeskZoneDrawer({ rect, onRectChange }) {
  const containerRef = useRef(null);
  const drawingRef = useRef(false);
  const startRef = useRef({ x: 0, y: 0 });

  function pos(e) {
    const b = containerRef.current.getBoundingClientRect();
    return { x: e.clientX - b.left, y: e.clientY - b.top };
  }
  function onMouseDown(e) {
    const p = pos(e);
    startRef.current = p;
    drawingRef.current = true;
    onRectChange({ x: p.x, y: p.y, w: 0, h: 0 });
  }
  function onMouseMove(e) {
    if (!drawingRef.current) return;
    const p = pos(e);
    const s = startRef.current;
    onRectChange({ x: Math.min(s.x, p.x), y: Math.min(s.y, p.y), w: Math.abs(p.x - s.x), h: Math.abs(p.y - s.y) });
  }
  function stop() {
    drawingRef.current = false;
  }

  return (
    <div
      ref={containerRef}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={stop}
      onMouseLeave={stop}
      className="relative w-full h-56 rounded-xl bg-ink-900 overflow-hidden select-none cursor-crosshair"
    >
      <p className="absolute top-3 left-3 text-xs text-white/50">
        Camera frame — click and drag to mark a desk zone
      </p>
      {rect && rect.w > 0 && rect.h > 0 && (
        <div
          className="absolute pointer-events-none"
          style={{
            left: rect.x,
            top: rect.y,
            width: rect.w,
            height: rect.h,
            border: `2px solid ${DRAW_ACCENT}`,
            background: "rgba(236,72,153,0.15)",
          }}
        />
      )}
    </div>
  );
}

export default function Workforce() {
  const [tab, setTab] = useState("Overview");
  const [stats, setStats] = useState(null);
  const [peopleAnalytics, setPeopleAnalytics] = useState([]);
  const [deskAnalytics, setDeskAnalytics] = useState([]);

  const [paSearch, setPaSearch] = useState("");
  const [viewingClip, setViewingClip] = useState(null);

  const [zoneModalOpen, setZoneModalOpen] = useState(false);
  const [deskZones, setDeskZones] = useState(initialDeskZones);
  const [drawRect, setDrawRect] = useState(null);
  const [deskForm, setDeskForm] = useState({ name: "", type: "Entry" });

  useEffect(() => {
    api.getWorkforceStats().then(setStats);
    api.getWorkforcePeopleAnalytics().then((rows) =>
      setPeopleAnalytics([...rows, ...peopleAnalyticsExtra])
    );
    api.getDeskAnalytics().then((rows) => setDeskAnalytics([...rows, ...deskAnalyticsExtra]));
  }, []);

  const enrollmentSplit = useMemo(() => {
    const enrolled = mock.people.filter((p) => p.faceEnrolled).length;
    const notEnrolled = mock.people.length - enrolled;
    return [
      { name: "Enrolled", value: enrolled, color: "var(--color-brand-500)" },
      { name: "Not enrolled", value: notEnrolled, color: "var(--color-border-300)" },
    ];
  }, []);

  const filteredPeopleAnalytics = peopleAnalytics.filter((r) =>
    r.name.toLowerCase().includes(paSearch.toLowerCase())
  );

  function openZoneModal() {
    setDrawRect(null);
    setDeskForm({ name: "", type: "Entry" });
    setZoneModalOpen(true);
  }

  function handleAddDesk(e) {
    e.preventDefault();
    if (!deskForm.name) return;
    setDeskZones((prev) => [...prev, { name: deskForm.name, type: deskForm.type }]);
    setDeskAnalytics((prev) => [
      ...prev,
      {
        person: "—",
        firstSeen: "—",
        lastSeen: "—",
        deskTime: "—",
        awayTime: "—",
        currentDesk: deskForm.name,
        status: "Unassigned",
        movements: 0,
      },
    ]);
    setZoneModalOpen(false);
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Workforce" />
      <Tabs tabs={["Overview", "People Analytics", "Desk Analytics"]} active={tab} onChange={setTab} />

      {tab === "Overview" && stats && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Total Employee" value={stats.totalEmployees} sub="Since morning" subTone="neutral" />
            <StatCard label="Employee Exited" value={stats.employeeExited} subTone="danger" sub="+1 vs yesterday" />
            <StatCard label="Attendance Percentage" value={stats.attendancePct} sub="+2% vs yesterday" />
            <StatCard label="Employee not detected" value={stats.notDetected} subTone="danger" sub="Since morning" />
          </div>

          <div className="grid lg:grid-cols-2 gap-5">
            <div className="card p-5">
              <h3 className="font-semibold text-ink-900 mb-4">Attendance Trend</h3>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={weeklyAttendanceTrend}>
                    <XAxis dataKey="name" tick={{ fontSize: 12, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                    <YAxis hide />
                    <Tooltip />
                    <Line type="monotone" dataKey="value" stroke="var(--color-brand-500)" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card p-5">
              <h3 className="font-semibold text-ink-900 mb-4">People Data Overview</h3>
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
            <input type="date" defaultValue="2026-08-28" className="input-field w-auto" />
          </div>

          <DataTable
            columns={[
              { key: "photo", label: "Photo", render: (r) => <Avatar name={r.name} /> },
              { key: "name", label: "Name" },
              { key: "lastSeen", label: "Last seen" },
              { key: "lastSeenAt", label: "Last seen at" },
              { key: "clips", label: "Clips" },
              {
                key: "action",
                label: "Action",
                render: (r) => (
                  <button onClick={() => setViewingClip(r)} className="text-brand-600 text-sm font-medium">
                    View
                  </button>
                ),
              },
            ]}
            rows={filteredPeopleAnalytics}
          />
        </div>
      )}

      {tab === "Desk Analytics" && (
        <div className="space-y-4">
          <div className="flex justify-end">
            <button onClick={openZoneModal} className="btn-secondary text-sm">
              + Draw desk zone
            </button>
          </div>
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
          />
        </div>
      )}

      <SidePanel open={!!viewingClip} onClose={() => setViewingClip(null)} title="Clip preview">
        {viewingClip && (
          <div className="space-y-4">
            <div className="w-full aspect-video rounded-xl border border-border-200 bg-[#f1f2f7] flex items-center justify-center text-sm text-slate-400">
              Still frame — {viewingClip.lastSeen}
            </div>
            <div>
              <p className="font-semibold text-ink-900">{viewingClip.name}</p>
              <p className="text-sm text-slate-500 mt-0.5">
                Last seen at {viewingClip.lastSeen} · {viewingClip.lastSeenAt}
              </p>
              <p className="text-xs text-slate-400 mt-1">{viewingClip.clips} clips recorded</p>
            </div>
            <button onClick={() => setViewingClip(null)} className="btn-secondary w-full">
              Close
            </button>
          </div>
        )}
      </SidePanel>

      <Modal open={zoneModalOpen} onClose={() => setZoneModalOpen(false)} title="Draw desk zone" width="max-w-lg">
        <form onSubmit={handleAddDesk} className="space-y-4">
          <DeskZoneDrawer rect={drawRect} onRectChange={setDrawRect} />

          {deskZones.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {deskZones.map((z) => (
                <span key={z.name} className="badge badge-neutral">
                  {z.name} · {z.type}
                </span>
              ))}
            </div>
          )}

          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Desk name</label>
            <input
              required
              value={deskForm.name}
              onChange={(e) => setDeskForm((f) => ({ ...f, name: e.target.value }))}
              placeholder="e.g. Desk 12"
              className="input-field"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Zone type</label>
            <select
              value={deskForm.type}
              onChange={(e) => setDeskForm((f) => ({ ...f, type: e.target.value }))}
              className="input-field"
            >
              <option>Entry</option>
              <option>Exit</option>
            </select>
          </div>
          <div className="flex items-center gap-3 pt-2">
            <button type="button" onClick={() => setZoneModalOpen(false)} className="btn-secondary flex-1">
              Cancel
            </button>
            <button type="submit" className="btn-primary flex-1">
              Add desk
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
