import { useEffect, useMemo, useState } from "react";
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { UserPlus } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import * as api from "../api/client";
import { composition, compositionStats, footfallVisitorsExtra, hourlyTraffic } from "../data/footfallExtra";

const TONE_COLOR = {
  brand: "var(--color-brand-500)",
  success: "var(--color-success-500)",
  danger: "var(--color-danger-500)",
};

const FILTERS = ["All", "Unknown visitors", "Check enrollment"];
const ENROLLMENT_OPTIONS = ["All", "Enrolled", "Not enrolled", "Unknown"];

export default function Footfall() {
  const [stats, setStats] = useState(null);
  const [visitors, setVisitors] = useState([]);
  const [filter, setFilter] = useState("All");
  const [enrollmentPick, setEnrollmentPick] = useState("All");
  const [enrollmentMenuOpen, setEnrollmentMenuOpen] = useState(false);
  const [showYesterday, setShowYesterday] = useState(true);
  const [addedGuests, setAddedGuests] = useState([]);

  useEffect(() => {
    api.getFootfallStats().then(setStats);
    api.getFootfallVisitors().then((rows) => setVisitors([...rows, ...footfallVisitorsExtra]));
  }, []);

  const rows = useMemo(() => {
    let list = visitors;
    if (filter === "Unknown visitors") list = list.filter((v) => v.enrollment === "Unknown");
    if (filter === "Check enrollment" && enrollmentPick !== "All") {
      list = list.filter((v) => v.enrollment === enrollmentPick);
    }
    return list;
  }, [visitors, filter, enrollmentPick]);

  const totalComposition = composition.reduce((sum, c) => sum + c.value, 0);

  function quickAdd(row) {
    setAddedGuests((prev) => [...prev, row.person]);
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Footfall" />

      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="People Counted" value={stats.peopleCounted} sub="18 in / 3 out" subTone="neutral" />
          <StatCard label="Busiest Hour" value={stats.busiestHour} sub="by first-seen visits" subTone="neutral" />
          <StatCard label="Unknown Visitors" value={stats.unknownVisitors} sub="faces not recognized" subTone="danger" />
          <StatCard label="Current Occupancy" value={stats.currentOccupancy} sub="currently detected inside" subTone="neutral" />
        </div>
      )}

      <div className="grid lg:grid-cols-2 gap-5">
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-ink-900">Peak traffic time</h3>
            <label className="flex items-center gap-2 text-xs text-slate-500 cursor-pointer select-none">
              <span
                onClick={() => setShowYesterday((v) => !v)}
                className={`inline-block w-9 h-5 rounded-full transition-colors ${
                  showYesterday ? "bg-brand-500" : "bg-slate-300"
                } relative`}
              >
                <span
                  className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${
                    showYesterday ? "left-4.5" : "left-0.5"
                  }`}
                />
              </span>
              Yesterday
            </label>
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={hourlyTraffic}>
                <defs>
                  <linearGradient id="todayFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--color-brand-500)" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="var(--color-brand-500)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="#eceef4" />
                <XAxis dataKey="hour" tick={{ fontSize: 12, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                <YAxis hide />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="today"
                  name="Today"
                  stroke="var(--color-brand-500)"
                  strokeWidth={2}
                  fill="url(#todayFill)"
                />
                {showYesterday && (
                  <Area
                    type="monotone"
                    dataKey="yesterday"
                    name="Yesterday"
                    stroke="var(--color-muted-400)"
                    strokeWidth={2}
                    strokeDasharray="4 3"
                    fill="transparent"
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card p-5">
          <h3 className="font-semibold text-ink-900 mb-4">People Composition</h3>
          <div className="flex items-center gap-6">
            <div className="flex-1 space-y-3">
              <div className="flex h-3 w-full rounded-full overflow-hidden">
                {composition.map((c, i) => (
                  <div
                    key={c.label}
                    style={{
                      width: `${(c.value / totalComposition) * 100}%`,
                      background: TONE_COLOR[c.tone],
                      marginLeft: i === 0 ? 0 : 2,
                    }}
                  />
                ))}
              </div>
              <div className="space-y-2">
                {composition.map((c) => (
                  <div key={c.label} className="flex items-center gap-2 text-sm">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: TONE_COLOR[c.tone] }} />
                    <span className="text-slate-500 flex-1">{c.label}</span>
                    <span className="font-semibold text-ink-900">{c.value}%</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="w-px self-stretch bg-border-100" />
            <div className="space-y-4 text-sm shrink-0">
              <div>
                <p className="text-slate-400 text-xs">Top hour</p>
                <p className="font-semibold text-ink-900 mt-0.5">{compositionStats.topHour}</p>
              </div>
              <div>
                <p className="text-slate-400 text-xs">Unique visitors</p>
                <p className="font-semibold text-ink-900 mt-0.5">{compositionStats.uniqueVisitors}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-1">
        {FILTERS.map((f) => (
          <div key={f} className="relative">
            <button
              onClick={() => {
                setFilter(f);
                if (f === "Check enrollment") setEnrollmentMenuOpen((o) => !o);
                else setEnrollmentMenuOpen(false);
              }}
              className={`px-3.5 py-1.5 rounded-full text-sm font-medium transition-colors ${
                filter === f ? "bg-brand-500 text-white" : "bg-white border border-border-200 text-slate-600 hover:bg-[#f5f6fa]"
              }`}
            >
              {f}
              {f === "Check enrollment" ? " ▾" : ""}
            </button>
            {f === "Check enrollment" && enrollmentMenuOpen && (
              <div className="absolute z-10 top-full mt-1 card p-1.5 min-w-[160px]">
                {ENROLLMENT_OPTIONS.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => {
                      setEnrollmentPick(opt);
                      setEnrollmentMenuOpen(false);
                    }}
                    className={`w-full text-left px-2.5 py-1.5 rounded-lg text-sm ${
                      enrollmentPick === opt ? "bg-brand-50 text-brand-600" : "text-slate-600 hover:bg-[#f5f6fa]"
                    }`}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <DataTable
        columns={[
          { key: "person", label: "Person" },
          { key: "firstSeen", label: "First seen" },
          { key: "lastSeen", label: "Last seen" },
          { key: "camera", label: "Camera" },
          { key: "enrollment", label: "Enrollment", render: (r) => <StatusBadge value={r.enrollment} /> },
          {
            key: "action",
            label: "",
            render: (r) =>
              r.enrollment === "Unknown" ? (
                addedGuests.includes(r.person) ? (
                  <span className="text-xs text-slate-400">Added</span>
                ) : (
                  <button
                    onClick={() => quickAdd(r)}
                    className="text-brand-600 text-sm font-medium flex items-center gap-1"
                  >
                    <UserPlus size={13} /> Add person
                  </button>
                )
              ) : null,
          },
        ]}
        rows={rows}
      />
    </div>
  );
}
