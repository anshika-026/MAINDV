import { useEffect, useState } from "react";
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { DoorOpen } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import * as api from "../api/client";
import { setVisibleInterval, clearVisibleInterval } from "../lib/visibleInterval";

// Real unique footfall across every entry gate (backend/app/footfall.py):
// all gates share one identity gallery, so a person who enters at one gate
// and leaves or re-enters through another is still counted once.

const REFRESH_MS = 30_000;

function hourLabel(h) {
  if (h === null || h === undefined) return "—";
  const suffix = h < 12 ? "am" : "pm";
  const h12 = h % 12 === 0 ? 12 : h % 12;
  return `${h12}${suffix}`;
}

function timeLabel(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export default function Footfall() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [showYesterday, setShowYesterday] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      api
        .getFootfallSummary()
        .then((d) => {
          if (!cancelled) {
            setData(d);
            setError("");
          }
        })
        .catch(() => !cancelled && setError("Couldn't load footfall — is the backend running?"));
    load();
    const id = setVisibleInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearVisibleInterval(id);
    };
  }, []);

  if (error && !data) return <p className="text-sm text-danger-500">{error}</p>;
  if (!data) return <p className="text-sm text-slate-400">Loading footfall…</p>;

  const hourly = data.hourly.map((r) => ({ ...r, hour: hourLabel(r.hour) }));
  const maxGate = Math.max(1, ...data.gates.map((g) => g.unique_today));

  return (
    <div className="space-y-5">
      <PageHeader title="Footfall" />

      {!data.model_ready && (
        <p className="card px-4 py-3 text-sm text-danger-500">
          The Re-ID model isn't installed, so counts will run high. Run{" "}
          <code>python -m scripts.fetch_reid_model</code> in <code>backend/</code> and restart the backend.
        </p>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Unique Footfall Today"
          value={data.unique_today}
          sub={`each person once, across ${data.gates.length} gate${data.gates.length === 1 ? "" : "s"}`}
          subTone="neutral"
        />
        <StatCard label="Average per Gate" value={data.avg_per_gate} sub="unique people seen at each gate" subTone="neutral" />
        <StatCard label="Busiest Hour" value={hourLabel(data.busiest_hour)} sub="by first arrival" subTone="neutral" />
        <StatCard
          label="Returning Today"
          value={data.returning_today}
          sub={`${data.new_today} first-time visitors`}
          subTone="neutral"
        />
      </div>

      <div className="grid lg:grid-cols-2 gap-5">
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-ink-900">Arrivals by hour</h3>
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
              <AreaChart data={hourly}>
                <defs>
                  <linearGradient id="todayFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--color-brand-500)" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="var(--color-brand-500)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="#eceef4" />
                <XAxis dataKey="hour" tick={{ fontSize: 12, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                <YAxis hide allowDecimals={false} />
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
          <h3 className="font-semibold text-ink-900 mb-1">By gate</h3>
          <p className="text-xs text-slate-400 mb-4">
            A person seen at two gates shows under both, so these add up to more than the unique total.
          </p>
          {data.gates.length === 0 ? (
            <p className="text-sm text-slate-500">
              No gate cameras yet. Set a camera's purpose to <b>Entry/Exit</b> in Camera Management.
            </p>
          ) : (
            <div className="space-y-3">
              {data.gates.map((g) => (
                <div key={g.camera_id} className="text-sm">
                  <div className="flex items-center gap-2 mb-1">
                    <DoorOpen size={14} className="text-slate-400" />
                    <span className="text-ink-900 flex-1">{g.name}</span>
                    {!g.counting && <span className="badge badge-warning">not counting</span>}
                    <span className="font-semibold text-ink-900 w-10 text-right">{g.unique_today}</span>
                  </div>
                  <div className="h-2 rounded-full bg-[#f1f2f7] overflow-hidden">
                    <div
                      className="h-full rounded-full bg-brand-500"
                      style={{ width: `${(g.unique_today / maxGate) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <DataTable
        columns={[
          { key: "person", label: "Person" },
          { key: "first_seen", label: "First seen", render: (r) => timeLabel(r.first_seen) },
          { key: "last_seen", label: "Last seen", render: (r) => timeLabel(r.last_seen) },
          { key: "gates", label: "Gates", render: (r) => r.gates.join(", ") },
        ]}
        rows={data.visitors}
        emptyLabel="Nobody counted yet today"
      />
    </div>
  );
}
