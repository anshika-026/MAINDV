import { useEffect, useState } from "react";
import { AlertTriangle, Sparkles } from "lucide-react";
import * as api from "../api/client";
import StatCard from "../components/StatCard";
import { useAuth } from "../context/AuthContext";

export default function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState(null);

  useEffect(() => {
    api.getDashboardStats().then(setStats);
  }, []);

  if (!stats) return <p className="text-sm text-slate-400">Loading dashboard…</p>;

  const s = stats.admin;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-ink-900">
          Welcome, {user?.name?.split(" ")[0] || "Jay"}
        </h2>
        <p className="text-sm text-slate-500">
          {new Date().toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" })}
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
        <StatCard label="People Present" value={`${s.peoplePresent.value} of ${s.peoplePresent.of}`} sub={s.peoplePresent.sub} />
        <StatCard label="Footfall Today" value={s.footfallToday.value} sub={s.footfallToday.sub} />
        <StatCard label="Unknown Visitors" value={s.unknownVisitors.value} sub={s.unknownVisitors.sub} />
        <StatCard label="Cameras" value={s.camerasOnline.value} sub={s.camerasOnline.sub} subTone="danger" />
        <StatCard label="Guests Incoming" value={s.guestsIncoming.value} sub="" />
        <StatCard label="Active Alerts" value={s.activeAlerts.value} sub={s.activeAlerts.sub} subTone="danger" />
      </div>

      <div className="grid md:grid-cols-2 gap-5">
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-ink-900 flex items-center gap-2">
              <AlertTriangle size={16} className="text-danger-500" /> Needs attention
            </h3>
            <button className="text-xs text-brand-600 font-medium">View all</button>
          </div>
          <div className="space-y-3">
            {stats.needsAttention.map((item, i) => (
              <div key={i} className="flex items-center justify-between border border-[#f1f2f7] rounded-xl px-4 py-3">
                <div>
                  <p className="text-sm font-medium text-ink-900">{item.label}</p>
                  {item.detail && <p className="text-xs text-slate-500 mt-0.5">{item.detail}</p>}
                </div>
                <span
                  className={`badge ${item.tag === "Critical" ? "badge-danger" : "badge-warning"}`}
                >
                  {item.tag}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="card p-5">
          <h3 className="font-semibold text-ink-900 flex items-center gap-2 mb-4">
            <Sparkles size={16} className="text-brand-500" /> AI insights
          </h3>
          <ul className="space-y-3">
            {stats.aiInsights.map((line, i) => (
              <li key={i} className="text-sm text-slate-600 border border-[#f1f2f7] rounded-xl px-4 py-3">
                {line}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
