import { useEffect, useState } from "react";
import { Eye, ShieldAlert, UserCheck, Scale, ChevronRight, Radio } from "lucide-react";
import PageHeader from "../components/PageHeader";
import LiveCameraTile from "../components/LiveCameraTile";
import CameraViewerModal from "../components/CameraViewerModal";
import * as api from "../api/client";

const TABS = [
  { key: "all", label: "All cameras" },
  { key: "unsupervised", label: "Unsupervised" },
];

export default function LiveFeed() {
  const [cameras, setCameras] = useState([]);
  const [tab, setTab] = useState("all");
  const [activeInsight, setActiveInsight] = useState("unsupervised");
  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    // GET /api/cameras is scoped server-side now (backend/app/main.py):
    // an admin session gets every camera, a client session only their own
    // license's assigned cameras — trust the response directly rather
    // than re-filtering client-side against a locally cached allow-list,
    // which could go stale the moment an admin changes that assignment.
    api.getCameras().then(setCameras);
  }, []);

  const onlineCount = cameras.filter((c) => c.status === "Active").length;
  const offlineCount = cameras.length - onlineCount;
  // No supervision/intrusion/attendance analytics pipeline exists behind this
  // app yet, so these are honest placeholders derived from the real camera
  // list rather than fabricated backend numbers.
  const unsupervisedCameras = cameras.filter((c) => c.live !== "On");
  const insights = [
    { key: "unsupervised", icon: Eye, label: "Unsupervised", count: unsupervisedCameras.length },
    { key: "intrusion", icon: ShieldAlert, label: "Intrusion alert", count: offlineCount },
    { key: "attendance", icon: UserCheck, label: "Attendance sync", count: "Verify" },
    { key: "ratio", icon: Scale, label: "1:8 Ratio", count: cameras.length ? `1:${cameras.length}` : "-" },
  ];

  const visible = tab === "all" ? cameras : unsupervisedCameras;

  return (
    <div className="space-y-5">
      <PageHeader title="Vision" />

      <div className="card p-4 flex flex-wrap items-center gap-6">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 shrink-0 rounded-full bg-success-500 text-white flex items-center justify-center font-semibold shadow-sm">
            {onlineCount}/{cameras.length}
          </div>
          <div>
            <p className="text-sm font-semibold text-ink-900">Total Cameras</p>
            <div className="flex flex-wrap items-center gap-4 mt-1.5 text-xs text-slate-500">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-success-500" /> Online {onlineCount}
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-danger-500" /> Offline {offlineCount}
              </span>
              <span className="flex items-center gap-1.5">
                <Radio size={12} className="text-brand-500" /> Alerts generated {offlineCount}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="inline-flex items-center gap-1 rounded-full bg-white border border-border-200 p-1">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              tab === t.key ? "bg-brand-500 text-white" : "text-slate-500 hover:text-ink-900"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-[220px_1fr] gap-5 items-start">
        <div className="card p-4">
          <p className="text-sm font-semibold text-ink-900">Analytics</p>
          <p className="text-xs text-slate-400 mt-0.5 mb-3">Click on analytics to view</p>
          <div className="space-y-1">
            {insights.map((item) => {
              const Icon = item.icon;
              const isActive = activeInsight === item.key;
              return (
                <button
                  key={item.key}
                  onClick={() => setActiveInsight(item.key)}
                  className={`w-full flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm transition-colors ${
                    isActive ? "bg-brand-50 text-brand-600" : "text-ink-900 hover:bg-[#f5f6fa]"
                  }`}
                >
                  <Icon size={16} className={isActive ? "text-brand-500" : "text-slate-400"} />
                  <span className="flex-1 font-medium truncate">{item.label}</span>
                  {typeof item.count === "number" ? (
                    <span className="badge badge-neutral !px-2 !py-0.5 text-[11px]">{item.count}</span>
                  ) : (
                    <span className="text-xs text-slate-400">{item.count}</span>
                  )}
                  <ChevronRight size={14} className="text-slate-300 shrink-0" />
                </button>
              );
            })}
          </div>
        </div>

        <div>
          {visible.length === 0 ? (
            <div className="card p-10 text-center text-sm text-slate-400">
              {cameras.length === 0
                ? "No cameras yet — add one from Camera Management."
                : "No unsupervised cameras right now."}
            </div>
          ) : (
            <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
              {visible.map((cam) => (
                <LiveCameraTile key={cam.id} camera={cam} onClick={() => setExpanded(cam)} />
              ))}
            </div>
          )}
        </div>
      </div>

      <CameraViewerModal camera={expanded} onClose={() => setExpanded(null)} />
    </div>
  );
}
