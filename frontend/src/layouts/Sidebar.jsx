import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import {
  Gauge,
  ScanEye,
  MonitorPlay,
  Siren,
  Fingerprint,
  UserCheck,
  TrendingUp,
  Footprints,
  FlaskConical,
  Users,
  ShieldX,
  Cctv,
  MapPinned,
  IdCard,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import * as api from "../api/client";
import Avatar from "../components/Avatar";
import BrandLogo from "../components/BrandLogo";

// `requiresFeature` gates a nav item for a client login (user.role ===
// "client", see AuthContext.loginAsClient) to only what their license has
// enabled (backend/app/license_db.py's ALL_FEATURES) — an admin login
// always sees everything, feature keys are ignored for them. Live camera,
// Alerts and Dashboard have no feature of their own (base capability),
// so they stay visible to any logged-in client regardless of features.
const NAV_SECTIONS = [
  {
    items: [{ to: "/dashboard", label: "Dashboard", icon: Gauge }],
  },
  {
    title: "Monitoring",
    items: [
      { to: "/live-feed", label: "Live Feed", icon: MonitorPlay },
      { to: "/live-camera", label: "AI Analytics", icon: ScanEye, analytics: "live_overlay" },
      { to: "/alerts", label: "Alerts & Events", icon: Siren, badgeKey: "alerts" },
    ],
  },
  {
    title: "People",
    items: [
      { to: "/people", label: "Identity", icon: Fingerprint, requiresFeature: "face_recognition" },
      { to: "/attendance", label: "Presence", icon: UserCheck, requiresFeature: "attendance", analytics: "face_recognition" },
      { to: "/staff", label: "Staff Count", icon: Users, requiresFeature: "attendance", analytics: "staff_count" },
    ],
  },
  {
    title: "Analytics",
    items: [
      { to: "/workforce", label: "Workforce Insights", icon: TrendingUp, requiresFeature: "workforce_analytics", analytics: "desk_analytics" },
      { to: "/footfall", label: "Footfall", icon: Footprints, requiresFeature: "footfall_analytics", analytics: "footfall" },
      { to: "/footfall-uat", label: "Footfall UAT", icon: FlaskConical, requiresFeature: "footfall_analytics", adminOnly: true },
      { to: "/intrusion", label: "Intrusion", icon: ShieldX, requiresFeature: "intrusion_detection", analytics: "intrusion" },
    ],
  },
  {
    title: "Management",
    adminOnly: true, // managing other companies' cameras/sites/licenses isn't part of a client's own portal
    items: [
      { to: "/cameras", label: "Camera", icon: Cctv },
      { to: "/sites", label: "Site", icon: MapPinned },
      { to: "/licenses", label: "Client License", icon: IdCard },
    ],
  },
];

export default function Sidebar() {
  const { user } = useAuth();
  const isClient = user?.role === "client";
  // Live badge counts (currently: active alerts), refreshed every 30 s.
  const [counts, setCounts] = useState({});
  useEffect(() => {
    if (isClient) return;
    const load = () => api.getAlertsSummary().then((s) => setCounts({ alerts: s.active })).catch(() => {});
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, [isClient]);

  // Analytics on/off switches + live CPU (backend/app/analytics_settings.py).
  const [switches, setSwitches] = useState(null);
  const [cpu, setCpu] = useState(null);
  const [busy, setBusy] = useState(null);
  useEffect(() => {
    if (isClient) return;
    const load = () =>
      api
        .getAnalyticsSettings()
        .then((r) => {
          setSwitches(r.features);
          setCpu(r.cpu);
        })
        .catch(() => {});
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [isClient]);

  async function toggle(e, feature) {
    e.preventDefault();
    e.stopPropagation();
    if (!switches || busy) return;
    setBusy(feature);
    try {
      const r = await api.setAnalyticsFeature(feature, !switches[feature].on);
      setSwitches(r.features);
    } finally {
      setBusy(null);
    }
  }

  const sections = NAV_SECTIONS.filter((s) => !(isClient && s.adminOnly))
    .map((s) => ({
      ...s,
      items: s.items.filter(
        (item) =>
          !(isClient && item.adminOnly) &&
          (!isClient || !item.requiresFeature || (user.allowedFeatures || []).includes(item.requiresFeature))
      ),
    }))
    .filter((s) => s.items.length > 0);

  return (
    <aside className="hidden md:flex md:w-60 shrink-0 flex-col bg-white text-slate-600 border-r border-border-200 h-screen sticky top-0">
      <div className="flex items-center gap-2 px-5 h-16 border-b border-border-100">
        <BrandLogo size={28} />
        <span className="text-ink-900 font-semibold tracking-tight">Deco Vision</span>
      </div>

      <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-5">
        {sections.map((section, i) => (
          <div key={i}>
            {section.title && (
              <p className="px-2 mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                {section.title}
              </p>
            )}
            <div className="space-y-0.5">
              {section.items.map(({ to, label, icon: Icon, badgeKey, analytics }) => {
                const badge = badgeKey ? counts[badgeKey] : null;
                const sw = analytics && switches ? switches[analytics] : null;
                const blockedBy = sw && sw.on && !sw.effective ? switches[sw.requires]?.label : null;
                return (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                      isActive
                        ? "bg-brand-500 text-white"
                        : "text-slate-600 hover:bg-[#f4f5f9] hover:text-ink-900"
                    }`
                  }
                >
                  <Icon size={17} strokeWidth={1.8} />
                  <span className="flex-1">{label}</span>
                  {badge ? (
                    <span className="text-[11px] font-semibold bg-danger-500 text-white rounded-full px-1.5 py-0.5 leading-none">
                      {badge}
                    </span>
                  ) : null}
                  {sw && (
                    <span
                      role="switch"
                      tabIndex={0}
                      aria-checked={sw.on}
                      aria-label={`${sw.label}: ${sw.on ? "on" : "off"}`}
                      title={
                        blockedBy
                          ? `${sw.label} is on but paused: it needs ${blockedBy}`
                          : `${sw.label} is ${sw.on ? "on" : "off"}. Click to turn it ${sw.on ? "off" : "on"}.`
                      }
                      onClick={(e) => toggle(e, analytics)}
                      onKeyDown={(e) => (e.key === " " || e.key === "Enter") && toggle(e, analytics)}
                      className={`relative inline-block w-8 h-[18px] rounded-full shrink-0 transition-colors cursor-pointer ring-1 ring-white/40 ${
                        sw.effective ? "bg-success-500" : sw.on ? "bg-warning-500" : "bg-slate-300"
                      } ${busy === analytics ? "opacity-60" : ""}`}
                    >
                      <span
                        className={`absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white shadow transition-all ${
                          sw.on ? "left-[16px]" : "left-[2px]"
                        }`}
                      />
                    </span>
                  )}
                </NavLink>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {cpu && cpu.backend_pct !== undefined && (
        <div className="px-5 py-2 border-t border-border-100 text-[11px] text-slate-500 space-y-0.5" title="Analytics switches above turn processing on or off to manage this load">
          <p className="flex justify-between">
            <span>Backend CPU</span>
            <span className={`font-semibold ${cpu.backend_pct > 70 ? "text-danger-500" : "text-ink-900"}`}>{cpu.backend_pct}%</span>
          </p>
          <p className="flex justify-between">
            <span>Whole laptop</span>
            <span className={`font-semibold ${cpu.system_pct > 85 ? "text-danger-500" : "text-ink-900"}`}>{Math.round(cpu.system_pct)}%</span>
          </p>
          <p className="flex justify-between">
            <span>Free memory</span>
            <span className={`font-semibold ${cpu.memory_free_gb < 1.5 ? "text-danger-500" : "text-ink-900"}`}>
              {cpu.memory_free_gb} / {cpu.memory_total_gb} GB
            </span>
          </p>
        </div>
      )}

      <div className="p-3 border-t border-border-100">
        <NavLink
          to="/settings/profile"
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
              isActive ? "bg-brand-500 text-white" : "text-slate-600 hover:bg-[#f4f5f9] hover:text-ink-900"
            }`
          }
        >
          <Avatar name={user?.name || user?.username} size={28} />
          <span>Settings</span>
        </NavLink>
      </div>
    </aside>
  );
}
