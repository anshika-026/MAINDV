import { NavLink } from "react-router-dom";
import {
  Gauge,
  ScanEye,
  Siren,
  Fingerprint,
  UserCheck,
  TrendingUp,
  Footprints,
  ShieldX,
  Cctv,
  MapPinned,
  IdCard,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import Avatar from "../components/Avatar";

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
      { to: "/live-camera", label: "Vision", icon: ScanEye },
      { to: "/alerts", label: "Alerts & Events", icon: Siren, badge: 2 },
    ],
  },
  {
    title: "People",
    items: [
      { to: "/people", label: "Identity", icon: Fingerprint, requiresFeature: "face_recognition" },
      { to: "/attendance", label: "Presence", icon: UserCheck, requiresFeature: "attendance" },
    ],
  },
  {
    title: "Analytics",
    items: [
      { to: "/workforce", label: "Workforce Insights", icon: TrendingUp, requiresFeature: "workforce_analytics" },
      { to: "/footfall", label: "Footfall", icon: Footprints, requiresFeature: "footfall_analytics" },
      { to: "/intrusion", label: "Intrusion", icon: ShieldX, requiresFeature: "intrusion_detection" },
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

  const sections = NAV_SECTIONS.filter((s) => !(isClient && s.adminOnly))
    .map((s) => ({
      ...s,
      items: s.items.filter(
        (item) => !isClient || !item.requiresFeature || (user.allowedFeatures || []).includes(item.requiresFeature)
      ),
    }))
    .filter((s) => s.items.length > 0);

  return (
    <aside className="hidden md:flex md:w-60 shrink-0 flex-col bg-white text-slate-600 border-r border-border-200 h-screen sticky top-0">
      <div className="flex items-center gap-2 px-5 h-16 border-b border-border-100">
        <div className="w-7 h-7 rounded-md bg-brand-500 flex items-center justify-center text-white font-bold text-sm">
          D
        </div>
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
              {section.items.map(({ to, label, icon: Icon, badge }) => (
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
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

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
