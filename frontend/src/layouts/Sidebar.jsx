import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Video,
  Bell,
  Users,
  CalendarCheck,
  Briefcase,
  Footprints,
  ShieldAlert,
  Camera,
  Building2,
} from "lucide-react";

const NAV_SECTIONS = [
  {
    items: [{ to: "/dashboard", label: "Dashboard", icon: LayoutDashboard }],
  },
  {
    title: "Monitoring",
    items: [
      { to: "/live-camera", label: "Live camera", icon: Video },
      { to: "/alerts", label: "Alerts & Events", icon: Bell, badge: 2 },
    ],
  },
  {
    title: "People",
    items: [
      { to: "/people", label: "People", icon: Users },
      { to: "/attendance", label: "Attendance", icon: CalendarCheck },
    ],
  },
  {
    title: "Analytics",
    items: [
      { to: "/workforce", label: "Workforce", icon: Briefcase },
      { to: "/footfall", label: "Footfall", icon: Footprints },
      { to: "/intrusion", label: "Intrusion", icon: ShieldAlert },
    ],
  },
  {
    title: "Management",
    items: [
      { to: "/cameras", label: "Camera", icon: Camera },
      { to: "/sites", label: "Site", icon: Building2 },
    ],
  },
];

export default function Sidebar() {
  return (
    <aside className="hidden md:flex md:w-60 shrink-0 flex-col bg-white text-slate-600 border-r border-border-200 h-screen sticky top-0">
      <div className="flex items-center gap-2 px-5 h-16 border-b border-border-100">
        <div className="w-7 h-7 rounded-md bg-brand-500 flex items-center justify-center text-white font-bold text-sm">
          D
        </div>
        <span className="text-ink-900 font-semibold tracking-tight">Deco Vision</span>
      </div>

      <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-5">
        {NAV_SECTIONS.map((section, i) => (
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
          <span className="w-7 h-7 rounded-full bg-brand-100 flex items-center justify-center text-xs font-semibold text-brand-700">
            JJ
          </span>
          <span>Settings</span>
        </NavLink>
      </div>
    </aside>
  );
}
