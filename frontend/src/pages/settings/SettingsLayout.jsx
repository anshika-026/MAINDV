import { NavLink, Outlet } from "react-router-dom";

const TABS = [
  { to: "/settings/profile", label: "Profile" },
  { to: "/settings/notifications", label: "Notifications" },
  { to: "/settings/rules-policy", label: "Rules & Policy" },
  { to: "/settings/holiday-calendar", label: "Holiday Calendar" },
];

export default function SettingsLayout() {
  return (
    <div className="flex gap-8">
      <nav className="w-48 shrink-0 space-y-1">
        {TABS.map((t) => (
          <NavLink
            key={t.to}
            to={t.to}
            className={({ isActive }) =>
              `block px-3 py-2 rounded-lg text-sm font-medium ${
                isActive ? "bg-brand-50 text-brand-700" : "text-slate-600 hover:bg-[#f5f6fa]"
              }`
            }
          >
            {t.label}
          </NavLink>
        ))}
      </nav>
      <div className="flex-1 min-w-0 max-w-2xl">
        <Outlet />
      </div>
    </div>
  );
}
