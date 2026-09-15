import { Outlet, useLocation } from "react-router-dom";
import Sidebar from "./Sidebar";
import Topbar from "./Topbar";

const TITLES = {
  "/dashboard": "Dashboard",
  "/live-camera": "Live feed",
  "/alerts": "Alerts",
  "/people": "People",
  "/attendance": "Attendance",
  "/workforce": "Workforce",
  "/footfall": "Footfall",
  "/intrusion": "Intrusion",
  "/cameras": "Camera Management",
  "/sites": "Site Management",
  "/settings/profile": "Settings",
  "/settings/notifications": "Settings",
  "/settings/rules-policy": "Settings",
};

export default function AppShell() {
  const { pathname } = useLocation();
  const title = TITLES[pathname];

  return (
    <div className="flex min-h-screen bg-[#f5f6fa]">
      <Sidebar />
      <div className="flex-1 min-w-0 flex flex-col">
        <Topbar title={title} />
        <main className="flex-1 p-6 max-w-[1400px] w-full mx-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
