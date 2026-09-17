import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Bell, ChevronDown, Calendar, User, ShieldCheck, LogOut } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import * as api from "../api/client";

// Refreshed periodically (not just once) since Topbar stays mounted across
// every page navigation — the badge should reflect the fleet's current
// state, not just whatever it was when the app first loaded.
const CAMERA_REFRESH_MS = 30000;

export default function Topbar({ title }) {
  const { user, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [cameraStats, setCameraStats] = useState({ online: 0, total: 0 });
  const menuRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    function onClick(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  useEffect(() => {
    let cancelled = false;
    function refresh() {
      // GET /api/cameras is scoped server-side (backend/app/main.py) — a
      // client session only ever gets their own license's cameras back,
      // so this count is already correct without re-filtering here.
      api.getCameras().then((cams) => {
        if (cancelled) return;
        setCameraStats({ online: cams.filter((c) => c.status === "Active").length, total: cams.length });
      });
    }
    refresh();
    const interval = setInterval(refresh, CAMERA_REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const allOnline = cameraStats.total > 0 && cameraStats.online === cameraStats.total;
  const cameraBadgeTone = cameraStats.total === 0 ? "badge-neutral" : allOnline ? "badge-success" : "badge-warning";
  const cameraDotTone = cameraStats.total === 0 ? "bg-slate-400" : allOnline ? "bg-success-500" : "bg-warning-500";

  return (
    <header className="sticky top-0 z-20 flex items-center justify-between gap-4 h-16 px-6 bg-white border-b border-[#e7e8f0]">
      <div className="flex items-center gap-4 flex-1 min-w-0">
        {title && <h1 className="text-[15px] font-semibold text-ink-900 shrink-0">{title}</h1>}
        <div className="relative flex-1 max-w-md hidden sm:block">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for anything"
            className="input-field py-2 bg-[#f5f6fa] border-transparent focus:bg-white"
          />
        </div>
      </div>

      <div className="flex items-center gap-3 shrink-0">
        <span className={`badge ${cameraBadgeTone} hidden md:inline-flex`}>
          <span className={`w-1.5 h-1.5 rounded-full ${cameraDotTone}`} /> {cameraStats.online}/{cameraStats.total}{" "}
          Cameras online
        </span>
        <button className="btn-secondary !py-2 !px-3 hidden sm:inline-flex items-center gap-1.5 text-sm">
          <Calendar size={14} /> Today
        </button>
        <button className="w-9 h-9 rounded-full border border-[#e7e8f0] flex items-center justify-center text-slate-500 hover:bg-[#f5f6fa]">
          <Bell size={16} />
        </button>
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setMenuOpen((v) => !v)}
            className="flex items-center gap-2 pl-1 pr-2 py-1 rounded-full hover:bg-[#f5f6fa]"
          >
            <span className="w-8 h-8 rounded-full bg-brand-500 text-white flex items-center justify-center text-xs font-semibold">
              {(user?.name || "JJ")
                .split(" ")
                .map((n) => n[0])
                .join("")
                .slice(0, 2)}
            </span>
            <ChevronDown size={14} className="text-slate-400 hidden sm:block" />
          </button>
          {menuOpen && (
            <div className="absolute right-0 mt-2 w-48 bg-white border border-[#e7e8f0] rounded-xl shadow-lg py-1.5 text-sm">
              <div className="px-3 py-2 border-b border-[#f1f2f7]">
                <p className="font-medium text-ink-900 truncate">{user?.name || "Jay Jain"}</p>
                <p className="text-xs text-slate-500 truncate">{user?.email || user?.username}</p>
              </div>
              <button
                onClick={() => { setMenuOpen(false); navigate("/settings/profile"); }}
                className="w-full flex items-center gap-2.5 text-left px-3 py-2 hover:bg-[#f5f6fa]"
              >
                <User size={15} className="text-slate-400" /> Profile
              </button>
              <button
                onClick={() => { setMenuOpen(false); navigate("/settings/notifications"); }}
                className="w-full flex items-center gap-2.5 text-left px-3 py-2 hover:bg-[#f5f6fa]"
              >
                <span className="relative">
                  <Bell size={15} className="text-slate-400" />
                  <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-danger-500" />
                </span>
                Notifications
              </button>
              <button
                onClick={() => { setMenuOpen(false); navigate("/settings/rules-policy"); }}
                className="w-full flex items-center gap-2.5 text-left px-3 py-2 hover:bg-[#f5f6fa]"
              >
                <ShieldCheck size={15} className="text-slate-400" /> Rules &amp; Policy
              </button>
              <div className="border-t border-[#f1f2f7] mt-1 pt-1">
                <button
                  onClick={() => { setMenuOpen(false); logout(); navigate("/login"); }}
                  className="w-full flex items-center gap-2.5 text-left px-3 py-2 text-danger-600 hover:bg-danger-50"
                >
                  <LogOut size={15} /> Log out
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
