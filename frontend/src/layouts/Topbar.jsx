import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Search, Bell, ChevronDown, Calendar } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function Topbar({ title }) {
  const { user, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const [query, setQuery] = useState("");
  const menuRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    function onClick(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  return (
    <header className="sticky top-0 z-20 flex items-center justify-between gap-4 h-16 px-6 bg-white border-b border-[#e7e8f0]">
      <div className="flex items-center gap-4 flex-1 min-w-0">
        {title && <h1 className="text-[15px] font-semibold text-ink-900 shrink-0">{title}</h1>}
        <div className="relative flex-1 max-w-md hidden sm:block">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for anything"
            className="input-field pl-9 py-2 bg-[#f5f6fa] border-transparent focus:bg-white"
          />
        </div>
      </div>

      <div className="flex items-center gap-3 shrink-0">
        <span className="badge badge-success hidden md:inline-flex">
          <span className="w-1.5 h-1.5 rounded-full bg-success-500" /> 3/3 Cameras online
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
                <p className="text-xs text-slate-500 truncate">{user?.email}</p>
              </div>
              <button
                onClick={() => { setMenuOpen(false); navigate("/settings/profile"); }}
                className="w-full text-left px-3 py-2 hover:bg-[#f5f6fa]"
              >
                Profile
              </button>
              <button
                onClick={() => { setMenuOpen(false); navigate("/settings/notifications"); }}
                className="w-full text-left px-3 py-2 hover:bg-[#f5f6fa]"
              >
                Notifications
              </button>
              <button
                onClick={() => { setMenuOpen(false); navigate("/settings/rules-policy"); }}
                className="w-full text-left px-3 py-2 hover:bg-[#f5f6fa]"
              >
                Rules &amp; Policy
              </button>
              <div className="border-t border-[#f1f2f7] mt-1 pt-1">
                <button
                  onClick={() => { setMenuOpen(false); logout(); navigate("/login"); }}
                  className="w-full text-left px-3 py-2 text-danger-600 hover:bg-danger-50"
                >
                  Log out
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
