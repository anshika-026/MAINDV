import { createContext, useContext, useState, useCallback, useEffect, useRef } from "react";
import * as api from "../api/client";

const AuthContext = createContext(null);

// How often an already-logged-in client's session is re-checked against
// the live license (backend/app/license_routes.py's /client-me, which
// itself re-validates status/expiry on every call — see auth.py's
// load_active_client_license). This is what makes a suspend/feature/
// camera change an admin makes reach an already-open client tab without
// them needing to log out and back in, and is also how a suspended/
// expired license's access actually stops within a bounded time even if
// the client isn't clicking around triggering other API calls.
const CLIENT_SESSION_REFRESH_MS = 60000;

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem("deco_user");
    return saved ? JSON.parse(saved) : null;
  });
  const userRef = useRef(user);
  userRef.current = user;

  const login = useCallback(async (email, password) => {
    const res = await api.login(email, password);
    localStorage.setItem("deco_token", res.token);
    localStorage.setItem("deco_user", JSON.stringify(res.user));
    setUser(res.user);
    return res.user;
  }, []);

  function clientUserFrom(res) {
    return {
      role: "client",
      name: res.company_name || res.username,
      email: "",
      username: res.username,
      licenseId: res.license_id,
      companyId: res.company_id,
      companyName: res.company_name,
      companyPortalSlug: res.company_portal_slug,
      allowedCameras: res.cameras,
      allowedCameraIds: res.cameras.map((c) => c.id),
      allowedFeatures: res.features,
    };
  }

  // Client-portal login (license username/password, see api.clientLogin) —
  // a separate credential type from the admin login above, but stored the
  // same way (localStorage "deco_user" + this same `user` state) so the
  // rest of the app (ProtectedRoute, Topbar, Sidebar) doesn't need two
  // parallel session concepts. `role: "client"` plus the license's
  // assigned cameras/features is what Sidebar/LiveFeed/Topbar use to scope
  // what a client can see — see Sidebar.jsx and LiveFeed.jsx.
  const loginAsClient = useCallback(async (username, password) => {
    const res = await api.clientLogin(username, password);
    const clientUser = clientUserFrom(res);
    localStorage.setItem("deco_token", res.session_token);
    localStorage.setItem("deco_user", JSON.stringify(clientUser));
    setUser(clientUser);
    return clientUser;
  }, []);

  const signup = useCallback(async (payload) => {
    const res = await api.signup(payload);
    localStorage.setItem("deco_token", res.token);
    localStorage.setItem("deco_user", JSON.stringify(res.user));
    setUser(res.user);
    return res.user;
  }, []);

  const logout = useCallback(() => {
    // Best-effort server-side revoke — the local session clears either
    // way, but this stops the token from working if it leaked (e.g. was
    // logged, or the tab is shared) rather than just forgetting it here.
    if (userRef.current?.role === "client") api.clientLogout();
    else if (userRef.current) api.logoutAdmin();
    localStorage.removeItem("deco_token");
    localStorage.removeItem("deco_user");
    setUser(null);
  }, []);

  // Merges profile edits (Settings > Profile) into the logged-in user so the
  // Topbar/sidebar avatar and name reflect a save immediately, without
  // requiring a fresh login or page reload.
  const updateUser = useCallback((patch) => {
    setUser((prev) => {
      const next = { ...prev, ...patch };
      localStorage.setItem("deco_user", JSON.stringify(next));
      return next;
    });
  }, []);

  useEffect(() => {
    if (user?.role !== "client") return undefined;
    let cancelled = false;
    async function refresh() {
      try {
        const res = await api.clientMe();
        if (cancelled) return;
        updateUser(clientUserFrom(res));
      } catch {
        // A 401/403 here already triggered client.js's handleAuthFailure()
        // (clears storage + redirects) — nothing further to do.
      }
    }
    const interval = setInterval(refresh, CLIENT_SESSION_REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only re-arm when switching between client/non-client, not on every user-field change
  }, [user?.role]);

  return (
    <AuthContext.Provider value={{ user, login, loginAsClient, signup, logout, updateUser }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
