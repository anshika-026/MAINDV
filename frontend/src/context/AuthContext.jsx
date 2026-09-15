import { createContext, useContext, useState, useCallback } from "react";
import * as api from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem("deco_user");
    return saved ? JSON.parse(saved) : null;
  });

  const login = useCallback(async (email, password) => {
    const res = await api.login(email, password);
    localStorage.setItem("deco_token", res.token);
    localStorage.setItem("deco_user", JSON.stringify(res.user));
    setUser(res.user);
    return res.user;
  }, []);

  const signup = useCallback(async (payload) => {
    const res = await api.signup(payload);
    localStorage.setItem("deco_token", res.token);
    localStorage.setItem("deco_user", JSON.stringify(res.user));
    setUser(res.user);
    return res.user;
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("deco_token");
    localStorage.removeItem("deco_user");
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
