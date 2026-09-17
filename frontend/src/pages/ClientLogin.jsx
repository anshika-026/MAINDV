import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import AuthLayout from "../layouts/AuthLayout";
import PasswordField from "../components/PasswordField";
import { useAuth } from "../context/AuthContext";
import * as api from "../api/client";

// Separate from the admin /login page on purpose (see AuthContext's
// loginAsClient) — a license's own username/password, so a client can
// open their portal from any device/browser instead of a device-bound
// key. Once signed in, they land on the same app (Sidebar/LiveFeed/Topbar
// scope what they see down to their license's assigned cameras/features).
//
// Mounted at two routes (App.jsx): the plain /client-login, and
// /client/:slug/login — the dev-mode stand-in for a real
// client-<slug>.decovision.com subdomain (see the deployment notes in
// the accompanying report). The slug only changes what name shows on
// this page; it never grants access by itself — the same username and
// password are always required, exactly as if there were no slug at all.
export default function ClientLogin() {
  const { loginAsClient } = useAuth();
  const navigate = useNavigate();
  const { slug } = useParams();
  const [companyName, setCompanyName] = useState(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!slug) return;
    api
      .getCompanyBySlug(slug)
      .then((c) => setCompanyName(c.name))
      .catch(() => setCompanyName(null)); // unknown slug — fall back to the generic heading below
  }, [slug]);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await loginAsClient(username.trim(), password);
      navigate("/dashboard");
    } catch (err) {
      setError(err.message || "Couldn't log in. Check your username and password and try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthLayout>
      <h1 className="text-lg font-semibold text-ink-900">
        {companyName ? `${companyName} Client Portal` : "Client portal login"}
      </h1>
      <p className="text-sm text-slate-500 mt-1 mb-6">Sign in with the username and password issued for your license</p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="text-sm font-medium text-ink-900 block mb-1.5">Username</label>
          <input
            required
            autoFocus
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Username"
            className="input-field"
          />
        </div>
        <div>
          <label className="text-sm font-medium text-ink-900 block mb-1.5">Password</label>
          <PasswordField
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
          />
        </div>

        {error && <p className="text-sm text-danger-500">{error}</p>}

        <button type="submit" disabled={loading} className="btn-primary w-full">
          {loading ? "Logging in..." : "Log in"}
        </button>
      </form>

      <p className="text-sm text-slate-500 text-center mt-6">
        Deco Vision admin?{" "}
        <Link to="/login" className="text-brand-600 font-medium">
          Log in here
        </Link>
      </p>
    </AuthLayout>
  );
}
