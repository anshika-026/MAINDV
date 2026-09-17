import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

// Wraps ProtectedRoute's job with one more check: a license/client login
// (see AuthContext.loginAsClient, user.role === "client") can reach the
// normal app, but Camera/Site/License Management are admin-only —
// managing other companies' cameras/licenses isn't part of "their portal".
// Sidebar.jsx already hides these links for a client; this is the actual
// enforcement in case someone navigates to the URL directly.
export default function AdminRoute({ children }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  if (user.role === "client") return <Navigate to="/dashboard" replace />;
  return children;
}
