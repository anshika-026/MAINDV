import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import AdminRoute from "./components/AdminRoute";
import ErrorBoundary from "./components/ErrorBoundary";
import AppShell from "./layouts/AppShell";

import Login from "./pages/Login";
import ClientLogin from "./pages/ClientLogin";

// Pages load on demand: the app used to ship every page (and the charting
// library) in one ~850 kB bundle before the login screen could render.
const Dashboard = lazy(() => import("./pages/Dashboard"));
const LiveFeed = lazy(() => import("./pages/LiveFeed"));
const PlainLiveFeed = lazy(() => import("./pages/PlainLiveFeed"));
const Alerts = lazy(() => import("./pages/Alerts"));
const People = lazy(() => import("./pages/People"));
const Attendance = lazy(() => import("./pages/Attendance"));
const Workforce = lazy(() => import("./pages/Workforce"));
const Footfall = lazy(() => import("./pages/Footfall"));
const StaffCount = lazy(() => import("./pages/StaffCount"));
const FootfallUat = lazy(() => import("./pages/FootfallUat"));
const Intrusion = lazy(() => import("./pages/Intrusion"));
const ObjectDetection = lazy(() => import("./pages/ObjectDetection"));
const CameraManagement = lazy(() => import("./pages/CameraManagement"));
const SiteManagement = lazy(() => import("./pages/SiteManagement"));
const LicenseManagement = lazy(() => import("./pages/LicenseManagement"));
const FaceTraining = lazy(() => import("./pages/FaceTraining"));

const SettingsLayout = lazy(() => import("./pages/settings/SettingsLayout"));
const Profile = lazy(() => import("./pages/settings/Profile"));
const Notifications = lazy(() => import("./pages/settings/Notifications"));
const RulesPolicy = lazy(() => import("./pages/settings/RulesPolicy"));
const HolidayCalendar = lazy(() => import("./pages/settings/HolidayCalendar"));

function PageFallback() {
  return <div className="p-8 text-sm text-slate-500">Loading…</div>;
}

// Error boundary + lazy-load boundary per route (keyed by path, so moving to
// another page recovers from a crashed one).
function Page({ children }) {
  const { pathname } = useLocation();
  return (
    <ErrorBoundary key={pathname}>
      <Suspense fallback={<PageFallback />}>{children}</Suspense>
    </ErrorBoundary>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/client-login" element={<ClientLogin />} />
          {/* Dev-mode equivalent of client-<slug>.decovision.com — see
              backend/app/license_routes.py's public_company_by_slug and
              the report's production-deployment notes for real subdomains. */}
          <Route path="/client/:slug/login" element={<ClientLogin />} />
          {/* No self-signup: admin accounts are created by an operator
              (python -m app.manage create-admin), clients by an admin. */}
          <Route
            path="/face-training"
            element={
              <AdminRoute>
                <Page>
                  <FaceTraining />
                </Page>
              </AdminRoute>
            }
          />

          <Route
            element={
              <ProtectedRoute>
                <AppShell />
              </ProtectedRoute>
            }
          >
            <Route path="/dashboard" element={<Page><Dashboard /></Page>} />
            <Route path="/live-feed" element={<Page><PlainLiveFeed /></Page>} />
            <Route path="/live-camera" element={<Page><LiveFeed /></Page>} />
            <Route path="/alerts" element={<Page><Alerts /></Page>} />
            <Route path="/people" element={<Page><People /></Page>} />
            <Route path="/attendance" element={<Page><Attendance /></Page>} />
            <Route path="/staff" element={<Page><StaffCount /></Page>} />
            <Route path="/workforce" element={<Page><Workforce /></Page>} />
            <Route path="/footfall" element={<Page><Footfall /></Page>} />
            <Route path="/footfall-uat" element={<AdminRoute><Page><FootfallUat /></Page></AdminRoute>} />
            <Route path="/intrusion" element={<Page><Intrusion /></Page>} />
            <Route path="/objects" element={<Page><ObjectDetection /></Page>} />
            <Route path="/cameras" element={<AdminRoute><Page><CameraManagement /></Page></AdminRoute>} />
            <Route path="/sites" element={<AdminRoute><Page><SiteManagement /></Page></AdminRoute>} />
            <Route path="/licenses" element={<AdminRoute><Page><LicenseManagement /></Page></AdminRoute>} />

            <Route path="/settings" element={<Page><SettingsLayout /></Page>}>
              <Route index element={<Navigate to="profile" replace />} />
              <Route path="profile" element={<Page><Profile /></Page>} />
              <Route path="notifications" element={<Page><Notifications /></Page>} />
              <Route path="rules-policy" element={<Page><RulesPolicy /></Page>} />
              <Route path="holiday-calendar" element={<Page><HolidayCalendar /></Page>} />
            </Route>
          </Route>

          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
