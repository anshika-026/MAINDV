import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import AdminRoute from "./components/AdminRoute";
import AppShell from "./layouts/AppShell";

import Login from "./pages/Login";
import ClientLogin from "./pages/ClientLogin";
import Dashboard from "./pages/Dashboard";
import LiveFeed from "./pages/LiveFeed";
import PlainLiveFeed from "./pages/PlainLiveFeed";
import Alerts from "./pages/Alerts";
import People from "./pages/People";
import Attendance from "./pages/Attendance";
import Workforce from "./pages/Workforce";
import Footfall from "./pages/Footfall";
import StaffCount from "./pages/StaffCount";
import FootfallUat from "./pages/FootfallUat";
import Intrusion from "./pages/Intrusion";
import CameraManagement from "./pages/CameraManagement";
import SiteManagement from "./pages/SiteManagement";
import LicenseManagement from "./pages/LicenseManagement";
import FaceTraining from "./pages/FaceTraining";

import SettingsLayout from "./pages/settings/SettingsLayout";
import Profile from "./pages/settings/Profile";
import Notifications from "./pages/settings/Notifications";
import RulesPolicy from "./pages/settings/RulesPolicy";
import HolidayCalendar from "./pages/settings/HolidayCalendar";

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
                <FaceTraining />
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
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/live-feed" element={<PlainLiveFeed />} />
            <Route path="/live-camera" element={<LiveFeed />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/people" element={<People />} />
            <Route path="/attendance" element={<Attendance />} />
            <Route path="/staff" element={<StaffCount />} />
            <Route path="/workforce" element={<Workforce />} />
            <Route path="/footfall" element={<Footfall />} />
            <Route path="/footfall-uat" element={<AdminRoute><FootfallUat /></AdminRoute>} />
            <Route path="/intrusion" element={<Intrusion />} />
            <Route path="/cameras" element={<AdminRoute><CameraManagement /></AdminRoute>} />
            <Route path="/sites" element={<AdminRoute><SiteManagement /></AdminRoute>} />
            <Route path="/licenses" element={<AdminRoute><LicenseManagement /></AdminRoute>} />

            <Route path="/settings" element={<SettingsLayout />}>
              <Route index element={<Navigate to="profile" replace />} />
              <Route path="profile" element={<Profile />} />
              <Route path="notifications" element={<Notifications />} />
              <Route path="rules-policy" element={<RulesPolicy />} />
              <Route path="holiday-calendar" element={<HolidayCalendar />} />
            </Route>
          </Route>

          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
