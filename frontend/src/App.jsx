import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import AppShell from "./layouts/AppShell";

import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Dashboard from "./pages/Dashboard";
import LiveFeed from "./pages/LiveFeed";
import Alerts from "./pages/Alerts";
import People from "./pages/People";
import Attendance from "./pages/Attendance";
import Workforce from "./pages/Workforce";
import Footfall from "./pages/Footfall";
import Intrusion from "./pages/Intrusion";
import CameraManagement from "./pages/CameraManagement";
import SiteManagement from "./pages/SiteManagement";

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
          <Route path="/signup" element={<Signup />} />

          <Route
            element={
              <ProtectedRoute>
                <AppShell />
              </ProtectedRoute>
            }
          >
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/live-camera" element={<LiveFeed />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/people" element={<People />} />
            <Route path="/attendance" element={<Attendance />} />
            <Route path="/workforce" element={<Workforce />} />
            <Route path="/footfall" element={<Footfall />} />
            <Route path="/intrusion" element={<Intrusion />} />
            <Route path="/cameras" element={<CameraManagement />} />
            <Route path="/sites" element={<SiteManagement />} />

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
