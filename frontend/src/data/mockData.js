// ---------------------------------------------------------------------------
// MOCK DATA
// Every page currently reads from here. When you wire up your backend,
// replace the corresponding functions in `src/api/client.js` with real
// fetch/axios calls — the pages already call through that layer, so you
// only need to change one file per resource.
// ---------------------------------------------------------------------------

export const currentUser = {
  name: "Jay Jain",
  email: "jay.jain@deco.in",
  mobile: "+91 9284746367",
  role: "Admin",
};

export const dashboardStats = {
  admin: {
    peoplePresent: { value: 25, of: 40, sub: "+2 vs yesterday" },
    footfallToday: { value: "18 in / 3 out", sub: "+12 vs yesterday" },
    unknownVisitors: { value: 12, sub: "+3 vs yesterday" },
    camerasOnline: { value: "8 / 10", sub: "-1 vs yesterday" },
    guestsIncoming: { value: 7, sub: "" },
    activeAlerts: { value: 2, sub: "1 unacknowledged" },
  },
  needsAttention: [
    { label: "Unknown Person", detail: "Entry / Exit · 05:37 PM", tag: "Critical" },
    { label: "Intrusion Detected", detail: "", tag: "Acknowledged" },
  ],
  aiInsights: [
    "Footfall peaked at 11:00 AM, with 59 visitors recorded at Entry / Exit.",
    "3 employees have not been detected since morning log in.",
    "Most active zone: Detected 847 movements in Technical section.",
  ],
};

export const alertsSummary = {
  active: 3,
  acknowledged: 1,
  resolvedToday: 8,
  liveAlerts: 3,
};

export const alerts = [
  { id: "AL-1001", date: "Aug 28", event: "Unknown Person", camera: "CAM-08-EE", location: "Entry / Exit", time: "02:18 PM", severity: "Critical", status: "Active" },
  { id: "AL-1002", date: "Aug 28", event: "Smoke Detection", camera: "CAM-09-CA", location: "Canteen", time: "04:00 PM", severity: "Critical", status: "Active" },
  { id: "AL-1003", date: "Aug 28", event: "Intrusion Detection", camera: "CAM-09-CA", location: "Meeting room", time: "04:00 PM", severity: "High", status: "Active" },
  { id: "AL-1004", date: "Aug 28", event: "Intrusion Detection", camera: "CAM-09-CA", location: "Meeting room", time: "04:00 PM", severity: "Medium", status: "Acknowledged" },
  { id: "AL-1005", date: "Aug 28", event: "Unknown Person", camera: "CAM-08-EE", location: "Entry / Exit", time: "02:18 PM", severity: "Low", status: "Resolved", resolvedBy: "Harsh Gupta" },
  { id: "AL-1006", date: "Aug 28", event: "Intrusion Detection", camera: "CAM-09-CA", location: "Meeting room", time: "04:00 PM", severity: "Low", status: "Resolved", resolvedBy: "Sachin Singh" },
];

export const cameras = [
  { code: "CAM-UNC-0AB53D32-NOIDA2MP1", label: "Entry / Exit", site: "Noida", purpose: "General", status: "Active", live: "On" },
  { code: "CAM-UNC-0AB53D32-NOIDA2MP2", label: "Entry / Exit", site: "Mumbai", purpose: "General", status: "Active", live: "On" },
];

export const sites = [
  { name: "Noida", cameras: 8, wgs: 2, status: "Active" },
  { name: "Mumbai", cameras: 2, wgs: "-", status: "Inactive" },
];

export const people = [
  { name: "Aarti Prajapati", faceEnrolled: true, designs: 4, date: "Aug 28", enrollment: "Enrolled" },
  { name: "Rohan Sawant", faceEnrolled: true, designs: 4, date: "Aug 28", enrollment: "Enrolled" },
  { name: "K.V Ramasubram...", faceEnrolled: false, designs: 6, date: "Aug 28", enrollment: "Not enrolled" },
  { name: "Rohan Sawant", faceEnrolled: true, designs: 4, date: "Aug 28", enrollment: "Enrolled" },
];

export const validatedPeople = [
  { name: "Sushant Mehta", enrollment: "Validated", date: "Aug 26" },
  { name: "Unknown person", enrollment: "Acknowledged", date: "Aug 26", needsValidation: true },
];

export const attendance = [
  { employee: "Aarti Prajapati", empId: "0J01", timeIn: "09:05 AM", timeOut: "-", timeStay: "2h 04m", arrival: "On time", status: "Present" },
  { employee: "Kashish Yadav", empId: "0J01", timeIn: "-", timeOut: "-", timeStay: "-", arrival: "-", status: "Absent" },
  { employee: "Mukul Singh", empId: "0J01", timeIn: "09:05 AM", timeOut: "-", timeStay: "2h 04m", arrival: "Late arrival", status: "On site" },
];

export const attendanceStats = {
  present: 25, presentOf: 40,
  absent: 15,
  attendancePct: "62.5%",
  lateArrivals: 2,
};

export const workforceStats = {
  totalEmployees: 40,
  employeeExited: 2,
  attendancePct: "62.5%",
  notDetected: 15,
};

export const workforcePeopleAnalytics = [
  { name: "Shilpa Chaudhary", lastSeen: "Main Section", lastSeenAt: "27 Aug, 02:17 PM", clips: 77 },
  { name: "Shilpa Chaudhary", lastSeen: "Main Section", lastSeenAt: "27 Aug, 04:00 PM", clips: 54 },
  { name: "Shilpa Chaudhary", lastSeen: "Control Section", lastSeenAt: "26 Aug, 03:55 PM", clips: 48 },
  { name: "Shilpa Chaudhary", lastSeen: "Main Section", lastSeenAt: "26 Aug, 02:17 PM", clips: 39 },
];

export const deskAnalytics = [
  { person: "Aarti Prajapati", firstSeen: "28 Aug, 09:52 AM", lastSeen: "28 Aug, 09:52 AM", deskTime: "2h 53m", awayTime: "12m", currentDesk: "Desk 2", status: "At Desk", movements: 0 },
  { person: "Aarti Prajapati", firstSeen: "28 Aug, 09:52 AM", lastSeen: "28 Aug, 09:52 AM", deskTime: "2h 32m", awayTime: "12m", currentDesk: "Desk 2", status: "At Desk", movements: 0 },
];

export const footfallStats = {
  peopleCounted: 21,
  busiestHour: "11:00 AM",
  unknownVisitors: 12,
  currentOccupancy: 18,
};

export const footfallVisitors = [
  { person: "Unknown person", firstSeen: "28 Aug, 09:52 AM", lastSeen: "28 Aug, 09:52 AM", camera: "Entry / Exit", enrollment: "Unknown" },
  { person: "Aarti Prajapati", firstSeen: "28 Aug, 09:52 AM", lastSeen: "28 Aug, 09:52 AM", camera: "Entry / Exit", enrollment: "Enrolled" },
];

export const intrusionZones = [
  { zone: "Reception Area", access: 24, action: "Give Access" },
  { zone: "Canteen Area", access: 60, action: "Give Access" },
];
