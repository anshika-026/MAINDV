// Extra local mock detail for the People page. mockData.js's `people` /
// `validatedPeople` arrays are deliberately thin (shared with other pages'
// scope) — this file adds the richer fields the People UI needs, keyed by
// array position against those exports.

export const personTypes = [
  { key: "Employee", label: "Employee", desc: "Add to your organization." },
  { key: "Guest", label: "Guest", desc: "Create a one-time pass." },
];

export const departments = ["Accounts", "Engineering", "Operations", "Sales", "Admin"];

export const enrollmentFilters = ["All enrollment", "Enrolled", "Not enrolled"];

// Aligned 1:1 with mockData.people
export const employeeMeta = [
  { employeeId: "EMP-2201", syncStatus: "Active", zone: "Main Section", camera: "CAM-1-MH", lastSeenDesk: "09:12 AM", confidence: 94 },
  { employeeId: "EMP-2202", syncStatus: "Active", zone: "Tech Section", camera: "CAM-4-TS", lastSeenDesk: "09:40 AM", confidence: 91 },
  { employeeId: "EMP-2203", syncStatus: "Pending sync", zone: "-", camera: "-", lastSeenDesk: "-", confidence: 0 },
  { employeeId: "EMP-2204", syncStatus: "Active", zone: "Lift Area", camera: "CAM-3-LA", lastSeenDesk: "10:05 AM", confidence: 88 },
];

// Guests who registered but haven't checked in / been validated at a camera
// yet — distinct from mockData.validatedPeople, which are guests who have
// already been seen and validated/acknowledged on site.
export const incomingGuests = [
  {
    name: "Vikram Rao",
    faceEnrolled: false,
    designs: 0,
    date: "Aug 27",
    enrollment: "Not enrolled",
    employeeId: "-",
    syncStatus: "Pending sync",
    guestOf: "Aarti Prajapati",
    status: "Away",
    zone: "-",
    camera: "-",
    lastSeenDesk: "-",
    confidence: 0,
  },
  {
    name: "Neha Kapoor",
    faceEnrolled: true,
    designs: 1,
    date: "Aug 28",
    enrollment: "Enrolled",
    employeeId: "-",
    syncStatus: "Active",
    guestOf: "Rohan Sawant",
    status: "On site",
    zone: "Admin block",
    camera: "CAM-8-EE",
    lastSeenDesk: "02:18 PM",
    confidence: 92,
  },
];

// Aligned 1:1 with mockData.validatedPeople
export const validatedMeta = [
  { guestOf: "Aarti Prajapati", reason: "Vendor visit", zone: "Reception", camera: "CAM-2-MH", lastSeenDesk: "-", confidence: 97 },
  { guestOf: "Front desk", reason: "Walk-in enquiry", zone: "-", camera: "-", lastSeenDesk: "-", confidence: 0 },
];
