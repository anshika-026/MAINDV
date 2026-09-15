// Extra local mock detail for the People page. mockData.js's `people` /
// `validatedPeople` arrays are deliberately thin (shared with other pages'
// scope) — this file adds the richer fields the People UI needs, keyed by
// array position against those exports.

export const personTypes = [
  { key: "Employee", label: "Employee", desc: "Full-time or contract staff with facility access." },
  { key: "Guest", label: "Guest", desc: "Visitor, vendor or incoming guest for a limited stay." },
];

// Aligned 1:1 with mockData.people
export const employeeMeta = [
  { employeeId: "EMP-2201", syncStatus: "Active" },
  { employeeId: "EMP-2202", syncStatus: "Active" },
  { employeeId: "EMP-2203", syncStatus: "Pending sync" },
  { employeeId: "EMP-2204", syncStatus: "Active" },
];

// Guests who registered but haven't checked in / been validated at a camera
// yet — distinct from mockData.validatedPeople, which are guests who have
// already been seen and validated/acknowledged on site.
export const incomingGuests = [
  { name: "Vikram Rao", faceEnrolled: false, designs: 0, date: "Aug 27", enrollment: "Not enrolled", employeeId: "-", syncStatus: "Pending sync", guestOf: "Aarti Prajapati" },
  { name: "Neha Kapoor", faceEnrolled: true, designs: 1, date: "Aug 28", enrollment: "Enrolled", employeeId: "-", syncStatus: "Active", guestOf: "Rohan Sawant" },
];

// Aligned 1:1 with mockData.validatedPeople
export const validatedMeta = [
  { guestOf: "Aarti Prajapati", reason: "Vendor visit" },
  { guestOf: "Front desk", reason: "Walk-in enquiry" },
];
