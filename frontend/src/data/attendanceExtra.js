// Extra local mock detail for the Attendance page's filters and the
// row-detail side panel. Aligned 1:1 with mockData.attendance by index.

export const departments = ["All departments", "Design", "Engineering", "Operations"];

export const statusFilters = ["All statuses", "Present", "Absent", "On site", "On Leave"];

export const attendanceMeta = [
  {
    date: "28 Aug, 2026",
    department: "Design",
    role: "Design Lead",
    zone: "Main Section",
    camera: "CAM-08-EE",
    lastSeenDesk: "09:52 AM",
    confidence: 96,
    history: [
      { date: "27 Aug", status: "Present" },
      { date: "26 Aug", status: "Present" },
      { date: "25 Aug", status: "Late arrival" },
      { date: "24 Aug", status: "Present" },
      { date: "23 Aug", status: "Absent" },
    ],
  },
  {
    date: "28 Aug, 2026",
    department: "Engineering",
    role: "Site Engineer",
    zone: "-",
    camera: "-",
    lastSeenDesk: "-",
    confidence: 0,
    history: [
      { date: "27 Aug", status: "Absent" },
      { date: "26 Aug", status: "Present" },
      { date: "25 Aug", status: "Present" },
      { date: "24 Aug", status: "Present" },
      { date: "23 Aug", status: "Present" },
    ],
  },
  {
    date: "28 Aug, 2026",
    department: "Operations",
    role: "Project Manager",
    zone: "Control Section",
    camera: "CAM-09-CA",
    lastSeenDesk: "11:20 AM",
    confidence: 89,
    history: [
      { date: "27 Aug", status: "Late arrival" },
      { date: "26 Aug", status: "Present" },
      { date: "25 Aug", status: "Present" },
      { date: "24 Aug", status: "Late arrival" },
      { date: "23 Aug", status: "Present" },
    ],
  },
];
