// Extra local mock detail for the Footfall page — an hourly traffic series
// (today vs yesterday), the visitor-composition split, and a few extra
// visitor-log rows layered on top of mockData.js's thin `footfallVisitors`.

export const hourlyTraffic = [
  { hour: "8am", today: 3, yesterday: 2 },
  { hour: "9am", today: 8, yesterday: 6 },
  { hour: "10am", today: 14, yesterday: 11 },
  { hour: "11am", today: 21, yesterday: 17 },
  { hour: "12pm", today: 15, yesterday: 19 },
  { hour: "1pm", today: 12, yesterday: 10 },
  { hour: "2pm", today: 9, yesterday: 8 },
  { hour: "3pm", today: 11, yesterday: 9 },
  { hour: "4pm", today: 7, yesterday: 6 },
];

// tone keys map to design-system colors in Footfall.jsx (brand / success / danger)
export const composition = [
  { label: "Employee", value: 58, tone: "brand" },
  { label: "Guest", value: 20, tone: "success" },
  { label: "Unknown Visitors", value: 22, tone: "danger" },
];

export const compositionStats = { topHour: "11:00 AM", uniqueVisitors: 21 };

export const footfallVisitorsExtra = [
  { person: "Rohan Sawant", firstSeen: "28 Aug, 08:40 AM", lastSeen: "28 Aug, 06:02 PM", camera: "Entry / Exit", enrollment: "Enrolled" },
  { person: "Unknown person", firstSeen: "28 Aug, 10:05 AM", lastSeen: "28 Aug, 10:05 AM", camera: "Canteen", enrollment: "Unknown" },
  { person: "Guest - Vikram Rao", firstSeen: "28 Aug, 11:15 AM", lastSeen: "28 Aug, 01:20 PM", camera: "Entry / Exit", enrollment: "Not enrolled" },
];
