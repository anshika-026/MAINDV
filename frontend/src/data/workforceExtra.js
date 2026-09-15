// Extra local mock detail for the Workforce page (trend chart, richer
// People/Desk analytics rows, and the seed list of desk zones the "Draw desk
// zone" flow appends to). mockData.js's `workforcePeopleAnalytics` /
// `deskAnalytics` only ship a couple of near-duplicate rows, so this adds
// realistic variety without touching that shared file.

export const weeklyAttendanceTrend = [
  { name: "Mon", value: 34 },
  { name: "Tue", value: 37 },
  { name: "Wed", value: 31 },
  { name: "Thu", value: 33 },
  { name: "Fri", value: 29 },
  { name: "Sat", value: 18 },
  { name: "Sun", value: 12 },
];

export const peopleAnalyticsExtra = [
  { name: "Rohan Sawant", lastSeen: "Control Section", lastSeenAt: "27 Aug, 01:10 PM", clips: 62 },
  { name: "Aarti Prajapati", lastSeen: "Main Section", lastSeenAt: "27 Aug, 11:42 AM", clips: 45 },
  { name: "K.V Ramasubramanian", lastSeen: "Technical Section", lastSeenAt: "26 Aug, 05:03 PM", clips: 28 },
];

export const deskAnalyticsExtra = [
  { person: "Rohan Sawant", firstSeen: "28 Aug, 09:10 AM", lastSeen: "28 Aug, 01:40 PM", deskTime: "3h 40m", awayTime: "22m", currentDesk: "Desk 5", status: "At Desk", movements: 3 },
  { person: "K.V Ramasubramanian", firstSeen: "28 Aug, 08:55 AM", lastSeen: "28 Aug, 12:05 PM", deskTime: "1h 58m", awayTime: "40m", currentDesk: "Desk 8", status: "Away", movements: 6 },
  { person: "Mukul Singh", firstSeen: "28 Aug, 09:30 AM", lastSeen: "28 Aug, 02:15 PM", deskTime: "4h 05m", awayTime: "8m", currentDesk: "Desk 3", status: "At Desk", movements: 1 },
];

export const initialDeskZones = [
  { name: "Desk 2", type: "Entry" },
  { name: "Desk 3", type: "Entry" },
  { name: "Desk 5", type: "Exit" },
  { name: "Desk 8", type: "Entry" },
];
