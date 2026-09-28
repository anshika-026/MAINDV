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

