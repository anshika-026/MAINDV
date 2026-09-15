// ---------------------------------------------------------------------------
// Extra mock data for the Intrusion page & its zone-detail view. Keyed by
// zone name so it can be looked up from mockData.js's `intrusionZones`.
// ---------------------------------------------------------------------------

export const weeklyUnauthorized = [
  { day: "Mon", count: 1 },
  { day: "Tue", count: 3 },
  { day: "Wed", count: 2 },
  { day: "Thu", count: 4 },
  { day: "Fri", count: 2 },
  { day: "Sat", count: 1 },
  { day: "Sun", count: 3 },
];

export const zoneAccessDetail = {
  "Reception Area": [
    { name: "Aarti Prajapati", activeWindow: "Always", enabled: true },
    { name: "Rohan Sawant", activeWindow: "9:00 AM - 6:00 PM", enabled: true },
    { name: "K.V Ramasubramanian", activeWindow: "Always", enabled: false },
    { name: "Sushant Mehta", activeWindow: "8:00 AM - 8:00 PM", enabled: true },
  ],
  "Canteen Area": [
    { name: "Harsh Gupta", activeWindow: "Always", enabled: true },
    { name: "Sachin Singh", activeWindow: "12:00 PM - 3:00 PM", enabled: true },
    { name: "Mukul Singh", activeWindow: "Always", enabled: true },
    { name: "Kashish Yadav", activeWindow: "Always", enabled: false },
  ],
};

export const defaultZoneAccess = [
  { name: "Aarti Prajapati", activeWindow: "Always", enabled: true },
];

export const waitingRoomStats = {
  "Reception Area": {
    byHour: [
      { label: "9 AM", value: 3 },
      { label: "11 AM", value: 7 },
      { label: "1 PM", value: 5 },
      { label: "3 PM", value: 9 },
      { label: "5 PM", value: 4 },
    ],
    byDay: [
      { label: "Mon", value: 12 },
      { label: "Tue", value: 18 },
      { label: "Wed", value: 9 },
      { label: "Thu", value: 21 },
      { label: "Fri", value: 15 },
    ],
  },
  "Canteen Area": {
    byHour: [
      { label: "9 AM", value: 2 },
      { label: "11 AM", value: 5 },
      { label: "1 PM", value: 14 },
      { label: "3 PM", value: 6 },
      { label: "5 PM", value: 3 },
    ],
    byDay: [
      { label: "Mon", value: 20 },
      { label: "Tue", value: 24 },
      { label: "Wed", value: 17 },
      { label: "Thu", value: 26 },
      { label: "Fri", value: 22 },
    ],
  },
};

export const defaultWaitingRoomStats = {
  byHour: [
    { label: "9 AM", value: 2 },
    { label: "11 AM", value: 4 },
    { label: "1 PM", value: 3 },
    { label: "3 PM", value: 5 },
    { label: "5 PM", value: 2 },
  ],
  byDay: [
    { label: "Mon", value: 8 },
    { label: "Tue", value: 10 },
    { label: "Wed", value: 6 },
    { label: "Thu", value: 11 },
    { label: "Fri", value: 9 },
  ],
};

export const recentDetections = {
  "Reception Area": [
    { time: "10:04 AM", name: "Vivek Jaiswar" },
    { time: "10:04 AM", name: "Unknown Person" },
    { time: "09:47 AM", name: "Aarti Prajapati" },
    { time: "09:30 AM", name: "Unknown Person" },
    { time: "09:12 AM", name: "Rohan Sawant" },
    { time: "08:58 AM", name: "Unknown Person" },
  ],
  "Canteen Area": [
    { time: "01:22 PM", name: "Mukul Singh" },
    { time: "01:10 PM", name: "Unknown Person" },
    { time: "12:55 PM", name: "Harsh Gupta" },
    { time: "12:40 PM", name: "Sachin Singh" },
    { time: "12:31 PM", name: "Unknown Person" },
  ],
};

export const defaultRecentDetections = [
  { time: "10:04 AM", name: "Unknown Person" },
];

// A slightly richer people directory for the "Add people" multi-select in
// the Add-new-zone modal (mockData.js's `people` array is sparse / has
// repeats, so this fills it out for a realistic search-and-pick UI).
export const peopleDirectory = [
  "Aarti Prajapati",
  "Rohan Sawant",
  "K.V Ramasubramanian",
  "Sushant Mehta",
  "Harsh Gupta",
  "Sachin Singh",
  "Mukul Singh",
  "Kashish Yadav",
  "Vivek Jaiswar",
  "Shilpa Chaudhary",
];
