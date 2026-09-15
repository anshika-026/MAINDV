// ---------------------------------------------------------------------------
// Extra mock data for the Alerts detail side-panel — journeys, confidence
// scores and "person known" state that mockData.js's `alerts` array doesn't
// carry. Keyed by alert id so it can be merged onto each row at render time.
// ---------------------------------------------------------------------------

export const alertDetails = {
  "AL-1001": {
    confidence: 94,
    personKnown: "Unknown",
    journey: [
      { time: "02:10 PM", camera: "CAM-08-EE", location: "Entry / Exit" },
      { time: "02:14 PM", camera: "CAM-03-CO", location: "Corridor" },
      { time: "02:18 PM", camera: "CAM-08-EE", location: "Entry / Exit" },
    ],
  },
  "AL-1002": {
    confidence: 88,
    personKnown: "Unknown",
    journey: [
      { time: "03:52 PM", camera: "CAM-09-CA", location: "Canteen" },
      { time: "04:00 PM", camera: "CAM-09-CA", location: "Canteen" },
    ],
  },
  "AL-1003": {
    confidence: 76,
    personKnown: "Unknown",
    journey: [
      { time: "03:58 PM", camera: "CAM-09-CA", location: "Meeting room" },
      { time: "04:00 PM", camera: "CAM-09-CA", location: "Meeting room" },
    ],
  },
  "AL-1004": {
    confidence: 81,
    personKnown: "Acknowledged",
    journey: [
      { time: "03:55 PM", camera: "CAM-09-CA", location: "Meeting room" },
      { time: "04:00 PM", camera: "CAM-09-CA", location: "Meeting room" },
    ],
  },
  "AL-1005": {
    confidence: 97,
    personKnown: "Acknowledged",
    journey: [
      { time: "02:12 PM", camera: "CAM-08-EE", location: "Entry / Exit" },
      { time: "02:18 PM", camera: "CAM-08-EE", location: "Entry / Exit" },
    ],
  },
  "AL-1006": {
    confidence: 90,
    personKnown: "Unknown",
    journey: [
      { time: "03:57 PM", camera: "CAM-09-CA", location: "Meeting room" },
      { time: "04:00 PM", camera: "CAM-09-CA", location: "Meeting room" },
    ],
  },
};

export const defaultAlertDetail = {
  confidence: 82,
  personKnown: "Unknown",
  journey: [],
};

export const resolutionReasons = ["Vendor", "False alarm", "Duplicate", "Other"];
