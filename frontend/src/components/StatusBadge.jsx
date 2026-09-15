const TONE_MAP = {
  active: "badge-danger",
  critical: "badge-danger",
  high: "badge-danger",
  absent: "badge-danger",
  unknown: "badge-danger",
  medium: "badge-warning",
  acknowledged: "badge-warning",
  "late arrival": "badge-warning",
  resolved: "badge-success",
  present: "badge-success",
  enrolled: "badge-success",
  validated: "badge-success",
  active_camera: "badge-success",
  "on time": "badge-success",
  "on site": "badge-success",
  low: "badge-neutral",
  inactive: "badge-neutral",
  "not enrolled": "badge-neutral",
};

export default function StatusBadge({ value }) {
  const key = String(value).toLowerCase();
  const tone = TONE_MAP[key] || "badge-neutral";
  return <span className={`badge ${tone}`}>{value}</span>;
}
