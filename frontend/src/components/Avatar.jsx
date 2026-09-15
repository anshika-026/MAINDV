// Small circular avatar used wherever a person needs a photo placeholder —
// People, Attendance and Workforce all show "photo" columns but there are no
// real images in this mock-data-driven app, so this renders initials on a
// deterministic colored circle instead.
const PALETTE = [
  { bg: "bg-brand-100", text: "text-brand-700" },
  { bg: "bg-[#eefbf0]", text: "text-success-600" },
  { bg: "bg-[#fdf5e6]", text: "text-[#8a6410]" },
  { bg: "bg-[#fdeeee]", text: "text-danger-600" },
  { bg: "bg-[#eef0f5]", text: "text-[#52546b]" },
];

function colorFor(seed) {
  let hash = 0;
  for (let i = 0; i < seed.length; i++) hash = seed.charCodeAt(i) + ((hash << 5) - hash);
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

function initialsFor(name) {
  const parts = String(name || "?").trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

export default function Avatar({ name, size = 32, className = "" }) {
  const { bg, text } = colorFor(String(name || "?"));
  return (
    <span
      className={`inline-flex items-center justify-center rounded-full font-semibold shrink-0 ${bg} ${text} ${className}`}
      style={{ width: size, height: size, fontSize: Math.max(10, size * 0.36) }}
    >
      {initialsFor(name)}
    </span>
  );
}
