export default function StatCard({ label, value, sub, subTone = "success" }) {
  const tone = {
    success: "text-success-600",
    danger: "text-danger-500",
    neutral: "text-slate-400",
  }[subTone];

  return (
    <div className="stat-card">
      <p className="text-2xl font-semibold text-ink-900 leading-none">{value}</p>
      <p className="text-sm text-slate-500 mt-2">{label}</p>
      {sub ? <p className={`text-xs mt-2 ${tone}`}>{sub}</p> : null}
    </div>
  );
}
