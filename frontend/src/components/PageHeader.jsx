export default function PageHeader({ title, action }) {
  return (
    <div className="flex items-center justify-between mb-5">
      <h2 className="text-lg font-semibold text-ink-900">{title}</h2>
      {action}
    </div>
  );
}
