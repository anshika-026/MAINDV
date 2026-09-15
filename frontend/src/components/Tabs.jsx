export default function Tabs({ tabs, active, onChange }) {
  return (
    <div className="flex items-center gap-1 border-b border-[#e7e8f0] mb-5">
      {tabs.map((tab) => (
        <button
          key={tab}
          onClick={() => onChange(tab)}
          className={`px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
            active === tab
              ? "border-brand-500 text-brand-600"
              : "border-transparent text-slate-500 hover:text-ink-900"
          }`}
        >
          {tab}
        </button>
      ))}
    </div>
  );
}
