const HOLIDAYS = [
  { name: "Republic Day", date: "26/01/2026" },
  { name: "Independence Day", date: "15/08/2026" },
];

export default function HolidayCalendar() {
  return (
    <div className="card p-6">
      <h2 className="text-base font-semibold text-ink-900 mb-5">Holiday Calendar</h2>
      <ul className="divide-y divide-[#f1f2f7]">
        {HOLIDAYS.map((h) => (
          <li key={h.name} className="flex items-center justify-between py-3 text-sm">
            <span className="text-ink-900 font-medium">{h.name}</span>
            <span className="text-slate-500">{h.date}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
