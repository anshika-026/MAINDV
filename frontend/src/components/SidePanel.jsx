import { X } from "lucide-react";

// Right-docked sliding panel — used for row-detail drill-ins (Attendance,
// Workforce clip viewer) where a centered Modal would feel too heavy for a
// quick "peek at this record" interaction.
export default function SidePanel({ open, onClose, title, children, width = "max-w-md" }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className={`relative w-full ${width} h-full bg-white shadow-xl p-6 overflow-y-auto`}>
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-slate-600"
        >
          <X size={18} />
        </button>
        {title && <h2 className="text-base font-semibold text-ink-900 mb-4 pr-6">{title}</h2>}
        {children}
      </div>
    </div>
  );
}
