import { X } from "lucide-react";

export default function Modal({ open, onClose, title, children, width = "max-w-md" }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4 py-8">
      {/* max-h + flex column so a tall form (e.g. Generate License, Face
          enrollment) scrolls internally instead of overflowing past the
          viewport with the Save/Submit button unreachable. */}
      <div className={`w-full ${width} max-h-full bg-white rounded-2xl shadow-xl relative flex flex-col`}>
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-slate-600 z-10"
        >
          <X size={18} />
        </button>
        <div className="overflow-y-auto p-6">
          {title && <h2 className="text-base font-semibold text-ink-900 mb-4 pr-6">{title}</h2>}
          {children}
        </div>
      </div>
    </div>
  );
}
