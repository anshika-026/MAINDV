import { X } from "lucide-react";

export default function Modal({ open, onClose, title, children, width = "max-w-md" }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
      <div className={`w-full ${width} bg-white rounded-2xl shadow-xl p-6 relative`}>
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
