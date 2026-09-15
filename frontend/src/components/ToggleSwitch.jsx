export default function ToggleSwitch({ checked, onChange, disabled = false }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange?.(!checked)}
      className={`inline-flex items-center w-9 h-5 rounded-full relative shrink-0 transition-colors ${
        checked ? "bg-brand-500" : "bg-slate-300"
      } ${disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer"}`}
    >
      <span
        className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-all ${
          checked ? "left-4.5" : "left-0.5"
        }`}
      />
    </button>
  );
}
