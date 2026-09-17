import { useState } from "react";
import { Eye, EyeOff } from "lucide-react";

// Drop-in replacement for <input type="password" className="input-field" />
// with a show/hide toggle on the right — forwards every other prop
// (value, onChange, required, minLength, placeholder, ...) straight
// through to the underlying <input>.
export default function PasswordField({ className = "", ...inputProps }) {
  const [visible, setVisible] = useState(false);
  return (
    <div className="relative">
      <input
        {...inputProps}
        type={visible ? "text" : "password"}
        className={`input-field pr-10 ${className}`}
      />
      <button
        type="button"
        onClick={() => setVisible((v) => !v)}
        tabIndex={-1}
        title={visible ? "Hide password" : "Show password"}
        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
      >
        {visible ? <EyeOff size={16} /> : <Eye size={16} />}
      </button>
    </div>
  );
}
