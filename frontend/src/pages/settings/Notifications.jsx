import { useState } from "react";

function Toggle({ checked, onChange }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={`w-10 h-5.5 rounded-full relative transition-colors ${checked ? "bg-brand-500" : "bg-slate-300"}`}
      style={{ height: 22, width: 40 }}
    >
      <span
        className="absolute top-0.5 w-4.5 h-4.5 rounded-full bg-white transition-all"
        style={{ left: checked ? 20 : 2 }}
      />
    </button>
  );
}

const SECTIONS = [
  {
    title: "Camera",
    items: [
      { key: "cam_offline", label: "Camera offline, tracking data", checked: true },
      { key: "obj_detection", label: "Object detection stops", checked: false },
      { key: "ai_insights", label: "AI insights stops coming", checked: false },
    ],
  },
  {
    title: "Alerts",
    items: [
      { key: "camera_stop", label: "Camera stops tracking data", checked: true },
      { key: "obj_detection2", label: "Object detection stops", checked: false },
      { key: "ai_insights2", label: "AI insights stops coming", checked: false },
    ],
  },
];

export default function Notifications() {
  const [state, setState] = useState(() =>
    Object.fromEntries(SECTIONS.flatMap((s) => s.items.map((i) => [i.key, i.checked])))
  );

  return (
    <div className="card p-6">
      <h2 className="text-base font-semibold text-ink-900 mb-1">General updates</h2>
      <p className="text-sm text-slate-500 mb-6">Choose which notifications you'd like to receive.</p>

      <div className="space-y-6">
        {SECTIONS.map((section) => (
          <div key={section.title}>
            <h3 className="text-sm font-semibold text-ink-900 mb-3">{section.title}</h3>
            <div className="space-y-3">
              {section.items.map((item) => (
                <div key={item.key} className="flex items-center justify-between">
                  <span className="text-sm text-slate-600">{item.label}</span>
                  <Toggle
                    checked={state[item.key]}
                    onChange={(v) => setState((s) => ({ ...s, [item.key]: v }))}
                  />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <button className="btn-primary mt-6 text-sm">Save</button>
    </div>
  );
}
