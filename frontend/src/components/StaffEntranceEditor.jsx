import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeftRight, RefreshCw, Undo2 } from "lucide-react";
import ToggleSwitch from "./ToggleSwitch";
import * as api from "../api/client";

// Set up one entrance camera for Staff Count (backend/app/staff/): draw the
// entry line across the doorway (2 clicks), choose which side is the office,
// and optionally outline the office area (people outside it are ignored, e.g.
// a reception desk). Points are fractions of the frame.

const LINE = "#facc15";
const ROI = "#22d3ee";

function officeArrow(line, insideSign) {
  // Midpoint of the line and a short arrow pointing to the office side.
  const [[x1, y1], [x2, y2]] = line;
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  // The backend's "inside" = positive cross product (b - a) x (p - a), in pixel
  // space; with y pointing down that is the normal (-dy, dx).
  const dx = (x2 - x1) * 16;
  const dy = (y2 - y1) * 9;
  const len = Math.hypot(dx, dy) || 1;
  const nx = (-dy / len) * insideSign;
  const ny = (dx / len) * insideSign;
  return { mx, my, ex: mx + (nx * 0.08 * 9) / 16, ey: my + ny * 0.08 };
}

export default function StaffEntranceEditor({ onSaved }) {
  const [cameras, setCameras] = useState([]);
  const [cameraId, setCameraId] = useState(null);
  const [frameUrl, setFrameUrl] = useState(null);
  const [frameError, setFrameError] = useState("");
  const [mode, setMode] = useState("line");
  const [line, setLine] = useState([]);
  const [roi, setRoi] = useState([]);
  const [insideSign, setInsideSign] = useState(1);
  const [enabled, setEnabled] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const areaRef = useRef(null);

  useEffect(() => {
    api.getStaffCameras().then((cs) => {
      setCameras(cs);
      const configured = cs.find((c) => c.config);
      setCameraId((configured || cs[0])?.camera_id ?? null);
    });
  }, []);

  useEffect(() => {
    const c = cameras.find((x) => x.camera_id === cameraId);
    const cfg = c?.config;
    setLine(cfg?.line || []);
    setRoi(cfg?.roi || []);
    setInsideSign(cfg?.inside_sign ?? 1);
    setEnabled(cfg ? cfg.enabled : true);
    setMode("line");
    setMessage("");
  }, [cameraId, cameras]);

  const loadFrame = useCallback(() => {
    if (!cameraId) return;
    setFrameError("");
    api
      .fetchCameraFrameObjectUrl(cameraId)
      .then((u) =>
        setFrameUrl((old) => {
          if (old) URL.revokeObjectURL(old);
          return u;
        })
      )
      .catch((e) => setFrameError(e.message));
  }, [cameraId]);

  useEffect(() => {
    setFrameUrl(null);
    loadFrame();
  }, [loadFrame]);

  function click(e) {
    if (!frameUrl) return;
    const r = areaRef.current.getBoundingClientRect();
    const p = [+Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)).toFixed(4), +Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)).toFixed(4)];
    if (mode === "line") setLine((l) => (l.length >= 2 ? [p] : [...l, p]));
    else setRoi((pts) => [...pts, p]);
    setMessage("");
  }

  async function save() {
    setSaving(true);
    try {
      await api.setStaffCameraConfig(cameraId, { enabled, line, insideSign, roi: roi.length >= 3 ? roi : null });
      setMessage(enabled ? "Saved. Counting starts within a few seconds." : "Saved. Counting is off for this camera.");
      const cs = await api.getStaffCameras();
      setCameras(cs);
      onSaved?.();
    } catch (e) {
      setMessage(e.message.includes("422") ? "Draw the entry line first: two clicks across the doorway." : "Couldn't save. Check the backend is running.");
    } finally {
      setSaving(false);
    }
  }

  const svg = (pts) => pts.map(([x, y]) => `${x * 100},${y * 100}`).join(" ");
  const arrow = line.length === 2 ? officeArrow(line, insideSign) : null;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor="staff-camera" className="text-sm font-medium text-ink-900">
          Entrance camera
        </label>
        <select id="staff-camera" value={cameraId ?? ""} onChange={(e) => setCameraId(Number(e.target.value))} className="input-field w-auto">
          {cameras.map((c) => (
            <option key={c.camera_id} value={c.camera_id}>
              {c.name}
              {c.config?.enabled ? " (counting)" : ""}
            </option>
          ))}
        </select>
        <button type="button" onClick={loadFrame} className="px-2.5 py-2 rounded-lg text-sm text-slate-500 flex items-center gap-1.5">
          <RefreshCw size={14} /> New picture
        </button>
      </div>

      <div className="flex flex-wrap gap-1">
        {[
          ["line", "1. Entry line"],
          ["roi", "2. Office area (optional)"],
        ].map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => setMode(key)}
            className={`px-3 py-1.5 rounded-full text-sm font-medium ${mode === key ? "bg-brand-500 text-white" : "bg-white border border-border-200 text-slate-600"}`}
          >
            {label}
          </button>
        ))}
      </div>

      <div ref={areaRef} onClick={click} className="relative w-full aspect-video bg-[#11151c] rounded-lg overflow-hidden select-none cursor-crosshair">
        {frameUrl ? (
          <img src={frameUrl} alt="Live picture from the camera" className="w-full h-full object-fill pointer-events-none" draggable={false} />
        ) : (
          <p className="absolute inset-0 flex items-center justify-center text-sm text-slate-300 px-4 text-center">{frameError || "Loading the camera picture…"}</p>
        )}
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 w-full h-full pointer-events-none">
          {roi.length > 1 && <polygon points={svg(roi)} fill="rgba(34,211,238,0.14)" stroke={ROI} strokeWidth="2" vectorEffect="non-scaling-stroke" />}
          {line.length === 2 && <polyline points={svg(line)} stroke={LINE} strokeWidth="4" vectorEffect="non-scaling-stroke" />}
          {arrow && (
            <line x1={arrow.mx * 100} y1={arrow.my * 100} x2={arrow.ex * 100} y2={arrow.ey * 100} stroke={LINE} strokeWidth="3" strokeDasharray="3 2" vectorEffect="non-scaling-stroke" />
          )}
        </svg>
        {[...line.map((p) => [p, LINE]), ...roi.map((p) => [p, ROI])].map(([[x, y], c], i) => (
          <span key={i} className="absolute w-2.5 h-2.5 -ml-[5px] -mt-[5px] rounded-full border-2 border-white pointer-events-none" style={{ left: `${x * 100}%`, top: `${y * 100}%`, background: c }} />
        ))}
        {arrow && (
          <span
            className="absolute -translate-x-1/2 -translate-y-1/2 text-[11px] font-semibold text-black px-1.5 py-0.5 rounded pointer-events-none"
            style={{ left: `${arrow.ex * 100}%`, top: `${arrow.ey * 100}%`, background: LINE }}
          >
            Office side
          </span>
        )}
      </div>

      <p className="text-xs text-slate-500">
        {mode === "line"
          ? "Click two points across the doorway, on the floor where people's feet cross. Then check the arrow points into the office."
          : "Click the corners of the area to count in (leave empty to use the whole picture). People outside it, like someone at a reception desk, are ignored."}
      </p>

      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => setInsideSign((s) => -s)} disabled={line.length !== 2} className="btn-secondary text-sm flex items-center gap-1.5 disabled:opacity-50">
          <ArrowLeftRight size={14} /> Flip office side
        </button>
        <button
          type="button"
          onClick={() => (mode === "line" ? setLine([]) : setRoi((p) => p.slice(0, -1)))}
          className="btn-secondary text-sm flex items-center gap-1.5"
        >
          <Undo2 size={14} /> {mode === "line" ? "Clear line" : "Undo point"}
        </button>
        {mode === "roi" && roi.length > 0 && (
          <button type="button" onClick={() => setRoi([])} className="btn-secondary text-sm">
            Clear area
          </button>
        )}
        <span className="flex items-center gap-2 text-sm text-ink-900 ml-auto">
          <ToggleSwitch checked={enabled} onChange={setEnabled} /> Count at this camera
        </span>
        <button type="button" onClick={save} disabled={saving || (enabled && line.length !== 2) || (roi.length > 0 && roi.length < 3)} className="btn-primary text-sm disabled:opacity-50">
          {saving ? "Saving…" : "Save"}
        </button>
      </div>
      {message && <p className="text-sm text-ink-900">{message}</p>}
    </div>
  );
}
