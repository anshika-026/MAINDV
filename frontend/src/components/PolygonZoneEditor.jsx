import { useCallback, useEffect, useRef, useState } from "react";
import { RefreshCw, Undo2, X } from "lucide-react";
import * as api from "../api/client";

// Draw polygon zones on a camera's live picture: click each corner, then save.
// Shared by desk analytics (DeskZoneEditor) and intrusion zones. Points are
// fractions of the frame (0..1), so zones survive a resolution change.
//
// Props:
//   cameras        cameras to choose from ({ id, label })
//   loadZones      (cameraId) => Promise<[{ id, label, polygon }]>
//   saveZone       (cameraId, points) => Promise<string>   resolves the saved zone's name
//   deleteZone     (zone) => Promise
//   extraFields    form fields shown above the Save button (e.g. name, hours)
//   canSave        extra condition for saving (default true)
//   color          outline colour for saved zones
//   hint           one line under the picture explaining where to click

const DRAFT = "#ec4899"; // the outline being drawn: must read on any camera picture

export default function PolygonZoneEditor({
  cameras, loadZones, saveZone, deleteZone, extraFields, canSave = true, color = "#22c55e", hint, onChanged,
}) {
  const [cameraId, setCameraId] = useState(() => cameras[0]?.id ?? null);
  const [frameUrl, setFrameUrl] = useState(null);
  const [frameError, setFrameError] = useState("");
  const [zones, setZones] = useState([]);
  const [points, setPoints] = useState([]);
  const [saving, setSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(null);
  const [message, setMessage] = useState("");
  const areaRef = useRef(null);

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

  const refreshZones = useCallback(() => {
    if (!cameraId) return;
    loadZones(cameraId).then(setZones).catch(() => setZones([]));
  }, [cameraId, loadZones]);

  useEffect(() => {
    setFrameUrl(null);
    setPoints([]);
    setMessage("");
    loadFrame();
    refreshZones();
  }, [loadFrame, refreshZones]);

  function addPoint(e) {
    if (!frameUrl) return;
    const r = areaRef.current.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    setPoints((p) => [...p, [+x.toFixed(4), +y.toFixed(4)]]);
    setMessage("");
  }

  async function save() {
    setSaving(true);
    try {
      const name = await saveZone(cameraId, points);
      setPoints([]);
      setMessage(`${name} saved.`);
      refreshZones();
      onChanged?.();
    } catch (e) {
      setMessage(e.message?.includes("422") ? "Check the zone details and try again." : "Couldn't save. Check the backend is running.");
    } finally {
      setSaving(false);
    }
  }

  async function remove(zone) {
    await deleteZone(zone);
    setConfirmDelete(null);
    setMessage(`${zone.label} removed.`);
    refreshZones();
    onChanged?.();
  }

  const toSvg = (poly) => poly.map(([x, y]) => `${x * 100},${y * 100}`).join(" ");

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor="zone-camera" className="text-sm font-medium text-ink-900">
          Camera
        </label>
        <select id="zone-camera" value={cameraId ?? ""} onChange={(e) => setCameraId(Number(e.target.value))} className="input-field w-auto">
          {cameras.map((c) => (
            <option key={c.id} value={c.id}>
              {c.label}
            </option>
          ))}
        </select>
        <button type="button" onClick={loadFrame} className="px-2.5 py-2 rounded-lg text-sm text-slate-500 flex items-center gap-1.5">
          <RefreshCw size={14} /> New picture
        </button>
      </div>

      <div ref={areaRef} onClick={addPoint} className="relative w-full aspect-video bg-[#11151c] rounded-lg overflow-hidden select-none cursor-crosshair">
        {frameUrl ? (
          <img src={frameUrl} alt="Live picture from the camera" className="w-full h-full object-fill pointer-events-none" draggable={false} />
        ) : (
          <p className="absolute inset-0 flex items-center justify-center text-sm text-slate-300 px-4 text-center">
            {frameError || "Loading the camera picture…"}
          </p>
        )}
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 w-full h-full pointer-events-none">
          {zones.map((z) => (
            <polygon key={z.id} points={toSvg(z.polygon)} fill={`${color}2e`} stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
          ))}
          {points.length > 1 && (
            <polyline points={toSvg(points)} fill={points.length > 2 ? "rgba(236,72,153,0.18)" : "none"} stroke={DRAFT} strokeWidth="2" vectorEffect="non-scaling-stroke" />
          )}
          {points.length > 2 && (
            <line x1={points.at(-1)[0] * 100} y1={points.at(-1)[1] * 100} x2={points[0][0] * 100} y2={points[0][1] * 100}
              stroke={DRAFT} strokeWidth="1" strokeDasharray="4 3" vectorEffect="non-scaling-stroke" />
          )}
        </svg>
        {points.map(([x, y], i) => (
          <span key={i} className="absolute w-2.5 h-2.5 -ml-[5px] -mt-[5px] rounded-full border-2 border-white pointer-events-none"
            style={{ left: `${x * 100}%`, top: `${y * 100}%`, background: DRAFT }} />
        ))}
        {zones.map((z) => {
          const cx = z.polygon.reduce((s, p) => s + p[0], 0) / z.polygon.length;
          const cy = z.polygon.reduce((s, p) => s + p[1], 0) / z.polygon.length;
          return (
            <span key={z.id} className="absolute -translate-x-1/2 -translate-y-1/2 text-[11px] font-semibold text-white bg-black/60 px-1.5 py-0.5 rounded pointer-events-none"
              style={{ left: `${cx * 100}%`, top: `${cy * 100}%` }}>
              {z.label}
            </span>
          );
        })}
      </div>

      <p className="text-xs text-slate-500">
        {hint || "Click each corner of the area, then save."} {points.length} point{points.length === 1 ? "" : "s"} so far
        {points.length > 0 && points.length < 3 ? " (need at least 3)" : ""}.
      </p>

      {extraFields}

      <div className="flex flex-wrap items-center gap-2 justify-end">
        <button type="button" onClick={() => setPoints((p) => p.slice(0, -1))} disabled={!points.length} className="btn-secondary text-sm flex items-center gap-1.5 disabled:opacity-50">
          <Undo2 size={14} /> Undo point
        </button>
        <button type="button" onClick={save} disabled={points.length < 3 || saving || !canSave} className="btn-primary text-sm disabled:opacity-50">
          {saving ? "Saving…" : "Save zone"}
        </button>
      </div>
      {message && <p className="text-sm text-ink-900">{message}</p>}

      {zones.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {zones.map((z) =>
            confirmDelete === z.id ? (
              <span key={z.id} className="badge badge-danger flex items-center gap-1.5">
                Remove {z.label}?
                <button type="button" onClick={() => remove(z)} className="font-semibold underline">Yes</button>
                <button type="button" onClick={() => setConfirmDelete(null)}>No</button>
              </span>
            ) : (
              <span key={z.id} className="badge badge-neutral flex items-center gap-1">
                {z.label}
                <button type="button" aria-label={`Remove ${z.label}`} onClick={() => setConfirmDelete(z.id)}>
                  <X size={12} />
                </button>
              </span>
            )
          )}
        </div>
      )}
    </div>
  );
}
