import { useEffect, useRef, useState } from "react";
import { RefreshCw } from "lucide-react";
import * as api from "../api/client";

// Draw one gate's counting zone on a live still: drag a box over the
// doorway. Only people whose body centre is inside it are counted, which
// keeps seating areas and passers-by at the edge of the view out of the
// unique count. A box rather than a thin line on purpose: each gate is
// sampled about once a second, so a walker would often cross a thin line
// between two looks and never be seen on it.

function toPolygon({ x0, y0, x1, y1 }) {
  const [l, r] = [Math.min(x0, x1), Math.max(x0, x1)];
  const [t, b] = [Math.min(y0, y1), Math.max(y0, y1)];
  return [[l, t], [r, t], [r, b], [l, b]].map(([x, y]) => [+x.toFixed(4), +y.toFixed(4)]);
}

function boxOf(roi) {
  if (!roi?.length) return null;
  const xs = roi.map((p) => p[0]);
  const ys = roi.map((p) => p[1]);
  return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) };
}

export default function ZoneEditor({ gate, onSaved }) {
  const [frameUrl, setFrameUrl] = useState(null);
  const [frameError, setFrameError] = useState("");
  const [box, setBox] = useState(() => boxOf(gate.zone));
  const [dragging, setDragging] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const areaRef = useRef(null);

  function loadFrame() {
    setFrameError("");
    api
      .fetchFootfallFrameObjectUrl(gate.camera_id)
      .then((u) =>
        setFrameUrl((old) => {
          if (old) URL.revokeObjectURL(old);
          return u;
        })
      )
      .catch((e) => setFrameError(e.message));
  }

  useEffect(() => {
    setBox(boxOf(gate.zone));
    setMessage("");
    loadFrame();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gate.camera_id]);

  function point(e) {
    const r = areaRef.current.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
  }

  function onDown(e) {
    e.currentTarget.setPointerCapture(e.pointerId);
    const [x, y] = point(e);
    setBox({ x0: x, y0: y, x1: x, y1: y });
    setDragging(true);
    setMessage("");
  }
  function onMove(e) {
    if (!dragging) return;
    const [x, y] = point(e);
    setBox((b) => ({ ...b, x1: x, y1: y }));
  }
  function onUp() {
    setDragging(false);
    setBox((b) => (b && Math.abs(b.x1 - b.x0) > 0.02 && Math.abs(b.y1 - b.y0) > 0.02 ? b : null));
  }

  async function save(roi) {
    setSaving(true);
    try {
      await api.setFootfallZone(gate.camera_id, roi);
      setMessage(roi ? "Zone saved. Only people inside it are counted at this gate." : "Zone cleared. The whole view counts.");
      onSaved?.();
    } catch {
      setMessage("Couldn't save the zone. Check the backend is running.");
    } finally {
      setSaving(false);
    }
  }

  const b = box && {
    left: `${Math.min(box.x0, box.x1) * 100}%`,
    top: `${Math.min(box.y0, box.y1) * 100}%`,
    width: `${Math.abs(box.x1 - box.x0) * 100}%`,
    height: `${Math.abs(box.y1 - box.y0) * 100}%`,
  };

  return (
    <div className="space-y-3">
      <div
        ref={areaRef}
        className="relative w-full aspect-video bg-[#11151c] rounded-lg overflow-hidden select-none touch-none cursor-crosshair"
        onPointerDown={frameUrl ? onDown : undefined}
        onPointerMove={onMove}
        onPointerUp={onUp}
      >
        {frameUrl ? (
          <img src={frameUrl} alt={`Live still from ${gate.name}`} className="w-full h-full object-fill pointer-events-none" draggable={false} />
        ) : (
          <p className="absolute inset-0 flex items-center justify-center text-sm text-slate-300 px-4 text-center">
            {frameError || "Loading a still from the camera…"}
          </p>
        )}
        {b && (
          <div className="absolute border-2 border-brand-500 bg-brand-500/20 pointer-events-none" style={b}>
            <span className="absolute -top-6 left-0 text-[11px] font-medium bg-brand-500 text-white px-1.5 py-0.5 rounded">
              Counting zone
            </span>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button onClick={() => save(box ? toPolygon(box) : null)} disabled={saving || !box} className="btn-primary disabled:opacity-60">
          {saving ? "Saving…" : "Save zone"}
        </button>
        <button
          onClick={() => {
            setBox(null);
            save(null);
          }}
          disabled={saving}
          className="px-3.5 py-2 rounded-lg text-sm font-medium border border-border-200 text-slate-600"
        >
          Count whole view
        </button>
        <button onClick={loadFrame} className="px-3 py-2 rounded-lg text-sm text-slate-500 flex items-center gap-1.5">
          <RefreshCw size={14} /> New still
        </button>
        <span className="text-xs text-slate-500">
          {gate.zone ? "This gate has a zone." : "No zone yet: the whole view counts."} Drag on the picture to draw one.
        </span>
      </div>
      {message && <p className="text-sm text-ink-900">{message}</p>}
    </div>
  );
}
