import { useEffect, useState } from "react";
import { ShieldAlert, X } from "lucide-react";
import useLiveCameraFeed from "../hooks/useLiveCameraFeed";

function formatTimestamp(date) {
  const pad = (n) => String(n).padStart(2, "0");
  const time = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  return `${pad(date.getDate())}/${pad(date.getMonth() + 1)}/${date.getFullYear()} ${time}`;
}

// Full-size single-camera viewer opened by clicking a LiveCameraTile —
// same feed, same websocket wiring, just enlarged with a live clock overlay
// instead of the tile's compact badge.
export default function CameraViewerModal({ camera, onClose }) {
  const { canvasRef, status } = useLiveCameraFeed(camera);
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    if (!camera) return;
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, [camera]);

  if (!camera) return null;

  const isLive = status === "live";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4">
      <div className="w-full max-w-4xl bg-white rounded-2xl shadow-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-ink-900">{camera.label}</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600" title="Close">
            <X size={20} />
          </button>
        </div>

        <div className="aspect-video bg-ink-900 rounded-xl overflow-hidden relative">
          <canvas ref={canvasRef} className="w-full h-full object-cover" />

          {!isLive && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-white/40 text-sm">
              <ShieldAlert size={28} />
              {status === "connecting" ? "Connecting…" : "Offline"}
            </div>
          )}

          <span className="absolute top-3 left-3 flex items-center gap-1.5 text-[11px] font-semibold text-white bg-black/50 rounded-full px-2.5 py-1 tracking-wide">
            <span className={`w-1.5 h-1.5 rounded-full ${isLive ? "bg-danger-500 animate-pulse" : "bg-slate-400"}`} />
            {isLive ? "LIVE" : status === "connecting" ? "CONNECTING" : "OFFLINE"}
          </span>

          <div className="absolute top-3 right-3 text-right text-white bg-black/40 rounded-md px-2.5 py-1">
            <p className="text-[11px] font-semibold tracking-wide">{isLive ? "PLAYING (HD)" : "PAUSED"}</p>
            <p className="text-[11px] font-mono text-white/80">{formatTimestamp(now)}</p>
          </div>
        </div>

        <p className="mt-3 text-xs text-slate-400">
          {camera.site} · {camera.code}
        </p>
      </div>
    </div>
  );
}
