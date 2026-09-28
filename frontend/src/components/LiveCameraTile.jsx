import { ShieldAlert } from "lucide-react";
import useLiveCameraFeed from "../hooks/useLiveCameraFeed";

// Grid tiles are small on screen, so they ask for 960 px frames instead of
// full HD: ~4.4 Mbps per camera instead of ~17 (measured). Clicking a tile
// opens CameraViewerModal, which streams full resolution.
const TILE_WIDTH = 960;

// Grid tile for the Live feed page. Clicking it opens the same feed full
// size in CameraViewerModal — this component only owns the small preview.
export default function LiveCameraTile({ camera, onClick, overlay = true }) {
  const { canvasRef, status } = useLiveCameraFeed(camera, { overlay, width: TILE_WIDTH });
  const isLive = status === "live";

  return (
    <button
      type="button"
      onClick={onClick}
      className="card overflow-hidden hover:shadow-md transition-shadow text-left w-full"
    >
      <div className="aspect-video bg-ink-900 relative flex items-center justify-center">
        <canvas ref={canvasRef} className="w-full h-full object-cover" />
        {!isLive && (
          <div className="absolute inset-0 flex items-center justify-center text-center text-white/40 text-xs">
            <div>
              <ShieldAlert size={24} className="mx-auto mb-1" />
              {status === "connecting" ? "Connecting…" : status === "disabled" ? "Feed switched off" : "Offline"}
            </div>
          </div>
        )}
        <span className={`absolute top-2 right-2 badge ${isLive ? "badge-success" : "badge-danger"}`}>
          <span
            className={`w-1.5 h-1.5 rounded-full ${isLive ? "bg-success-500 animate-pulse" : "bg-danger-500"}`}
          />
          {isLive ? "Live" : status === "connecting" ? "Connecting" : status === "disabled" ? "Off" : "Offline"}
        </span>
        {overlay && camera.attendanceTracking === false && (
          <span className="absolute top-2 left-2 badge badge-neutral">Face recognition off</span>
        )}
      </div>
      <div className="px-4 py-3 flex items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm font-medium text-ink-900 truncate">{camera.label}</p>
          <p className="text-xs text-slate-400 truncate">
            {camera.site} · {camera.code}
          </p>
        </div>
      </div>
    </button>
  );
}
