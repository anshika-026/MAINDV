import { ShieldAlert } from "lucide-react";
import useLiveCameraFeed from "../hooks/useLiveCameraFeed";

// Grid tile for the Live feed page. Clicking it opens the same feed full
// size in CameraViewerModal — this component only owns the small preview.
export default function LiveCameraTile({ camera, onClick }) {
  const { canvasRef, status } = useLiveCameraFeed(camera);
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
              {status === "connecting" ? "Connecting…" : "Offline"}
            </div>
          </div>
        )}
        <span className={`absolute top-2 right-2 badge ${isLive ? "badge-success" : "badge-danger"}`}>
          <span
            className={`w-1.5 h-1.5 rounded-full ${isLive ? "bg-success-500 animate-pulse" : "bg-danger-500"}`}
          />
          {isLive ? "Live" : status === "connecting" ? "Connecting" : "Offline"}
        </span>
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
