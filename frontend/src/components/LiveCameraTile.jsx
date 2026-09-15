import { useEffect, useRef, useState } from "react";
import { ShieldAlert } from "lucide-react";
import { WS_HOST, WS_PROTOCOL } from "../api/client";

// Opens the backend's per-camera live-view websocket (camera_stream.py) and
// paints each incoming JPEG frame onto a canvas — same wire protocol the
// previous frontend's CameraTile used, restyled for this design system.
export default function LiveCameraTile({ camera }) {
  const canvasRef = useRef(null);
  const [status, setStatus] = useState(camera.isConfigured ? "connecting" : "offline");

  useEffect(() => {
    if (!camera.isConfigured) {
      setStatus("offline");
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    let objectUrl = null;

    const ws = new WebSocket(`${WS_PROTOCOL}://${WS_HOST}/ws/live/${camera.id}`);
    ws.binaryType = "blob";
    ws.onopen = () => setStatus("live");
    ws.onclose = () => setStatus("offline");
    ws.onerror = () => setStatus("offline");
    ws.onmessage = (event) => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(event.data);
      img.src = objectUrl;
    };
    img.onload = () => {
      if (canvas.width !== img.width || canvas.height !== img.height) {
        canvas.width = img.width;
        canvas.height = img.height;
      }
      ctx.drawImage(img, 0, 0);
    };

    return () => {
      ws.close();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [camera.id, camera.isConfigured]);

  const isLive = status === "live";

  return (
    <div className="card overflow-hidden hover:shadow-md transition-shadow">
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
    </div>
  );
}
