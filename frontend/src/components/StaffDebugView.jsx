import { useEffect, useRef, useState } from "react";
import * as api from "../api/client";

// Staff Count debug view: the camera's live video with what the counter sees
// drawn on top (backend /ws/staff/debug): every tracked person's box, track
// id, name + confidence once identified, which side of the line they're on,
// the entry line, the office side, the office area, and the live count.
// Boxes: green = inside, red = outside, grey = on the line (not decided).

const SIDE_COLOR = { in: "#22c55e", out: "#ef4444" };

export default function StaffDebugView({ cameraId }) {
  const canvasRef = useRef(null);
  const debugRef = useRef(null);
  const [info, setInfo] = useState(null);
  const [status, setStatus] = useState("connecting");

  useEffect(() => {
    if (!cameraId) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    let url = null;

    function draw() {
      if (!img.width) return;
      ctx.drawImage(img, 0, 0);
      const d = debugRef.current;
      if (!d) return;
      const W = img.width;
      const H = img.height;
      const s = d.frame_w ? W / d.frame_w : 1;
      if (d.roi) {
        ctx.beginPath();
        d.roi.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
        ctx.closePath();
        ctx.fillStyle = "rgba(34,211,238,0.12)";
        ctx.fill();
        ctx.strokeStyle = "#22d3ee";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      if (d.line) {
        const [[x1, y1], [x2, y2]] = d.line;
        ctx.strokeStyle = "#facc15";
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(x1 * W, y1 * H);
        ctx.lineTo(x2 * W, y2 * H);
        ctx.stroke();
      }
      ctx.font = "600 13px sans-serif";
      for (const t of d.tracks) {
        if (!t.bbox || t.lost_for > 1) continue;
        const [bx1, by1, bx2, by2] = t.bbox.map((v) => v * s);
        const color = SIDE_COLOR[t.side] || "#9ca3af";
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);
        const who = t.name ? `${t.name} ${Math.round((t.confidence || 0) * 100)}%` : "unknown";
        const label = `#${t.track_id} ${who} · ${t.side === "in" ? "INSIDE" : t.side === "out" ? "OUTSIDE" : "on line"}`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(bx1, Math.max(0, by1 - 18), tw + 8, 18);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, bx1 + 4, Math.max(13, by1 - 5));
        if (t.last_event && t.event_age !== null && t.event_age < 4) {
          ctx.fillStyle = t.last_event === "ENTRY" ? "#22c55e" : "#ef4444";
          ctx.font = "700 16px sans-serif";
          ctx.fillText(t.last_event, bx1, by2 + 18);
          ctx.font = "600 13px sans-serif";
        }
      }
      const c = d.count || {};
      const banner = `Staff inside: ${c.current_staff_count ?? "-"}   Known ${c.known_staff ?? "-"} · Unknown ${c.unknown_persons ?? "-"}   In ${c.total_entries_today ?? "-"} / Out ${c.total_exits_today ?? "-"}`;
      ctx.font = "700 15px sans-serif";
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(8, 8, ctx.measureText(banner).width + 16, 26);
      ctx.fillStyle = "#fff";
      ctx.fillText(banner, 16, 26);
    }

    const video = new WebSocket(api.staffSocketUrl(`/ws/live/${cameraId}`) + "&plain=1&w=960");
    video.binaryType = "blob";
    video.onopen = () => setStatus("live");
    video.onclose = () => setStatus("offline");
    video.onmessage = (e) => {
      if (url) URL.revokeObjectURL(url);
      url = URL.createObjectURL(e.data);
      img.src = url;
    };
    img.onload = () => {
      if (canvas.width !== img.width) {
        canvas.width = img.width;
        canvas.height = img.height;
      }
      draw();
    };
    const dbg = new WebSocket(api.staffSocketUrl(`/ws/staff/debug/${cameraId}`));
    dbg.onmessage = (e) => {
      try {
        debugRef.current = JSON.parse(e.data);
        setInfo(debugRef.current);
      } catch {
        // keep the last good overlay
      }
    };
    return () => {
      video.close();
      dbg.close();
      if (url) URL.revokeObjectURL(url);
    };
  }, [cameraId]);

  return (
    <div className="space-y-2">
      <div className="relative w-full aspect-video bg-[#11151c] rounded-lg overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-full" />
        {status !== "live" && (
          <p className="absolute inset-0 flex items-center justify-center text-sm text-slate-300">{status === "connecting" ? "Connecting…" : "No video from this camera"}</p>
        )}
      </div>
      <p className="text-xs text-slate-500">
        {info?.configured
          ? info.processing
            ? `Counting live: ${info.tracks.length} tracked. Green = inside, red = outside, grey = on the line.`
            : info.failed
              ? "Counting failed on this camera. Check the backend log."
              : "Waiting for frames. Is the Staff count switch on?"
          : "This camera isn't set up as an entrance yet."}
      </p>
    </div>
  );
}
