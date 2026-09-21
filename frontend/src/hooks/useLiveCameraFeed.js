import { useEffect, useRef, useState } from "react";
import { WS_HOST, WS_PROTOCOL } from "../api/client";

// One consistent color for every person box/label, recognized or not —
// box, "Person" text, and a recognized name are all this same blue. No
// green/amber/red split, no "Unknown" anywhere (see face_pipeline.py's
// PersonTrackState: a person with no confident identity is still a real,
// continuously-tracked person, just without a name attached yet).
const PERSON_BOX_COLOR = "#2563eb";

// Draws one PERSON's box (backend already sends the full body box, not a
// face box — see face_pipeline.py's _update_person_overlay) + a compact
// name-or-"Person" label directly onto the video canvas, in the same pixel
// coordinate space the backend's bbox is already in (camera_stream.py
// feeds face_pipeline the exact frame it also JPEG-encodes for /ws/live —
// no resizing in between), so this needs no scaling math even though the
// canvas itself is stretched via CSS.
function drawDetection(ctx, det) {
  const [x1, y1, x2, y2] = det.bbox;
  ctx.lineWidth = Math.max(2, (x2 - x1) * 0.01);
  ctx.strokeStyle = PERSON_BOX_COLOR;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  // Plain name (or "Person") — no confidence percentage or extra text, to
  // stay a compact label rather than a UI badge.
  const label = det.employee_id ? det.name || det.employee_id : "Person";
  ctx.font = "600 13px sans-serif";
  const padding = 4;
  const textWidth = ctx.measureText(label).width;
  const labelHeight = 18;
  const labelY = y1 - labelHeight >= 0 ? y1 - labelHeight : y1;
  ctx.fillStyle = PERSON_BOX_COLOR;
  ctx.fillRect(x1, labelY, textWidth + padding * 2, labelHeight);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, x1 + padding, labelY + labelHeight - 5);
}

// Opens the backend's per-camera live-view websocket (camera_stream.py) and
// paints each incoming JPEG frame onto a canvas, plus a second websocket
// (main.py's /ws/detections) for the live PERSON overlay — a full-body box
// per detected person, labeled with their recognized name once face
// recognition is confident and stable, or "Person" otherwise. A person's
// box never depends on their face being visible (see face_pipeline.py's
// person-first architecture). Shared by the grid tile and the enlarged
// viewer modal so both draw from their own independent connections using
// identical wiring.
export default function useLiveCameraFeed(camera) {
  const canvasRef = useRef(null);
  const [status, setStatus] = useState(camera?.isConfigured ? "connecting" : "offline");
  const detectionsRef = useRef([]); // latest detections, redrawn on top of every new frame

  useEffect(() => {
    if (!camera?.isConfigured) {
      setStatus("offline");
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    let objectUrl = null;

    function redraw() {
      if (!img.width) return;
      ctx.drawImage(img, 0, 0);
      for (const det of detectionsRef.current) drawDetection(ctx, det);
    }

    // Browsers can't attach an Authorization header to a WebSocket
    // handshake, so the session token travels as a query param instead —
    // the backend (main.py's _authorize_camera_ws) looks it up the same
    // way it would a header, and closes the connection if it's missing,
    // expired, or doesn't own this camera.
    const token = localStorage.getItem("deco_token") || "";
    const ws = new WebSocket(
      `${WS_PROTOCOL}://${WS_HOST}/ws/live/${camera.id}?token=${encodeURIComponent(token)}`
    );
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
      redraw();
    };

    // Best-effort: if this fails to connect for any reason, the video feed
    // above still works — this only adds the overlay on top of it.
    const detWs = new WebSocket(
      `${WS_PROTOCOL}://${WS_HOST}/ws/detections/${camera.id}?token=${encodeURIComponent(token)}`
    );
    detWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        detectionsRef.current = data.people || [];
      } catch {
        // malformed payload — keep showing the last good overlay
      }
      redraw();
    };
    detWs.onerror = () => {};

    return () => {
      ws.close();
      detWs.close();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [camera?.id, camera?.isConfigured]);

  return { canvasRef, status };
}
