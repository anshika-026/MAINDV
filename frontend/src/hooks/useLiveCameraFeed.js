import { useEffect, useRef, useState } from "react";
import { WS_HOST, WS_PROTOCOL } from "../api/client";

// Box color is company-based, resolved server-side from the recognized
// employee_id (see employee_directory.py) and sent as det.color on every
// detection. This is only the fallback for an unrecognized person (no
// employee_id) or a detection sent by a backend that predates the "color"
// field — kept visually distinct from every company color so it's never
// mistaken for one. Never a guessed company. No "Unknown" anywhere (see
// face_pipeline.py's PersonTrackState: a person with no confident identity
// is still a real, continuously-tracked person, just without a name
// attached yet).
const NEUTRAL_BOX_COLOR = "#6b7280";

// One-off display exception, requested for employee 026 (Mahesh
// Chaudhary) specifically: his box border is split per-side instead of
// the normal single company color. Purely a rendering choice — his
// company/label color (still resolved from det.color, same as everyone
// else) is untouched, only which color each border segment uses.
const MAHESH_EMPLOYEE_ID = "026";
const MAHESH_BORDER_COLORS = { top: "#000000", right: "#2563eb", bottom: "#000000", left: "#f97316" };

// Draws a rectangle as four independently-colored line segments rather
// than one strokeRect() — still one continuous box, just each side its
// own color. `lineCap = "square"` extends each segment half a line-width
// past its endpoint, which is what makes adjoining sides meet cleanly at
// the corners instead of leaving a notch (butt-capped segments don't
// cover the corner pixels where two colors meet).
function drawFourSidedBox(ctx, x1, y1, x2, y2, lineWidth) {
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "square";
  const sides = [
    [x1, y1, x2, y1, MAHESH_BORDER_COLORS.top],
    [x2, y1, x2, y2, MAHESH_BORDER_COLORS.right],
    [x2, y2, x1, y2, MAHESH_BORDER_COLORS.bottom],
    [x1, y2, x1, y1, MAHESH_BORDER_COLORS.left],
  ];
  for (const [sx1, sy1, sx2, sy2, sideColor] of sides) {
    ctx.strokeStyle = sideColor;
    ctx.beginPath();
    ctx.moveTo(sx1, sy1);
    ctx.lineTo(sx2, sy2);
    ctx.stroke();
  }
}

// Draws one PERSON's box (backend already sends the full body box, not a
// face box — see face_pipeline.py's _update_person_overlay) + a compact
// name-or-"Person" label directly onto the video canvas, in the same pixel
// coordinate space the backend's bbox is already in (camera_stream.py
// feeds face_pipeline the exact frame it also JPEG-encodes for /ws/live —
// no resizing in between), so this needs no scaling math even though the
// canvas itself is stretched via CSS.
function drawDetection(ctx, det) {
  const [x1, y1, x2, y2] = det.bbox;
  const color = det.employee_id ? det.color || NEUTRAL_BOX_COLOR : NEUTRAL_BOX_COLOR;
  const lineWidth = Math.max(2, (x2 - x1) * 0.01);
  if (det.employee_id === MAHESH_EMPLOYEE_ID) {
    drawFourSidedBox(ctx, x1, y1, x2, y2, lineWidth);
  } else {
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  // Plain name (or "Person") — no confidence percentage or extra text, to
  // stay a compact label rather than a UI badge.
  const label = det.employee_id ? det.name || det.employee_id : "Person";
  ctx.font = "600 13px sans-serif";
  const padding = 4;
  const textWidth = ctx.measureText(label).width;
  const labelHeight = 18;
  const labelY = y1 - labelHeight >= 0 ? y1 - labelHeight : y1;
  ctx.fillStyle = color;
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
