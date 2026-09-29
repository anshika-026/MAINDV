import { useEffect, useRef, useState } from "react";
import { WS_HOST, WS_PROTOCOL } from "../api/client";

// Placeholder labels that mean "we could not identify this person". None
// of them are ever rendered — an unrecognized person gets NO overlay at
// all. Kept as a defensive filter here at the display layer so that a
// value like "Unknown" arriving from an older recognition component still
// can't reach the screen.
const NON_IDENTITY_LABELS = new Set([
  "unknown",
  "person",
  "unidentified",
  "face",
  "no match",
  "no_match",
]);

// The overlay is identity-only by design. Person detection and tracking
// keep running in the backend and the detection payload still carries
// bboxes and colours (other functionality depends on both — see
// face_pipeline.py's _update_person_overlay), but the generic person
// rectangle is never drawn: nothing appears on screen unless face
// recognition produced a confident identity. Returns the name to
// display, or null when there is nothing to show.
function recognizedNameOf(det) {
  if (!det || !det.employee_id) return null;
  const name = typeof det.name === "string" ? det.name.trim() : "";
  if (!name) return null;
  if (NON_IDENTITY_LABELS.has(name.toLowerCase())) return null;
  return name;
}

// Only ever used if a detection somehow arrives without a colour — the
// same neutral this file has always used for that case. The real colours
// come from the employee -> company -> colour pipeline in
// employee_directory.py (resolved server-side, sent as det.color); this
// file never picks, overrides or maps a colour itself.
const FALLBACK_LABEL_COLOR = "#6b7280";

// Draws ONLY a recognized person's name, positioned from the backend's bbox
// (in the camera's full-resolution pixels; the caller scales the canvas when
// the picture is smaller, e.g. 960 px grid tiles).
//
// The generic person rectangle is deliberately NOT drawn — the bbox is
// used purely to position the name over that person. The name keeps the
// existing colour-coded label styling (company colour behind white text),
// which is what makes every company colour legible over live video,
// including Shaurrya's #000000.
//
// `k` undoes the canvas scale for text sizes, so the label stays the same
// readable size whatever the picture's resolution.
function drawDetection(ctx, det, k = 1) {
  const name = recognizedNameOf(det);
  if (!name) return; // unrecognized -> intentionally nothing rendered

  // That person's existing company colour, exactly as assigned server-side.
  const color = det.color || FALLBACK_LABEL_COLOR;

  const [x1, y1, x2] = det.bbox;
  ctx.font = `600 ${13 * k}px sans-serif`;
  const padding = 4 * k;
  const labelHeight = 18 * k;
  const labelWidth = ctx.measureText(name).width + padding * 2;
  // Centred over the person rather than pinned to the (now invisible)
  // box's left edge, so the name still reads as belonging to them.
  const labelX = (x1 + x2) / 2 - labelWidth / 2;
  const labelY = y1 - labelHeight >= 0 ? y1 - labelHeight : y1;

  ctx.fillStyle = color;
  ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(name, labelX + padding, labelY + labelHeight - 5 * k);
}

// Opens the backend's per-camera live-view websocket (camera_stream.py) and
// paints each incoming JPEG frame onto a canvas, plus a second websocket
// (main.py's /ws/detections) for the live IDENTITY overlay. The backend
// still detects and tracks every person and still sends their bbox; this
// overlay deliberately renders only the names of people face recognition
// has confidently and stably identified, and renders nothing at all for
// anyone it hasn't (see drawDetection). Shared by the grid tile and the
// enlarged viewer modal so both draw from their own independent
// connections using identical wiring.
// Pass { overlay: false } for plain video: no detections websocket, and the
// live socket asks the backend not to run overlay detection for this viewer.
// width: ask the backend for frames scaled to this many pixels wide (grid
// tiles use 960, ~4x less bandwidth than full HD); omit for full resolution.
export default function useLiveCameraFeed(camera, { overlay = true, width } = {}) {
  const canvasRef = useRef(null);
  const [status, setStatus] = useState(camera?.isConfigured && camera?.feedOn !== false ? "connecting" : "offline");
  const detectionsRef = useRef({ people: [], frameW: null }); // latest detections, redrawn on every frame

  useEffect(() => {
    if (!camera?.isConfigured) {
      setStatus("offline");
      return;
    }
    if (camera.feedOn === false) {
      setStatus("disabled");
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    let objectUrl = null;

    function redraw() {
      if (!img.width) return;
      ctx.drawImage(img, 0, 0);
      // Boxes arrive in the camera's full-resolution pixels; scale them to
      // the (possibly smaller) picture being shown.
      const { people, frameW } = detectionsRef.current;
      const scale = frameW ? img.width / frameW : 1;
      ctx.save();
      ctx.scale(scale, scale);
      for (const det of people) drawDetection(ctx, det, 1 / scale);
      ctx.restore();
    }

    // Browsers can't attach an Authorization header to a WebSocket
    // handshake, so the session token travels as a query param instead —
    // the backend (main.py's _authorize_camera_ws) looks it up the same
    // way it would a header, and closes the connection if it's missing,
    // expired, or doesn't own this camera.
    const token = localStorage.getItem("deco_token") || "";
    const ws = new WebSocket(
      `${WS_PROTOCOL}://${WS_HOST}/ws/live/${camera.id}?token=${encodeURIComponent(token)}${overlay ? "" : "&plain=1"}${width ? `&w=${width}` : ""}`
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
    const detWs = overlay
      ? new WebSocket(`${WS_PROTOCOL}://${WS_HOST}/ws/detections/${camera.id}?token=${encodeURIComponent(token)}`)
      : null;
    if (detWs) {
      detWs.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          detectionsRef.current = { people: data.people || [], frameW: data.frame_w || null };
        } catch {
          // malformed payload — keep showing the last good overlay
        }
        redraw();
      };
      detWs.onerror = () => {};
    }

    return () => {
      ws.close();
      detWs?.close();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [camera?.id, camera?.isConfigured, camera?.feedOn, overlay, width]);

  return { canvasRef, status };
}
