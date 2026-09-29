import { useEffect, useRef, useState } from "react";
import { WS_HOST, WS_PROTOCOL } from "../api/client";

// Person-first overlay: every tracked person the backend sends gets a box
// (their full body, from face_pipeline.py's person tracks) labelled "Person".
// The label becomes the employee's name only once face recognition has
// confidently and stably identified them; the backend keeps sending that name
// for its identity-hold period (IDENTITY_GRACE_SECONDS) after the face is
// lost, then the track goes back to no identity and the label to "Person".
// "Unknown" is never shown: anything that isn't a real name reads "Person".
const GENERIC_LABEL = "Person";

// Labels that mean "not identified". An older recognition component could
// send one of these as a name; they're shown as "Person", never as text.
const NON_IDENTITY_LABELS = new Set(["unknown", "person", "unidentified", "face", "no match", "no_match"]);

function labelOf(det) {
  if (!det || !det.employee_id) return GENERIC_LABEL;
  const name = typeof det.name === "string" ? det.name.trim() : "";
  if (!name || NON_IDENTITY_LABELS.has(name.toLowerCase())) return GENERIC_LABEL;
  return name;
}

// One style for everyone, "Person" and named people alike. Identity never
// changes the colour (no per-company or per-employee colours).
const OVERLAY_COLOR = "#2563eb";

// Draws one person's box and label, in the camera's full-resolution pixels
// (the caller scales the canvas when the picture is smaller, e.g. 960 px grid
// tiles). `k` undoes that scale for line and text sizes, so the box and label
// look the same whatever the picture's resolution.
function drawDetection(ctx, det, k = 1) {
  const [x1, y1, x2, y2] = det.bbox;
  const label = labelOf(det);

  ctx.lineWidth = 2 * k;
  ctx.strokeStyle = OVERLAY_COLOR;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  ctx.font = `600 ${13 * k}px sans-serif`;
  const padding = 5 * k;
  const labelHeight = 20 * k;
  const labelWidth = ctx.measureText(label).width + padding * 2;
  const labelY = y1 - labelHeight >= 0 ? y1 - labelHeight : y1;
  ctx.fillStyle = OVERLAY_COLOR;
  ctx.fillRect(x1, labelY, labelWidth, labelHeight);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, x1 + padding, labelY + labelHeight - 6 * k);
}

// Opens the backend's per-camera live-view websocket (camera_stream.py) and
// paints each incoming JPEG frame onto a canvas, plus a second websocket
// (main.py's /ws/detections) for the live person overlay: a box per tracked
// person, labelled "Person" or, once confidently recognised, their name (see
// drawDetection). Shared by the grid tile and the
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
    const liveUrl = `${WS_PROTOCOL}://${WS_HOST}/ws/live/${camera.id}?token=${encodeURIComponent(token)}${overlay ? "" : "&plain=1"}${width ? `&w=${width}` : ""}`;
    const detUrl = `${WS_PROTOCOL}://${WS_HOST}/ws/detections/${camera.id}?token=${encodeURIComponent(token)}`;

    // A dropped socket (backend restart, network blip, camera reconnect)
    // reconnects on its own with backoff (1 s doubling to 30 s) instead of
    // leaving the tile "offline" until a page reload. Not retried when the
    // server refused access (4401 not authenticated / 4403 feed off) — that
    // won't fix itself by retrying.
    let disposed = false;
    const timers = new Set();
    function keepConnected(url, setup, onDown) {
      let delay = 1000;
      let current = null;
      const open = () => {
        if (disposed) return;
        const sock = new WebSocket(url);
        current = sock;
        setup(sock, () => (delay = 1000));
        sock.onclose = (ev) => {
          onDown?.();
          if (disposed || ev.code === 4401 || ev.code === 4403) return;
          const t = setTimeout(() => {
            timers.delete(t);
            open();
          }, delay);
          timers.add(t);
          delay = Math.min(delay * 2, 30000);
        };
      };
      open();
      return () => current?.close();
    }

    const closeLive = keepConnected(
      liveUrl,
      (sock, resetBackoff) => {
        sock.binaryType = "blob";
        sock.onopen = () => {
          resetBackoff();
          setStatus("live");
        };
        sock.onerror = () => setStatus("offline");
        sock.onmessage = (event) => {
          if (objectUrl) URL.revokeObjectURL(objectUrl);
          objectUrl = URL.createObjectURL(event.data);
          img.src = objectUrl;
        };
      },
      () => setStatus("offline")
    );
    img.onload = () => {
      if (canvas.width !== img.width || canvas.height !== img.height) {
        canvas.width = img.width;
        canvas.height = img.height;
      }
      redraw();
    };

    // Best-effort: if this fails to connect for any reason, the video feed
    // above still works — this only adds the overlay on top of it.
    const closeDet = overlay
      ? keepConnected(detUrl, (sock, resetBackoff) => {
          sock.onopen = resetBackoff;
          sock.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              detectionsRef.current = { people: data.people || [], frameW: data.frame_w || null };
            } catch {
              // malformed payload — keep showing the last good overlay
            }
            redraw();
          };
          sock.onerror = () => {};
        })
      : null;

    return () => {
      disposed = true;
      timers.forEach(clearTimeout);
      closeLive();
      closeDet?.();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [camera?.id, camera?.isConfigured, camera?.feedOn, overlay, width]);

  return { canvasRef, status };
}
