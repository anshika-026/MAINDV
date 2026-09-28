import { useEffect, useState } from "react";
import PageHeader from "../components/PageHeader";
import LiveCameraTile from "../components/LiveCameraTile";
import CameraViewerModal from "../components/CameraViewerModal";
import * as api from "../api/client";

// Plain live video from every camera: no boxes, names or analytics. Viewers
// here don't make the backend run any overlay detection (see useLiveCameraFeed's
// overlay option), so watching this page costs almost nothing extra. Face
// recognition, attendance and footfall keep running in the background either
// way; this page just doesn't draw them. AI Analytics is the view with overlays.
export default function PlainLiveFeed() {
  const [cameras, setCameras] = useState([]);
  const [expanded, setExpanded] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .getCameras()
      .then(setCameras)
      .catch(() => setError("Couldn't load cameras. Check the backend is running."));
  }, []);

  const online = cameras.filter((c) => c.status === "Active").length;

  return (
    <div className="space-y-5">
      <PageHeader title="Live Feed" />
      <p className="text-sm text-slate-500">
        {online} of {cameras.length} cameras online. Plain video only; open AI Analytics to see face recognition.
      </p>
      {error && <p className="text-sm text-danger-500">{error}</p>}

      {cameras.length === 0 && !error ? (
        <div className="card p-10 text-center text-sm text-slate-500">No cameras yet. Add one in Camera Management.</div>
      ) : (
        <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {cameras.map((cam) => (
            <LiveCameraTile key={cam.id} camera={cam} overlay={false} onClick={() => setExpanded(cam)} />
          ))}
        </div>
      )}

      <CameraViewerModal camera={expanded} overlay={false} onClose={() => setExpanded(null)} />
    </div>
  );
}
