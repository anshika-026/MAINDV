import { useEffect, useState } from "react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import * as api from "../api/client";
import { setVisibleInterval, clearVisibleInterval } from "../lib/visibleInterval";

// Latest object detections per camera (backend/app/object_detection/):
// backpack, handbag, bottle and laptop, refreshed every few seconds. Each
// camera is looked at every ~5 s by default, so numbers move in steps.

const REFRESH_MS = 10_000;
const CLASSES = [
  ["laptop", "Laptops"],
  ["backpack", "Backpacks"],
  ["handbag", "Handbags"],
  ["bottle", "Bottles"],
];

function timeLabel(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function ObjectDetection() {
  const [data, setData] = useState(null);
  const [names, setNames] = useState({});
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .getCameras()
      .then((cams) => !cancelled && setNames(Object.fromEntries(cams.map((c) => [c.id, c.label]))))
      .catch(() => {});
    const load = () =>
      api
        .getObjectDetections()
        .then((d) => {
          if (!cancelled) {
            setData(d);
            setError("");
          }
        })
        .catch((e) => !cancelled && setError(e.message || "Couldn't load object detection."));
    load();
    const id = setVisibleInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearVisibleInterval(id);
    };
  }, []);

  if (error && !data) return <p className="text-sm text-danger-500">{error}</p>;
  if (!data) return <p className="text-sm text-slate-400">Loading object detection…</p>;

  const cameras = data.cameras || [];
  const totals = Object.fromEntries(CLASSES.map(([k]) => [k, cameras.reduce((n, c) => n + (c.stale ? 0 : c.counts?.[k] || 0), 0)]));

  const columns = [
    { key: "camera", label: "Camera" },
    ...CLASSES.map(([k, label]) => ({ key: k, label })),
    { key: "objects", label: "Detected (confidence)" },
    { key: "updated", label: "Updated" },
  ];
  const rows = cameras.map((c) => ({
    id: c.camera_id,
    camera: names[c.camera_id] || `Camera ${c.camera_id}`,
    ...Object.fromEntries(CLASSES.map(([k]) => [k, c.counts?.[k] ?? 0])),
    objects: c.objects.length
      ? c.objects.map((o) => `${o.class}${o.color ? ` (${o.color})` : ""} ${Math.round(o.confidence * 100)}%`).join(", ")
      : "—",
    updated: (
      <span className={c.stale ? "text-danger-500" : "text-slate-500"}>
        {timeLabel(c.ts)}
        {c.stale ? " · stale" : ""}
      </span>
    ),
  }));

  return (
    <div className="space-y-5">
      <PageHeader title="Object Detection" />

      {!data.enabled && (
        <p className="card px-4 py-3 text-sm text-slate-600">
          Object detection is switched off. An administrator can turn on <strong>Object detection</strong> in the
          Analytics switches (sidebar). It uses about 0.6 s of CPU per camera look.
        </p>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {CLASSES.map(([k, label]) => (
          <StatCard key={k} label={label} value={totals[k]} sub="visible now, all cameras" subTone="neutral" />
        ))}
      </div>

      <DataTable
        columns={columns}
        rows={rows}
        emptyLabel="No results yet. Cameras with analytics on are checked every few seconds while the feature is on."
      />
    </div>
  );
}
