import { useCallback, useState } from "react";
import PolygonZoneEditor from "./PolygonZoneEditor";
import * as api from "../api/client";

// Mark desks on a camera's live picture (backend/app/desks.py). Anyone whose
// face is identified inside a desk outline is counted as at that desk, so only
// cameras with face recognition on can track desks.
export default function DeskZoneEditor({ cameras, onChanged }) {
  const usable = cameras.filter((c) => c.attendanceTracking);
  const [label, setLabel] = useState("");

  const loadZones = useCallback(
    (cameraId) => api.listDeskZones(cameraId).then((zs) => zs.map((z) => ({ id: z.id, label: z.zone_label, polygon: z.polygon }))),
    []
  );

  if (!usable.length) {
    return (
      <p className="text-sm text-slate-500">
        No camera has face recognition switched on. Desks need it to know who is sitting there: edit a camera in
        Camera Management and tick "Enable attendance tracking".
      </p>
    );
  }

  return (
    <PolygonZoneEditor
      cameras={usable}
      loadZones={loadZones}
      saveZone={async (cameraId, points) => {
        const z = await api.createDeskZone(cameraId, points, label.trim());
        setLabel("");
        return z.zone_label;
      }}
      deleteZone={(z) => api.deleteDeskZone(z.id)}
      hint="Click each corner of one desk, around the chair where the person's face will be, then save."
      extraFields={
        <input
          id="desk-label"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="Desk name (optional, e.g. Desk 12)"
          className="input-field w-full"
        />
      }
      onChanged={onChanged}
    />
  );
}
