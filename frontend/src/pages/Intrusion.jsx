import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import Modal from "../components/Modal";
import ToggleSwitch from "../components/ToggleSwitch";
import StatusBadge from "../components/StatusBadge";
import PolygonZoneEditor from "../components/PolygonZoneEditor";
import * as api from "../api/client";
import { setVisibleInterval, clearVisibleInterval } from "../lib/visibleInterval";

// Restricted-area (intrusion) zones, drawn on a camera's live picture
// (backend/app/intrusion.py). Anyone whose feet are inside an active zone
// raises an "Intrusion Detected" alert with a snapshot, listed below and on
// the Alerts page.

const BAR_COLOR = "#4f5fea";
const ZONE_COLOR = "#ef4444";
const REFRESH_MS = 20000;

function clock(ts) {
  return ts ? new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "-";
}

export default function Intrusion() {
  const [zones, setZones] = useState([]);
  const [stats, setStats] = useState(null);
  const [recent, setRecent] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [drawOpen, setDrawOpen] = useState(false);
  const [form, setForm] = useState({ name: "", from: "", to: "" });
  const [confirmDelete, setConfirmDelete] = useState(null);
  const [error, setError] = useState("");

  const load = useCallback(() => {
    Promise.all([api.getIntrusionZones(), api.getIntrusionStats(), api.getAlerts("week")])
      .then(([z, s, a]) => {
        setZones(z);
        setStats(s);
        setRecent(a.filter((x) => x.event === "Intrusion Detected").slice(0, 8));
        setError("");
      })
      .catch(() => setError("Couldn't load intrusion data. Check the backend is running."));
  }, []);

  useEffect(() => {
    load();
    api.getCameras().then(setCameras).catch(() => setCameras([]));
    const t = setVisibleInterval(load, REFRESH_MS);
    return () => clearVisibleInterval(t);
  }, [load]);

  const loadZonesFor = useCallback(
    (cameraId) => api.getIntrusionZones(cameraId).then((zs) => zs.map((z) => ({ id: z.id, label: z.name, polygon: z.polygon }))),
    []
  );

  const windowValid = (!form.from && !form.to) || (form.from && form.to);
  const openAlerts = recent.filter((a) => a.status !== "Resolved").length;

  async function setEnabled(zone, enabled) {
    await api.updateIntrusionZone(zone.id, { enabled });
    load();
  }

  async function remove(zone) {
    await api.deleteIntrusionZone(zone.id);
    setConfirmDelete(null);
    load();
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Intrusion"
        action={
          <button onClick={() => setDrawOpen(true)} className="btn-primary text-sm">
            + Draw restricted zone
          </button>
        }
      />
      {error && <p className="text-sm text-danger-500">{error}</p>}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Restricted zones" value={zones.length} sub={`${zones.filter((z) => z.enabled).length} enabled`} subTone="neutral" />
        <StatCard label="Guarding now" value={zones.filter((z) => z.active_now).length} sub="inside their active hours" subTone="neutral" />
        <StatCard label="Intrusions today" value={stats?.today ?? "—"} sub="detections logged" subTone="danger" />
        <StatCard label="Open intrusion alerts" value={openAlerts} sub="this week, not resolved" subTone="danger" />
      </div>

      <div className="grid lg:grid-cols-3 gap-5">
        <div className="card p-5 lg:col-span-1">
          <h3 className="font-semibold text-ink-900 mb-4">Intrusions, last 7 days</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats?.per_day || []} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke="#eceef4" />
                <XAxis dataKey="day" tickLine={false} axisLine={false} tick={{ fontSize: 12, fill: "#94a3b8" }} />
                <YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{ fontSize: 12, fill: "#94a3b8" }} />
                <Tooltip />
                <Bar dataKey="count" name="Intrusions" fill={BAR_COLOR} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card p-5 lg:col-span-2">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-ink-900">Recent intrusions</h3>
            <Link to="/alerts" className="text-xs text-brand-600 font-medium">
              Open in Alerts
            </Link>
          </div>
          {recent.length === 0 ? (
            <p className="text-sm text-slate-500">No intrusions this week.</p>
          ) : (
            <div className="divide-y divide-[#f1f2f7]">
              {recent.map((a) => (
                <div key={a.id} className="flex items-center justify-between py-2 text-sm gap-3">
                  <span className="text-slate-500 w-24 shrink-0">
                    {a.date} {a.time}
                  </span>
                  <span className="text-ink-900 flex-1 truncate">{a.message}</span>
                  <span className="text-slate-400 hidden sm:inline">{a.camera}</span>
                  <StatusBadge value={a.status} />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <DataTable
        columns={[
          { key: "name", label: "Zone" },
          { key: "camera_name", label: "Camera" },
          {
            key: "hours",
            label: "Active hours",
            render: (z) => (z.active_from ? `${z.active_from} – ${z.active_to}` : "Always"),
          },
          {
            key: "state",
            label: "Now",
            render: (z) => <StatusBadge value={!z.enabled ? "Off" : z.active_now ? "Guarding" : "Outside hours"} />,
          },
          { key: "intrusions_today", label: "Intrusions today" },
          { key: "last", label: "Last intrusion", render: (z) => clock(z.last_intrusion) },
          {
            key: "enabled",
            label: "Enabled",
            render: (z) => <ToggleSwitch checked={z.enabled} onChange={(v) => setEnabled(z, v)} />,
          },
          {
            key: "action",
            label: "",
            render: (z) =>
              confirmDelete === z.id ? (
                <span className="text-sm flex items-center gap-2">
                  <button onClick={() => remove(z)} className="text-danger-500 font-medium">
                    Delete
                  </button>
                  <button onClick={() => setConfirmDelete(null)} className="text-slate-500">
                    Keep
                  </button>
                </span>
              ) : (
                <button onClick={() => setConfirmDelete(z.id)} className="text-slate-500 text-sm">
                  Remove
                </button>
              ),
          },
        ]}
        rows={zones}
        emptyLabel="No restricted zones yet. Draw one on a camera to start guarding it."
      />

      <Modal open={drawOpen} onClose={() => setDrawOpen(false)} title="Draw restricted zone" width="max-w-3xl">
        {drawOpen &&
          (cameras.length === 0 ? (
            <p className="text-sm text-slate-500">No cameras yet. Add one in Camera Management.</p>
          ) : (
            <PolygonZoneEditor
              cameras={cameras}
              loadZones={loadZonesFor}
              color={ZONE_COLOR}
              hint="Click each corner of the floor area to guard. Someone counts as inside when their feet are in it."
              canSave={!!form.name.trim() && windowValid}
              saveZone={async (cameraId, polygon) => {
                const z = await api.createIntrusionZone({ cameraId, name: form.name.trim(), polygon, from: form.from, to: form.to });
                setForm({ name: "", from: "", to: "" });
                return z.name;
              }}
              deleteZone={(z) => api.deleteIntrusionZone(z.id)}
              onChanged={load}
              extraFields={
                <div className="grid sm:grid-cols-3 gap-3">
                  <div className="sm:col-span-1">
                    <label htmlFor="iz-name" className="text-sm font-medium text-ink-900 block mb-1.5">
                      Zone name <span className="text-danger-500">*</span>
                    </label>
                    <input
                      id="iz-name"
                      value={form.name}
                      onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                      placeholder="e.g. Server room door"
                      className="input-field"
                    />
                  </div>
                  <div>
                    <label htmlFor="iz-from" className="text-sm font-medium text-ink-900 block mb-1.5">
                      Active from
                    </label>
                    <input id="iz-from" type="time" value={form.from} onChange={(e) => setForm((f) => ({ ...f, from: e.target.value }))} className="input-field" />
                  </div>
                  <div>
                    <label htmlFor="iz-to" className="text-sm font-medium text-ink-900 block mb-1.5">
                      Active until
                    </label>
                    <input id="iz-to" type="time" value={form.to} onChange={(e) => setForm((f) => ({ ...f, to: e.target.value }))} className="input-field" />
                  </div>
                  <p className="text-xs text-slate-500 sm:col-span-3 -mt-1">
                    Leave both times empty to guard around the clock. 19:00 to 08:00 guards overnight only.
                    {!windowValid && <span className="text-danger-500"> Set both times, or neither.</span>}
                  </p>
                </div>
              }
            />
          ))}
      </Modal>
    </div>
  );
}
