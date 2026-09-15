import { useEffect, useState } from "react";
import { ChevronLeft, Search, X, User } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import PageHeader from "../components/PageHeader";
import DataTable from "../components/DataTable";
import Modal from "../components/Modal";
import ToggleSwitch from "../components/ToggleSwitch";
import * as api from "../api/client";
import {
  weeklyUnauthorized,
  zoneAccessDetail,
  defaultZoneAccess,
  waitingRoomStats,
  defaultWaitingRoomStats,
  recentDetections,
  defaultRecentDetections,
  peopleDirectory,
} from "../data/intrusionDetail";

const BAR_COLOR = "#4f5fea"; // brand-500 — single series, so one hue throughout

function MiniBarChart({ data, dataKey = "count", labelKey = "day", height = 200 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke="#eceef4" />
        <XAxis
          dataKey={labelKey}
          tickLine={false}
          axisLine={false}
          tick={{ fill: "#8a8a8a", fontSize: 12 }}
        />
        <YAxis tickLine={false} axisLine={false} tick={{ fill: "#8a8a8a", fontSize: 12 }} width={28} />
        <Tooltip
          cursor={{ fill: "#f5f6fa" }}
          contentStyle={{ borderRadius: 10, border: "1px solid #e7e8f0", fontSize: 12 }}
        />
        <Bar dataKey={dataKey} fill={BAR_COLOR} radius={[4, 4, 0, 0]} maxBarSize={28} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function AddZoneModal({ open, onClose, onSubmit }) {
  const [form, setForm] = useState({ name: "", from: "", to: "" });
  const [search, setSearch] = useState("");
  const [selectedPeople, setSelectedPeople] = useState([]);

  function reset() {
    setForm({ name: "", from: "", to: "" });
    setSearch("");
    setSelectedPeople([]);
  }

  function close() {
    reset();
    onClose();
  }

  function addPerson(name) {
    setSelectedPeople((prev) => (prev.includes(name) ? prev : [...prev, name]));
    setSearch("");
  }

  function removePerson(name) {
    setSelectedPeople((prev) => prev.filter((p) => p !== name));
  }

  const suggestions = peopleDirectory.filter(
    (p) => p.toLowerCase().includes(search.toLowerCase()) && !selectedPeople.includes(p)
  );

  function handleSubmit(e) {
    e.preventDefault();
    if (!form.name.trim()) return;
    onSubmit({ ...form, people: selectedPeople });
    reset();
  }

  return (
    <Modal open={open} onClose={close} title="Add new zone for Intrusion" width="max-w-2xl">
      <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Add site / Zone name
          </p>
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Name</label>
            <input
              required
              value={form.name}
              onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
              placeholder="e.g. Server Room"
              className="input-field"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">
              Restricted time zone (optional)
            </label>
            <div className="flex items-center gap-3">
              <input
                type="time"
                value={form.from}
                onChange={(e) => setForm((f) => ({ ...f, from: e.target.value }))}
                className="input-field"
              />
              <span className="text-slate-400 text-sm">to</span>
              <input
                type="time"
                value={form.to}
                onChange={(e) => setForm((f) => ({ ...f, to: e.target.value }))}
                className="input-field"
              />
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Add people</p>
          <div className="relative">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search people"
              className="input-field pl-9"
            />
            {search && suggestions.length > 0 && (
              <div className="absolute z-10 top-full mt-1 w-full bg-white border border-[#e7e8f0] rounded-xl shadow-lg py-1 max-h-40 overflow-y-auto">
                {suggestions.map((p) => (
                  <button
                    type="button"
                    key={p}
                    onClick={() => addPerson(p)}
                    className="w-full text-left px-3.5 py-1.5 text-sm text-ink-900 hover:bg-[#f5f6fa]"
                  >
                    {p}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="flex flex-wrap gap-2 min-h-9">
            {selectedPeople.length === 0 ? (
              <p className="text-xs text-slate-400">No people added yet.</p>
            ) : (
              selectedPeople.map((p) => (
                <span
                  key={p}
                  className="inline-flex items-center gap-1.5 bg-brand-50 text-brand-700 text-xs font-medium pl-3 pr-1.5 py-1 rounded-full"
                >
                  {p}
                  <button
                    type="button"
                    onClick={() => removePerson(p)}
                    className="hover:bg-brand-100 rounded-full p-0.5"
                  >
                    <X size={12} />
                  </button>
                </span>
              ))
            )}
          </div>
        </div>

        <div className="md:col-span-2 flex items-center gap-3 pt-2 border-t border-[#eceef4]">
          <button type="button" onClick={close} className="btn-secondary flex-1">
            Cancel
          </button>
          <button type="submit" className="btn-primary flex-1">
            Add
          </button>
        </div>
      </form>
    </Modal>
  );
}

export default function Intrusion() {
  const [zones, setZones] = useState([]);
  const [openZone, setOpenZone] = useState(null);
  const [accessByZone, setAccessByZone] = useState({});
  const [addOpen, setAddOpen] = useState(false);

  useEffect(() => {
    api.getIntrusionZones().then(setZones);
  }, []);

  function accessRowsFor(zoneName) {
    return accessByZone[zoneName] || zoneAccessDetail[zoneName] || defaultZoneAccess;
  }

  function viewZone(zone) {
    setAccessByZone((prev) => ({
      ...prev,
      [zone.zone]: prev[zone.zone] || zoneAccessDetail[zone.zone] || defaultZoneAccess,
    }));
    setOpenZone(zone);
  }

  function setRowEnabled(zoneName, index, enabled) {
    setAccessByZone((prev) => {
      const rows = prev[zoneName] || zoneAccessDetail[zoneName] || defaultZoneAccess;
      const next = rows.map((r, i) => (i === index ? { ...r, enabled } : r));
      return { ...prev, [zoneName]: next };
    });
  }

  async function handleAddZone(payload) {
    await api.addZone(payload);
    const newZone = { zone: payload.name, access: payload.people.length, action: "Give Access" };
    setZones((prev) => [...prev, newZone]);
    if (payload.people.length > 0) {
      setAccessByZone((prev) => ({
        ...prev,
        [payload.name]: payload.people.map((name) => ({
          name,
          activeWindow: payload.from && payload.to ? `${payload.from} - ${payload.to}` : "Always",
          enabled: true,
        })),
      }));
    }
    setAddOpen(false);
  }

  const totalUnauthorized = weeklyUnauthorized.reduce((sum, d) => sum + d.count, 0);

  if (openZone) {
    const rows = accessRowsFor(openZone.zone);
    const waiting = waitingRoomStats[openZone.zone] || defaultWaitingRoomStats;
    const detections = recentDetections[openZone.zone] || defaultRecentDetections;

    return (
      <div className="space-y-5">
        <button
          onClick={() => setOpenZone(null)}
          className="flex items-center gap-1 text-sm text-slate-500 hover:text-ink-900"
        >
          <ChevronLeft size={16} /> Intrusion
        </button>
        <PageHeader title={`Intrusion / ${openZone.zone}`} />

        {/* Camera thumbnail placeholder */}
        <div className="card p-4 max-w-xs">
          <div className="relative w-full aspect-video rounded-lg bg-[#0c0c14] border border-[#23243a] flex items-center justify-center overflow-hidden">
            <User size={40} className="text-white/25" strokeWidth={1.5} />
            <span className="absolute top-2 left-2 badge badge-success">Online</span>
          </div>
          <p className="text-sm font-medium text-ink-900 mt-2.5">{openZone.zone}</p>
        </div>

        <DataTable
          columns={[
            { key: "name", label: "Access" },
            { key: "activeWindow", label: "Active window" },
            {
              key: "enabled",
              label: "Enabled",
              render: (r) => (
                <ToggleSwitch
                  checked={r.enabled}
                  onChange={(val) => setRowEnabled(openZone.zone, rows.indexOf(r), val)}
                />
              ),
            },
            {
              key: "action",
              label: "Action",
              render: (r) =>
                r.enabled ? (
                  <button
                    onClick={() => setRowEnabled(openZone.zone, rows.indexOf(r), false)}
                    className="text-danger-500 text-sm font-medium"
                  >
                    Revoke
                  </button>
                ) : (
                  <span className="text-xs text-slate-400">Revoked</span>
                ),
            },
          ]}
          rows={rows}
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="card p-4">
            <p className="text-sm font-semibold text-ink-900 mb-2">Waiting room — by hour</p>
            <MiniBarChart data={waiting.byHour} dataKey="value" labelKey="label" height={160} />
          </div>
          <div className="card p-4">
            <p className="text-sm font-semibold text-ink-900 mb-2">Waiting room — by day</p>
            <MiniBarChart data={waiting.byDay} dataKey="value" labelKey="label" height={160} />
          </div>
        </div>

        <div className="card p-4">
          <p className="text-sm font-semibold text-ink-900 mb-3">Recent detections</p>
          <div className="divide-y divide-[#f1f2f7]">
            {detections.map((d, i) => (
              <div key={i} className="flex items-center justify-between py-2 text-sm">
                <span className="text-slate-400">{d.time}</span>
                <span
                  className={d.name === "Unknown Person" ? "text-danger-500 font-medium" : "text-ink-900 font-medium"}
                >
                  {d.name}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Intrusion"
        action={
          <button onClick={() => setAddOpen(true)} className="btn-primary text-sm">
            + Add new zone
          </button>
        }
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card p-4 md:col-span-2">
          <p className="text-sm font-semibold text-ink-900 mb-1">Unauthorized people</p>
          <p className="text-xs text-slate-400 mb-2">Detections per day, this week</p>
          <MiniBarChart data={weeklyUnauthorized} dataKey="count" labelKey="day" />
        </div>
        <div className="card p-5 flex flex-col justify-center gap-3">
          <div>
            <p className="text-2xl font-semibold text-ink-900 leading-none">{zones.length}</p>
            <p className="text-sm text-slate-500 mt-2">Restricted zones</p>
          </div>
          <div>
            <p className="text-2xl font-semibold text-ink-900 leading-none">{totalUnauthorized}</p>
            <p className="text-sm text-slate-500 mt-2">Unauthorized people this week</p>
            <span className="badge badge-success mt-2">Acknowledged</span>
          </div>
        </div>
      </div>

      <DataTable
        columns={[
          { key: "zone", label: "Restricted Zone" },
          { key: "access", label: "Access" },
          {
            key: "action",
            label: "Action",
            render: (r) => (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  viewZone(r);
                }}
                className="text-brand-600 text-sm font-medium"
              >
                {r.action}
              </button>
            ),
          },
        ]}
        rows={zones}
        onRowClick={viewZone}
      />

      <AddZoneModal open={addOpen} onClose={() => setAddOpen(false)} onSubmit={handleAddZone} />
    </div>
  );
}
