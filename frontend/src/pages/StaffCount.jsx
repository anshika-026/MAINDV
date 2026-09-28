import { useCallback, useEffect, useState } from "react";
import { Bug, DoorOpen, RotateCcw } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import Modal from "../components/Modal";
import StaffEntranceEditor from "../components/StaffEntranceEditor";
import StaffDebugView from "../components/StaffDebugView";
import * as api from "../api/client";

// Staff Count (backend/app/staff/): who is inside the office right now, from
// entry/exit line crossings at entrance cameras. The number is the backend's
// occupancy state (one visit per person, shared by every entrance), not how
// many people a camera happens to see.

const REFRESH_MS = 10000;
const EVENT_LABEL = {
  ENTRY: "Entered",
  EXIT: "Left",
  AUTO_EXIT: "Closed at end of day",
  IDENTIFIED: "Identified after entering",
  DUPLICATE_ENTRY: "Duplicate ignored",
  EXIT_UNMATCHED: "Exit not matched",
};

function clock(ts) {
  return ts ? new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "-";
}

export default function StaffCount() {
  const [count, setCount] = useState(null);
  const [people, setPeople] = useState([]);
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState(null);
  const [setupOpen, setSetupOpen] = useState(false);
  const [debugCam, setDebugCam] = useState(null);
  const [confirmReset, setConfirmReset] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(() => {
    Promise.all([api.getStaffPresent(), api.getStaffEvents(100), api.getStaffStatus()])
      .then(([p, e, s]) => {
        setPeople(p);
        setEvents(e);
        setStatus(s);
        setError("");
      })
      .catch(() => setError("Couldn't load Staff Count. Check the backend is running."));
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, REFRESH_MS);
    return () => clearInterval(t);
  }, [load]);

  // Live count over the websocket, falling back to polling.
  useEffect(() => {
    let ws;
    let poll;
    try {
      ws = new WebSocket(api.staffSocketUrl("/ws/staff"));
      ws.onmessage = (e) => setCount(JSON.parse(e.data));
      ws.onerror = () => {
        poll = setInterval(() => api.getStaffCount().then(setCount).catch(() => {}), 5000);
      };
    } catch {
      poll = setInterval(() => api.getStaffCount().then(setCount).catch(() => {}), 5000);
    }
    api.getStaffCount().then(setCount).catch(() => {});
    return () => {
      ws?.close();
      clearInterval(poll);
    };
  }, []);

  async function reset() {
    await api.resetStaffOccupancy();
    setConfirmReset(false);
    load();
  }

  const entrances = status?.cameras || [];
  const anonymous = count?.mode === "anonymous";
  const rows = people.map((p) => ({
    id: p.visit_id || p.employee_id,
    name: p.employee_name || (p.employee_id ? p.employee_id : "Unknown person"),
    status: p.status === "PRESENT" ? "Present" : "Exited",
    entered: clock(p.entry_time),
    lastSeen: clock(p.last_seen),
    camera: p.camera_name || "-",
  }));

  return (
    <div className="space-y-5">
      <PageHeader
        title="Staff Count"
        action={
          <div className="flex flex-wrap gap-2">
            {entrances.length > 0 && (
              <button onClick={() => setDebugCam(entrances[0].camera_id)} className="btn-secondary text-sm flex items-center gap-1.5">
                <Bug size={14} /> Debug view
              </button>
            )}
            <button onClick={() => setSetupOpen(true)} className="btn-primary text-sm flex items-center gap-1.5">
              <DoorOpen size={14} /> Set up entrance
            </button>
          </div>
        }
      />
      {error && <p className="text-sm text-danger-500">{error}</p>}
      {status && !status.switch_on && (
        <p className="card px-4 py-3 text-sm text-warning-600">Staff count is switched off in the sidebar, so nobody is being counted.</p>
      )}
      {status && entrances.length === 0 && (
        <p className="card px-4 py-3 text-sm text-slate-600">No entrance is set up yet. Click Set up entrance and draw the entry line across the doorway.</p>
      )}
      {anonymous && (
        <p className="card px-4 py-3 text-sm text-slate-600">
          Face recognition is off, so people are counted without being named: the staff figure includes anyone inside.
        </p>
      )}

      <div className="grid md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)] gap-4">
        <div className="card p-6 flex flex-col justify-center">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Current staff</p>
          <p className="text-6xl font-semibold text-ink-900 leading-none mt-2 tabular-nums">{count?.current_staff_count ?? "—"}</p>
          <p className="text-xs text-slate-400 mt-3">Updated {count?.timestamp ? count.timestamp.slice(11, 19) : "-"}</p>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
          <StatCard label="Entries today" value={count?.total_entries_today ?? "—"} subTone="neutral" />
          <StatCard label="Exits today" value={count?.total_exits_today ?? "—"} subTone="neutral" />
          <StatCard label="Known staff inside" value={count?.known_staff ?? "—"} subTone="neutral" />
          <StatCard label="Unknown persons" value={count?.unknown_persons ?? "—"} sub="not matched to an employee" subTone="danger" />
          <StatCard label="Total persons" value={count?.total_persons ?? "—"} subTone="neutral" />
          <StatCard
            label="Duplicates prevented"
            value={count?.duplicates_prevented_today ?? "—"}
            sub={count?.unmatched_exits_today ? `${count.unmatched_exits_today} exits unmatched` : "today"}
            subTone="neutral"
          />
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-5">
        <div className="space-y-2">
          <h3 className="font-semibold text-ink-900">People today</h3>
          <DataTable
            columns={[
              { key: "name", label: "Person" },
              { key: "status", label: "Status", render: (r) => <StatusBadge value={r.status} /> },
              { key: "entered", label: "Entered" },
              { key: "lastSeen", label: "Last seen" },
              { key: "camera", label: "Camera" },
            ]}
            rows={rows}
            emptyLabel="Nobody has entered yet today."
          />
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-ink-900">Event log</h3>
            {confirmReset ? (
              <span className="text-sm flex items-center gap-2">
                Mark everyone as left?
                <button onClick={reset} className="text-danger-500 font-medium">
                  Yes, reset
                </button>
                <button onClick={() => setConfirmReset(false)} className="text-slate-500">
                  Cancel
                </button>
              </span>
            ) : (
              <button onClick={() => setConfirmReset(true)} className="text-xs text-slate-500 flex items-center gap-1">
                <RotateCcw size={12} /> Reset occupancy
              </button>
            )}
          </div>
          <DataTable
            columns={[
              { key: "time", label: "Time", render: (e) => clock(e.ts) },
              { key: "event", label: "Event", render: (e) => EVENT_LABEL[e.event_type] || e.event_type },
              { key: "who", label: "Person", render: (e) => e.employee_name || e.employee_id || "Unknown" },
              { key: "cam", label: "Camera", render: (e) => e.camera_name || "-" },
              { key: "track", label: "Track", render: (e) => (e.track_id != null ? `#${e.track_id}` : "-") },
              { key: "how", label: "Matched by", render: (e) => e.identity_source || "-" },
            ]}
            rows={events.map((e) => ({ ...e, id: e.id }))}
            emptyLabel="No events yet."
          />
        </div>
      </div>

      <Modal open={setupOpen} onClose={() => setSetupOpen(false)} title="Set up entrance" width="max-w-3xl">
        {setupOpen && <StaffEntranceEditor onSaved={load} />}
      </Modal>
      <Modal open={debugCam !== null} onClose={() => setDebugCam(null)} title="Staff Count debug view" width="max-w-4xl">
        {debugCam !== null && (
          <div className="space-y-3">
            {entrances.length > 1 && (
              <select value={debugCam} onChange={(e) => setDebugCam(Number(e.target.value))} className="input-field w-auto">
                {entrances.map((c) => (
                  <option key={c.camera_id} value={c.camera_id}>
                    Camera {c.camera_id}
                  </option>
                ))}
              </select>
            )}
            <StaffDebugView cameraId={debugCam} />
          </div>
        )}
      </Modal>
    </div>
  );
}
