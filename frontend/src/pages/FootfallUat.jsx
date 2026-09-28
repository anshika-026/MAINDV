import { useCallback, useEffect, useRef, useState } from "react";
import { DoorOpen, RotateCcw, UserRound } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import ZoneEditor from "../components/ZoneEditor";
import * as api from "../api/client";

// UAT panel for unique footfall: watch each unique person appear as they
// walk through the gates, and restart the count from zero between test
// runs. Refreshes every few seconds so testers see a new person without
// reloading. Snapshots are the body crop the Re-ID engine enrolled the
// person from (it matches on appearance, not on the face).

const REFRESH_MS = 5000;

function timeLabel(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// Snapshot URLs are fetched once per id with the auth header and kept for
// the life of the page, so the 5s refresh doesn't re-download every image.
const snapshotCache = new Map();

function Snapshot({ id, label }) {
  const [url, setUrl] = useState(() => snapshotCache.get(id) || null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (snapshotCache.has(id)) return;
    let cancelled = false;
    api
      .fetchFootfallSnapshotObjectUrl(id)
      .then((u) => {
        snapshotCache.set(id, u);
        if (!cancelled) setUrl(u);
      })
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (url) return <img src={url} alt={`Snapshot of ${label}`} className="w-full h-full object-cover" />;
  return (
    <div className="w-full h-full flex items-center justify-center text-slate-300">
      <UserRound size={28} aria-label={failed ? "Snapshot unavailable" : "Loading snapshot"} />
    </div>
  );
}

export default function FootfallUat() {
  const [summary, setSummary] = useState(null);
  const [people, setPeople] = useState([]);
  const [error, setError] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [lastReset, setLastReset] = useState(null);
  const knownIds = useRef(new Set());
  const [freshIds, setFreshIds] = useState(new Set());
  const [zoneGateId, setZoneGateId] = useState(null);

  const load = useCallback(async () => {
    try {
      const [s, p] = await Promise.all([api.getFootfallSummary(), api.getFootfallPeople()]);
      const arrived = p.filter((x) => knownIds.current.size && !knownIds.current.has(x.id)).map((x) => x.id);
      knownIds.current = new Set(p.map((x) => x.id));
      if (arrived.length) setFreshIds(new Set(arrived));
      setSummary(s);
      setPeople(p);
      setError("");
    } catch {
      setError("Couldn't reach the backend. Check it's running on port 8821.");
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, REFRESH_MS);
    return () => clearInterval(t);
  }, [load]);

  async function doReset() {
    setResetting(true);
    try {
      const r = await api.resetFootfall();
      for (const u of snapshotCache.values()) URL.revokeObjectURL(u);
      snapshotCache.clear();
      knownIds.current = new Set();
      setFreshIds(new Set());
      setLastReset(r);
      setConfirming(false);
      await load();
    } catch {
      setError("The reset didn't go through. Try again, or run scripts/reset_reid_identities.py.");
    } finally {
      setResetting(false);
    }
  }

  const gates = summary?.gates ?? [];
  const zoneGate = gates.find((g) => g.camera_id === zoneGateId) || gates[0];

  return (
    <div className="space-y-5">
      <PageHeader title="Footfall UAT" />

      {summary && !summary.model_ready && (
        <p className="card px-4 py-3 text-sm text-danger-500">
          The Re-ID model isn't installed, so counts will run high. Run{" "}
          <code>python -m scripts.fetch_reid_model</code> in <code>backend/</code> and restart the backend.
        </p>
      )}
      {error && <p className="card px-4 py-3 text-sm text-danger-500">{error}</p>}

      <div className="card p-5 flex flex-wrap items-center gap-4 justify-between">
        <div>
          <h3 className="font-semibold text-ink-900">Restart the count</h3>
          <p className="text-sm text-slate-500 mt-0.5">
            Deletes every unique person and snapshot so counting starts again at PERSON_001. Cameras, faces and
            attendance aren't touched.
          </p>
          {lastReset && (
            <p className="text-xs text-success-600 mt-2">
              Reset at {timeLabel(lastReset.reset_at)}: {lastReset.people_removed} people cleared.
            </p>
          )}
        </div>
        {confirming ? (
          <div className="flex items-center gap-2">
            <span className="text-sm text-ink-900">Clear {people.length} people?</span>
            <button
              onClick={doReset}
              disabled={resetting}
              className="px-3.5 py-2 rounded-lg text-sm font-medium bg-danger-500 text-white disabled:opacity-60"
            >
              {resetting ? "Resetting…" : "Yes, reset to 0"}
            </button>
            <button
              onClick={() => setConfirming(false)}
              disabled={resetting}
              className="px-3.5 py-2 rounded-lg text-sm font-medium border border-border-200 text-slate-600"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button onClick={() => setConfirming(true)} className="btn-primary flex items-center gap-2">
            <RotateCcw size={15} /> Restart count
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Unique people" value={summary ? summary.unique_today : "—"} sub="each counted once" subTone="neutral" />
        {gates.map((g) => (
          <StatCard
            key={g.camera_id}
            label={g.name}
            value={g.unique_today}
            sub={g.counting ? "counting" : "not counting"}
            subTone={g.counting ? "success" : "danger"}
          />
        ))}
        {summary && gates.length === 0 && (
          <StatCard label="Gates" value="0" sub="set a camera's purpose to Entry/Exit" subTone="danger" />
        )}
      </div>

      {gates.length > 0 && (
        <div className="card p-5 space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h3 className="font-semibold text-ink-900">Counting zone</h3>
              <p className="text-sm text-slate-500 mt-0.5">
                Draw a box over each gate's doorway. People outside it, such as anyone sitting in reception, aren't
                counted. Restart the count after changing a zone.
              </p>
            </div>
            <div className="flex flex-wrap gap-1">
              {gates.map((g) => (
                <button
                  key={g.camera_id}
                  onClick={() => setZoneGateId(g.camera_id)}
                  className={`px-3 py-1.5 rounded-full text-sm font-medium ${
                    zoneGate?.camera_id === g.camera_id
                      ? "bg-brand-500 text-white"
                      : "bg-white border border-border-200 text-slate-600"
                  }`}
                >
                  {g.name}
                  {g.zone ? " ✓" : ""}
                </button>
              ))}
            </div>
          </div>
          {zoneGate && <ZoneEditor key={zoneGate.camera_id} gate={zoneGate} onSaved={load} />}
        </div>
      )}

      <div className="flex items-baseline justify-between">
        <h3 className="font-semibold text-ink-900">Unique people ({people.length})</h3>
        <span className="text-xs text-slate-400">Updates every {REFRESH_MS / 1000}s · newest first</span>
      </div>

      {people.length === 0 ? (
        <div className="card p-10 text-center text-sm text-slate-500">
          Nobody counted yet. Walk through a gate with your whole body in view for 4–5 seconds.
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          {people.map((p) => (
            <div
              key={p.id}
              className={`card overflow-hidden transition-shadow ${freshIds.has(p.id) ? "ring-2 ring-brand-500" : ""}`}
            >
              <div className="aspect-[3/4] bg-[#f1f2f7] flex gap-px">
                {p.snapshot_ids.length ? (
                  <Snapshot id={p.snapshot_ids[0]} label={p.label} />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-slate-300">
                    <UserRound size={28} />
                  </div>
                )}
              </div>
              <div className="p-3 space-y-1">
                <p className="font-semibold text-ink-900 text-sm">{p.label}</p>
                <p className="text-xs text-slate-500">
                  First {timeLabel(p.first_seen)} · last {timeLabel(p.last_seen)}
                </p>
                <p className="text-xs text-slate-500 flex items-center gap-1">
                  <DoorOpen size={12} /> {p.gates.join(", ") || "—"}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
