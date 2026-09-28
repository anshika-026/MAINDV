import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Calendar } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import SidePanel from "../components/SidePanel";
import { useAuth } from "../context/AuthContext";
const resolutionReasons = ["Visitor with permission", "Employee not yet enrolled", "False alarm", "Fixed", "Duplicate", "Other"];
const RANGES = { Today: "today", Yesterday: "yesterday", "This week": "week", "All time": "all" };
const REFRESH_MS = 15000;
import * as api from "../api/client";

const STATUS_OPTIONS = ["Active", "Acknowledged", "Resolved"];
const SEVERITY_OPTIONS = ["Critical", "High", "Medium", "Low"];

// Small dropdown-style filter pill. Kept local to this page since it isn't
// a general-purpose primitive yet — just UI chrome for the filter row.
function FilterPill({ label, icon: Icon, options, value, onChange, menuKey, openKey, setOpenKey }) {
  const isOpen = openKey === menuKey;
  return (
    <div className="relative">
      <button
        onClick={() => setOpenKey(isOpen ? null : menuKey)}
        className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-sm font-medium border transition-colors ${
          value
            ? "bg-brand-50 border-brand-300 text-brand-700"
            : "bg-white border-[#e7e8f0] text-slate-600 hover:bg-[#f5f6fa]"
        }`}
      >
        {Icon ? <Icon size={14} /> : null}
        {value || label}
        <ChevronDown size={14} className="text-slate-400" />
      </button>
      {isOpen && (
        <div className="absolute z-20 top-full mt-1.5 left-0 min-w-[160px] bg-white border border-[#e7e8f0] rounded-xl shadow-lg py-1.5">
          <button
            onClick={() => {
              onChange(null);
              setOpenKey(null);
            }}
            className="w-full text-left px-3.5 py-1.5 text-sm text-slate-500 hover:bg-[#f5f6fa]"
          >
            All
          </button>
          {options.map((opt) => (
            <button
              key={opt}
              onClick={() => {
                onChange(opt);
                setOpenKey(null);
              }}
              className={`w-full text-left px-3.5 py-1.5 text-sm hover:bg-[#f5f6fa] ${
                value === opt ? "text-brand-700 font-medium" : "text-ink-900"
              }`}
            >
              {opt}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Alerts() {
  const { user } = useAuth();
  const [summary, setSummary] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [statusFilter, setStatusFilter] = useState(null);
  const [severityFilter, setSeverityFilter] = useState(null);
  const [eventFilter, setEventFilter] = useState(null);
  const [dateRange, setDateRange] = useState("Today");
  const [openFilter, setOpenFilter] = useState(null);

  const [selected, setSelected] = useState(null);
  const [resolving, setResolving] = useState(false);
  const [reason, setReason] = useState("");
  const [snapshotUrl, setSnapshotUrl] = useState(null);
  const [error, setError] = useState("");

  const filterRef = useRef(null);

  const load = useCallback(() => {
    Promise.all([api.getAlertsSummary(), api.getAlerts(RANGES[dateRange])])
      .then(([s, a]) => {
        setSummary(s);
        setAlerts(a);
        setError("");
      })
      .catch(() => setError("Couldn't load alerts. Check the backend is running."));
  }, [dateRange]);

  useEffect(() => {
    load();
    const t = setInterval(load, REFRESH_MS);
    return () => clearInterval(t);
  }, [load]);

  // Keep the open panel in sync with refreshed data (e.g. someone else resolved it).
  useEffect(() => {
    if (selected) {
      const fresh = alerts.find((a) => a.id === selected.id);
      if (fresh && fresh.status !== selected.status) setSelected(fresh);
    }
  }, [alerts, selected]);

  useEffect(() => {
    let url = null;
    setSnapshotUrl(null);
    if (selected?.hasSnapshot) {
      api
        .fetchAlertSnapshotObjectUrl(selected.id)
        .then((u) => {
          url = u;
          setSnapshotUrl(u);
        })
        .catch(() => setSnapshotUrl(null));
    }
    return () => url && URL.revokeObjectURL(url);
  }, [selected?.id, selected?.hasSnapshot]);

  useEffect(() => {
    function onOutside(e) {
      if (filterRef.current && !filterRef.current.contains(e.target)) setOpenFilter(null);
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, []);

  const eventOptions = useMemo(
    () => Array.from(new Set(alerts.map((a) => a.event))),
    [alerts]
  );

  const filtered = alerts.filter(
    (a) =>
      (!statusFilter || a.status === statusFilter) &&
      (!severityFilter || a.severity === severityFilter) &&
      (!eventFilter || a.event === eventFilter)
  );


  function clearFilters() {
    setStatusFilter(null);
    setSeverityFilter(null);
    setEventFilter(null);
  }

  function openAlert(row) {
    setSelected(row);
    setResolving(false);
    setReason("");
  }

  function closePanel() {
    setSelected(null);
    setResolving(false);
    setReason("");
  }

  async function handleAcknowledge() {
    await api.acknowledgeAlert(selected.id);
    setSelected({ ...selected, status: "Acknowledged", acknowledgedBy: user?.email || "you" });
    load();
  }

  async function handleConfirmResolve() {
    if (!selected || !reason) return;
    await api.resolveAlert(selected.id, reason);
    closePanel();
    load();
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Alerts & Events" />

      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Active Alerts" value={summary.active} subTone="danger" />
          <StatCard label="Acknowledged Alerts" value={summary.acknowledged} subTone="neutral" />
          <StatCard
            label="Resolved today"
            value={summary.resolved_today}
            sub={`${summary.resolved_yesterday} yesterday`}
            subTone="neutral"
          />
          <StatCard label="Raised in the last hour" value={summary.raised_last_hour} sub={`${summary.raised_today} today`} subTone="neutral" />
        </div>
      )}

      {error && <p className="text-sm text-danger-500">{error}</p>}

      <div ref={filterRef} className="flex items-center flex-wrap gap-2">
        <button
          onClick={clearFilters}
          className={`px-3.5 py-1.5 rounded-full text-sm font-medium transition-colors ${
            !statusFilter && !severityFilter && !eventFilter
              ? "bg-brand-500 text-white"
              : "bg-white border border-[#e7e8f0] text-slate-600 hover:bg-[#f5f6fa]"
          }`}
        >
          All
        </button>
        <FilterPill
          label="Status"
          options={STATUS_OPTIONS}
          value={statusFilter}
          onChange={setStatusFilter}
          menuKey="status"
          openKey={openFilter}
          setOpenKey={setOpenFilter}
        />
        <FilterPill
          label="Severity"
          options={SEVERITY_OPTIONS}
          value={severityFilter}
          onChange={setSeverityFilter}
          menuKey="severity"
          openKey={openFilter}
          setOpenKey={setOpenFilter}
        />
        <FilterPill
          label="Events"
          options={eventOptions}
          value={eventFilter}
          onChange={setEventFilter}
          menuKey="events"
          openKey={openFilter}
          setOpenKey={setOpenFilter}
        />
        <FilterPill
          label="Today"
          icon={Calendar}
          options={["Today", "Yesterday", "This week", "All time"]}
          value={dateRange === "Today" ? null : dateRange}
          onChange={(v) => setDateRange(v || "Today")}
          menuKey="date"
          openKey={openFilter}
          setOpenKey={setOpenFilter}
        />
      </div>

      <DataTable
        columns={[
          { key: "date", label: "Date" },
          { key: "event", label: "Event" },
          { key: "camera", label: "Camera" },
          { key: "location", label: "Location" },
          { key: "time", label: "Time" },
          {
            key: "status",
            label: "Status",
            render: (r) => (
              <div className="flex flex-col items-start gap-1">
                <StatusBadge value={r.status} />
                {r.status === "Resolved" && r.resolvedBy && (
                  <span className="text-xs text-slate-400">by {r.resolvedBy}</span>
                )}
              </div>
            ),
          },
        ]}
        rows={filtered}
        onRowClick={openAlert}
      />

      <SidePanel open={!!selected} onClose={closePanel} title="Detection detail" width="max-w-[420px]">
        {selected && (
          <div className="space-y-5">
            {/* Video-frame thumbnail with a simulated detection box */}
            {selected.hasSnapshot && (
              <div className="w-full rounded-xl bg-[#0c0c14] border border-[#23243a] overflow-hidden flex items-center justify-center min-h-[160px]">
                {snapshotUrl ? (
                  <img src={snapshotUrl} alt={`Snapshot for ${selected.event}`} className="max-h-72 object-contain" />
                ) : (
                  <span className="text-xs text-white/40">Loading snapshot…</span>
                )}
              </div>
            )}
            {selected.message && <p className="text-sm text-ink-900">{selected.message}</p>}

            <div className="space-y-2.5 text-sm">
              <Row label="Detection Type" value={selected.event} />
              <Row label="Camera" value={selected.camera} />
              <Row label="Location" value={selected.location} />
              <Row label="Timestamp" value={`${selected.date} · ${selected.time}`} />
              <Row label="Severity" value={selected.severity} />
              {selected.confidence != null && selected.event === "Unknown Person" && (
                <Row label="Closest employee match" value={`${Math.round(selected.confidence * 100)}%`} />
              )}
              {selected.occurrences > 1 && <Row label="Seen again" value={`${selected.occurrences - 1}× · last at ${selected.lastSeen}`} />}
              {selected.acknowledgedBy && <Row label="Acknowledged by" value={selected.acknowledgedBy} />}
            </div>


            <div className="border-t border-[#eceef4] pt-4">
              {selected.status === "Resolved" ? (
                <div className="rounded-xl bg-success-50 text-success-600 text-sm font-medium px-4 py-3 text-center">
                  Resolved by {selected.resolvedBy}
                  {selected.resolutionReason ? ` · ${selected.resolutionReason}` : ""}
                </div>
              ) : !resolving ? (
                <div className="flex items-center gap-3">
                  {selected.status === "Active" && (
                    <button onClick={handleAcknowledge} className="btn-secondary flex-1">
                      Acknowledge
                    </button>
                  )}
                  <button onClick={() => setResolving(true)} className="btn-primary flex-1">
                    Resolve
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  <div>
                    <label className="text-sm font-medium text-ink-900 block mb-1.5">
                      Resolution reason
                    </label>
                    <select
                      value={reason}
                      onChange={(e) => setReason(e.target.value)}
                      className="input-field"
                    >
                      <option value="">Select a reason</option>
                      {resolutionReasons.map((r) => (
                        <option key={r} value={r}>
                          {r}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="flex items-center gap-3">
                    <button onClick={() => setResolving(false)} className="btn-secondary flex-1">
                      Back
                    </button>
                    <button
                      onClick={handleConfirmResolve}
                      disabled={!reason}
                      className="btn-primary flex-1"
                    >
                      Confirm resolve
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </SidePanel>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-slate-400">{label}</span>
      <span className="text-ink-900 font-medium text-right">{value}</span>
    </div>
  );
}
