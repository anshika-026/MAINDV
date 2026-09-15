import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Calendar, Route } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import SidePanel from "../components/SidePanel";
import { useAuth } from "../context/AuthContext";
import { alertDetails, defaultAlertDetail, resolutionReasons } from "../data/alertsDetail";
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
  const [journeyOpen, setJourneyOpen] = useState(false);

  const filterRef = useRef(null);

  useEffect(() => {
    api.getAlertsSummary().then(setSummary);
    api.getAlerts().then(setAlerts);
  }, []);

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

  const activeCount = alerts.filter((a) => a.status === "Active").length;
  const acknowledgedCount = alerts.filter((a) => a.status === "Acknowledged").length;
  const resolvedTodayCount = alerts.filter((a) => a.status === "Resolved").length;

  function clearFilters() {
    setStatusFilter(null);
    setSeverityFilter(null);
    setEventFilter(null);
  }

  function openAlert(row) {
    setSelected(row);
    setResolving(false);
    setReason("");
    setJourneyOpen(false);
  }

  function closePanel() {
    setSelected(null);
    setResolving(false);
    setReason("");
    setJourneyOpen(false);
  }

  async function handleConfirmResolve() {
    if (!selected || !reason) return;
    await api.resolveAlert(selected.id, reason);
    const resolvedBy = user?.name || "You";
    setAlerts((prev) =>
      prev.map((a) =>
        a.id === selected.id ? { ...a, status: "Resolved", resolvedBy, resolutionReason: reason } : a
      )
    );
    closePanel();
  }

  const detail = selected ? alertDetails[selected.id] || defaultAlertDetail : defaultAlertDetail;

  return (
    <div className="space-y-5">
      <PageHeader title="Alerts & Events" />

      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Active Alerts" value={activeCount} subTone="danger" />
          <StatCard label="Acknowledged Alerts" value={acknowledgedCount} subTone="neutral" />
          <StatCard
            label="Resolved today"
            value={resolvedTodayCount}
            sub="+2 vs yesterday"
            subTone="success"
          />
          <StatCard
            label="Live alerts"
            value={summary.liveAlerts}
            sub="-2 vs yesterday"
            subTone="danger"
          />
        </div>
      )}

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
            <div className="relative w-full aspect-video rounded-xl bg-[#0c0c14] border border-[#23243a] overflow-hidden">
              <div
                className="absolute border-2 rounded-sm"
                style={{
                  top: "28%",
                  left: "38%",
                  width: "26%",
                  height: "48%",
                  borderColor: "#ec2f9c",
                  boxShadow: "0 0 0 1px rgba(236,47,156,0.35)",
                }}
              />
              <span className="absolute bottom-2 right-2 text-[11px] text-white/50 font-mono">
                {selected.camera}
              </span>
            </div>

            <div className="space-y-2.5 text-sm">
              <Row label="Detection Type" value={selected.event} />
              <Row label="Camera" value={selected.camera} />
              <Row label="Location" value={selected.location} />
              <Row label="Timestamp" value={`${selected.date} · ${selected.time}`} />
              <Row label="Confidence" value={`${detail.confidence}%`} />
              <div className="flex items-center justify-between py-1">
                <span className="text-slate-400">Person known</span>
                <StatusBadge value={detail.personKnown} />
              </div>
            </div>

            <button
              onClick={() => setJourneyOpen((v) => !v)}
              className="flex items-center gap-1.5 text-sm font-medium text-brand-600 hover:text-brand-700"
            >
              <Route size={15} />
              {journeyOpen ? "Hide journey" : "Track journey"}
            </button>

            {journeyOpen && (
              <div className="rounded-xl bg-[#f8f8fc] border border-[#eceef4] p-3 space-y-2">
                {detail.journey.length === 0 ? (
                  <p className="text-xs text-slate-400">No prior sightings recorded.</p>
                ) : (
                  detail.journey.map((j, i) => (
                    <div key={i} className="flex items-center justify-between text-xs">
                      <span className="text-slate-500">{j.time}</span>
                      <span className="text-ink-900 font-medium">{j.location}</span>
                      <span className="text-slate-400 font-mono">{j.camera}</span>
                    </div>
                  ))
                )}
              </div>
            )}

            <div className="border-t border-[#eceef4] pt-4">
              {selected.status === "Resolved" ? (
                <div className="rounded-xl bg-success-50 text-success-600 text-sm font-medium px-4 py-3 text-center">
                  Resolved by {selected.resolvedBy}
                  {selected.resolutionReason ? ` · ${selected.resolutionReason}` : ""}
                </div>
              ) : !resolving ? (
                <button onClick={() => setResolving(true)} className="btn-primary w-full">
                  Resolve
                </button>
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
