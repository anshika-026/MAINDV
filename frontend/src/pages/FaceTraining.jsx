import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../api/client";

// Standalone, keyboard-driven labeling tool for the face-training dataset —
// deliberately outside AppShell (no sidebar/topbar): this is a fast-repeat
// ops task (captured face -> type employee ID -> next), not a page someone
// navigates around in. See backend/FACE_TRAINING.md for the full pipeline
// this feeds (camera capture -> label here -> POST /api/faces/training/train).
export default function FaceTraining() {
  const [capture, setCapture] = useState(null); // {id, camera_id, camera_name, captured_at, detection_confidence, ...} | null
  const [reviewed, setReviewed] = useState(0);
  const [total, setTotal] = useState(0);
  const [employeeId, setEmployeeId] = useState("");
  const [employees, setEmployees] = useState([]); // existing roster, for the datalist — never invented here
  const [imageUrl, setImageUrl] = useState(null); // object URL for the current capture's image, see loadImage()
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const inputRef = useRef(null);

  // Correction list: recently labeled captures, so a mistyped ID can be
  // fixed without hunting through the dataset. See /recent-labels.
  const [recentLabels, setRecentLabels] = useState([]);
  const [fixingId, setFixingId] = useState(null); // capture id currently being corrected
  const [fixValue, setFixValue] = useState("");
  const [recentError, setRecentError] = useState("");

  const loadRecent = useCallback(() => {
    api.getRecentTrainingLabels(6).then(setRecentLabels).catch(() => {});
  }, []);

  const [undoNotice, setUndoNotice] = useState("");

  // Model training history — see /training-history. Purely informational
  // here (viewing it never triggers training); retraining only ever
  // happens via the explicit `python -m app.train_faces` command.
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  useEffect(() => {
    api.getTrainingHistory(10).then(setTrainingHistory).catch(() => {});
  }, []);

  const loadNext = useCallback(async () => {
    try {
      const data = await api.getNextTrainingCapture();
      setCapture(data.capture);
      setReviewed(data.reviewed);
      setTotal(data.total);
      setError("");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadNext();
  }, [loadNext]);

  useEffect(() => {
    loadRecent();
  }, [loadRecent]);

  // Deliberately NOT auto-synced from the external face-enrollment service
  // on every page load. That service's roster turned out to be stale/wrong
  // (e.g. it still maps ID 018 to a different employee than the real
  // current roster) — auto-syncing here would silently overwrite a manual
  // correction the moment this page is reopened. Local roster is the
  // source of truth for labeling; POST /employees/sync stays available for
  // an explicit, deliberate re-pull when that external service is actually
  // up to date again.
  useEffect(() => {
    api.getTrainingEmployees().then(setEmployees).catch(() => {});
  }, []);

  // The image needs its own authenticated fetch (see fetchTrainingImageObjectUrl) —
  // a plain <img src> can't send the Bearer token this endpoint now requires.
  // Revoke the previous object URL whenever the capture changes or this page
  // unmounts, so we don't leak a blob URL per capture over a long session.
  useEffect(() => {
    if (!capture) {
      setImageUrl(null);
      return;
    }
    let cancelled = false;
    let objectUrl = null;
    api
      .fetchTrainingImageObjectUrl(capture.id)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        objectUrl = url;
        setImageUrl(url);
      })
      .catch((e) => setError(e.message));
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [capture]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [capture]);

  // Only poll while caught up — a capture already on screen must never be
  // swapped out from under the person mid-type.
  useEffect(() => {
    if (capture) return;
    const timer = setInterval(loadNext, 30000);
    return () => clearInterval(timer);
  }, [capture, loadNext]);

  async function handleLabel(e) {
    e.preventDefault();
    if (!capture || !employeeId.trim()) return;
    try {
      await api.labelTrainingCapture(capture.id, employeeId.trim());
      setEmployeeId("");
      loadNext();
      loadRecent();
    } catch (e2) {
      setError(e2.message);
    }
  }

  async function handleSkip() {
    if (!capture) return;
    try {
      await api.skipTrainingCapture(capture.id);
      setEmployeeId("");
      loadNext();
    } catch (e2) {
      setError(e2.message);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Escape") {
      e.preventDefault();
      handleSkip();
    }
  }

  // Undo: sends a just-labeled capture back into the unlabeled queue (its
  // own captured_at timestamp puts it back at the front, since /next serves
  // oldest-first) — no employee_id guessed, plain revert.
  async function handleUndo(id) {
    setRecentError("");
    try {
      await api.unlabelTrainingCapture(id);
      setRecentLabels((prev) => prev.filter((r) => r.id !== id));
      setUndoNotice(`Undid label for capture #${id}`);
      setTimeout(() => setUndoNotice(""), 2500);
    } catch (e) {
      setRecentError(e.message);
    }
  }

  // Ctrl+Z undoes the MOST RECENT label, regardless of what's focused —
  // deliberately overrides the browser's native undo (e.g. of in-progress
  // typing in the Employee ID field), per explicit choice over the
  // less-aggressive "only when the field is empty" alternative.
  useEffect(() => {
    function onKeyDown(e) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (recentLabels.length > 0) handleUndo(recentLabels[0].id);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [recentLabels]);

  function startFix(row) {
    setFixingId(row.id);
    setFixValue(row.employee_id || "");
    setRecentError("");
  }

  async function submitFix(id) {
    if (!fixValue.trim()) return;
    try {
      await api.relabelTrainingCapture(id, fixValue.trim());
      setRecentLabels((prev) => prev.map((r) => (r.id === id ? { ...r, employee_id: fixValue.trim() } : r)));
      setFixingId(null);
    } catch (e) {
      setRecentError(e.message);
    }
  }

  const latestRun = trainingHistory[0];

  return (
    <div className="min-h-screen bg-[#0f1016] text-white flex flex-col items-center justify-center px-4 py-10 gap-6">
      <p className="text-xs font-semibold tracking-widest text-slate-400 uppercase">Face dataset labeling</p>

      {latestRun && (
        <div className="w-full max-w-sm rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-xs">
          <div className="flex items-center justify-between">
            <p className="font-semibold text-slate-200">Face Model Training</p>
            <button onClick={() => setShowHistory((s) => !s)} className="text-brand-400 hover:text-brand-300">
              {showHistory ? "Hide history" : "Show history"}
            </button>
          </div>
          <p className="text-slate-400 mt-1">
            Current validation accuracy:{" "}
            <span className="text-white font-semibold">
              {latestRun.validation_accuracy != null ? `${(latestRun.validation_accuracy * 100).toFixed(1)}%` : "n/a"}
            </span>{" "}
            ({latestRun.sample_count} samples, {latestRun.class_count} employees)
          </p>
          <p className="text-slate-500 mt-0.5">
            Held-out validation only — not a measurement of live-camera accuracy. Retrain from the terminal:{" "}
            <code className="text-slate-400">python -m app.train_faces</code>
          </p>
          {showHistory && (
            <ul className="mt-2 space-y-1 border-t border-white/10 pt-2">
              {trainingHistory.map((run, i) => (
                <li key={run.id} className="flex justify-between text-slate-400">
                  <span>
                    Run {trainingHistory.length - i} · {new Date(run.trained_at * 1000).toLocaleDateString()}
                  </span>
                  <span className="text-slate-300">
                    {run.validation_accuracy != null ? `${(run.validation_accuracy * 100).toFixed(1)}%` : "n/a"} ·{" "}
                    {run.sample_count} samples
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Existing roster only (see GET /api/faces/training/employees) — a
          convenience for finding an ID, never a suggestion of who this is.
          Always rendered (not nested under the capture-present branch) so
          it's available to the recent-labels "Fix" input too. */}
      <datalist id="known-employee-ids">
        {employees.map((e) => (
          <option key={e.employee_id} value={e.employee_id}>
            {e.name}
          </option>
        ))}
      </datalist>

      {undoNotice && (
        <p className="text-xs text-brand-400 bg-brand-500/10 rounded-full px-3 py-1">{undoNotice}</p>
      )}

      {loading ? (
        <p className="text-slate-400 text-sm">Loading…</p>
      ) : error && !capture ? (
        <p className="text-danger-500 text-sm">{error}</p>
      ) : !capture ? (
        <div className="text-center space-y-2">
          <p className="text-lg font-semibold">All caught up</p>
          <p className="text-sm text-slate-400">
            {reviewed} / {total} reviewed — checking for new captures every 30s.
          </p>
        </div>
      ) : (
        <>
          <div className="w-full max-w-sm rounded-2xl overflow-hidden bg-black border border-white/10">
            {imageUrl ? (
              <img src={imageUrl} alt="Captured face" className="w-full aspect-square object-cover" />
            ) : (
              <div className="w-full aspect-square flex items-center justify-center text-slate-500 text-sm">
                Loading image…
              </div>
            )}
          </div>

          <div className="text-center text-sm text-slate-400">
            <p>Camera: {capture.camera_name}</p>
            <p>Captured: {new Date(capture.captured_at * 1000).toLocaleString()}</p>
            {typeof capture.detection_confidence === "number" && (
              <p className="text-xs text-slate-500">
                Detection confidence: {(capture.detection_confidence * 100).toFixed(0)}%
                {typeof capture.blur_score === "number" && <> · Sharpness: {Math.round(capture.blur_score)}</>}
              </p>
            )}
          </div>

          <form onSubmit={handleLabel} className="w-full max-w-xs space-y-2">
            <label className="text-xs font-medium text-slate-400 block text-center">Employee ID</label>
            <input
              ref={inputRef}
              value={employeeId}
              onChange={(e) => setEmployeeId(e.target.value)}
              onKeyDown={handleKeyDown}
              autoFocus
              list="known-employee-ids"
              autoComplete="off"
              className="w-full text-center text-lg rounded-xl border border-white/15 bg-white/5 px-4 py-3 outline-none focus:border-brand-500"
              placeholder="e.g. 018"
            />
            {error && <p className="text-danger-500 text-xs text-center">{error}</p>}
            <p className="text-xs text-slate-500 text-center">Enter = save + next · Escape = skip · Ctrl+Z = undo last label</p>
          </form>

          <p className="text-sm text-slate-400">
            {reviewed} / {total} reviewed
          </p>
        </>
      )}

      {recentLabels.length > 0 && (
        <div className="w-full max-w-xs border-t border-white/10 pt-4 mt-2">
          <p className="text-xs font-medium text-slate-400 text-center mb-2">Just labeled — wrong ID? Fix it here</p>
          {recentError && <p className="text-danger-500 text-xs text-center mb-2">{recentError}</p>}
          <ul className="space-y-1.5">
            {recentLabels.map((r) => (
              <li key={r.id} className="flex items-center justify-between gap-2 text-xs bg-white/5 rounded-lg px-3 py-2">
                {fixingId === r.id ? (
                  <>
                    <input
                      autoFocus
                      value={fixValue}
                      onChange={(e) => setFixValue(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && submitFix(r.id)}
                      list="known-employee-ids"
                      className="flex-1 min-w-0 rounded border border-white/15 bg-white/10 px-2 py-1 text-white outline-none focus:border-brand-500"
                    />
                    <button onClick={() => submitFix(r.id)} className="text-brand-400 hover:text-brand-300 shrink-0">
                      Save
                    </button>
                    <button onClick={() => setFixingId(null)} className="text-slate-500 hover:text-slate-300 shrink-0">
                      Cancel
                    </button>
                  </>
                ) : (
                  <>
                    <span className="text-slate-300 truncate">
                      <span className="font-semibold text-white">{r.employee_id}</span> · {r.camera_name}
                    </span>
                    <span className="flex gap-3 shrink-0">
                      <button onClick={() => startFix(r)} className="text-brand-400 hover:text-brand-300">
                        Fix
                      </button>
                      <button onClick={() => handleUndo(r.id)} className="text-slate-500 hover:text-slate-300">
                        Undo
                      </button>
                    </span>
                  </>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
