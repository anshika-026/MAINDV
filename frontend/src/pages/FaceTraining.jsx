import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../api/client";

// Standalone, keyboard-driven labeling tool for the face-training dataset —
// deliberately outside AppShell (no sidebar/topbar): this is a fast-repeat
// ops task (captured face -> type employee ID -> next), not a page someone
// navigates around in. See backend/FACE_TRAINING.md for the full pipeline
// this feeds (camera capture -> label here -> POST /api/faces/training/train).
export default function FaceTraining() {
  const [capture, setCapture] = useState(null); // {id, camera_id, camera_name, captured_at} | null
  const [reviewed, setReviewed] = useState(0);
  const [total, setTotal] = useState(0);
  const [employeeId, setEmployeeId] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const inputRef = useRef(null);

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

  return (
    <div className="min-h-screen bg-[#0f1016] text-white flex flex-col items-center justify-center px-4 py-10 gap-6">
      <p className="text-xs font-semibold tracking-widest text-slate-400 uppercase">Face dataset labeling</p>

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
            <img
              src={api.trainingImageUrl(capture.id)}
              alt="Captured face"
              className="w-full aspect-square object-cover"
            />
          </div>

          <div className="text-center text-sm text-slate-400">
            <p>Camera: {capture.camera_name}</p>
            <p>Captured: {new Date(capture.captured_at * 1000).toLocaleString()}</p>
          </div>

          <form onSubmit={handleLabel} className="w-full max-w-xs space-y-2">
            <label className="text-xs font-medium text-slate-400 block text-center">Employee ID</label>
            <input
              ref={inputRef}
              value={employeeId}
              onChange={(e) => setEmployeeId(e.target.value)}
              onKeyDown={handleKeyDown}
              autoFocus
              className="w-full text-center text-lg rounded-xl border border-white/15 bg-white/5 px-4 py-3 outline-none focus:border-brand-500"
              placeholder="e.g. 018"
            />
            {error && <p className="text-danger-500 text-xs text-center">{error}</p>}
            <p className="text-xs text-slate-500 text-center">Enter = save + next · Escape = skip</p>
          </form>

          <p className="text-sm text-slate-400">
            {reviewed} / {total} reviewed
          </p>
        </>
      )}
    </div>
  );
}
