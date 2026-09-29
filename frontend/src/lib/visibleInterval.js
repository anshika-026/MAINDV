// Polling that pauses while the browser tab is hidden. Every page used to
// poll the backend on a fixed timer forever, even in a background tab, which
// multiplied backend load by the number of open tabs. Same call shape as
// setInterval/clearInterval, so call sites change by name only.
//
// When the tab becomes visible again and at least one interval has passed,
// fn runs immediately so the page is never showing stale numbers.

export function setVisibleInterval(fn, ms) {
  let last = Date.now();
  const tick = () => {
    if (typeof document !== "undefined" && document.hidden) return;
    last = Date.now();
    fn();
  };
  const onVisible = () => {
    if (!document.hidden && Date.now() - last >= ms) tick();
  };
  const id = setInterval(tick, ms);
  document.addEventListener("visibilitychange", onVisible);
  return { id, onVisible };
}

export function clearVisibleInterval(handle) {
  if (!handle) return;
  clearInterval(handle.id);
  document.removeEventListener("visibilitychange", handle.onVisible);
}
