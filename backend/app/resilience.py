"""
resilience.py

Failure handling for long-running analytics components (face recognition,
person detection, footfall Re-ID, staff count, intrusion, expression).

Each of these used to set a `failed = True` flag on its first exception and
never run again until the whole process restarted: one bad frame, a
transient "database is locked", or a model file that appeared a minute later
switched a feature off for good, silently.

FeatureHealth replaces that flag. After a failure the component is paused for
an exponentially growing interval (FEATURE_RETRY_BASE_SECONDS doubling up to
FEATURE_RETRY_MAX_SECONDS) and then tried again; after every
FEATURE_REBUILD_AFTER_FAILURES consecutive failures the caller is told to
rebuild the component (reload models) rather than just retry. The first
success after a failure returns it to RUNNING. There is no tight retry loop:
while paused, allow() returns False and the frame is simply skipped.

States:
  RUNNING     working normally
  DEGRADED    failed recently, retrying on a backoff
  RECOVERING  the backoff has elapsed; the next attempt is in progress
  FAILED      failed repeatedly (>= rebuild threshold); still retried at the
              maximum interval, reported as FAILED so operators notice
  STOPPED     deliberately not running (feature switched off / camera gone)
"""

from __future__ import annotations

import logging
import threading
import time

from . import config

log = logging.getLogger("resilience")

RUNNING = "RUNNING"
DEGRADED = "DEGRADED"
RECOVERING = "RECOVERING"
FAILED = "FAILED"
STOPPED = "STOPPED"


class FeatureHealth:
    def __init__(self, feature: str, camera_id: int | None = None, *, base: float | None = None,
                 maximum: float | None = None, rebuild_after: int | None = None, clock=time.monotonic):
        self.feature = feature
        self.camera_id = camera_id
        self.base = base if base is not None else config.FEATURE_RETRY_BASE_SECONDS
        self.maximum = maximum if maximum is not None else config.FEATURE_RETRY_MAX_SECONDS
        self.rebuild_after = max(1, rebuild_after if rebuild_after is not None else config.FEATURE_REBUILD_AFTER_FAILURES)
        self._clock = clock
        self._lock = threading.Lock()
        self.state = RUNNING
        self.consecutive_failures = 0
        self.total_failures = 0
        self.last_error: str | None = None
        self.last_failure_at: float | None = None   # wall clock, for reporting
        self.last_success_at: float | None = None
        self._retry_at = 0.0                         # monotonic

    def _label(self) -> str:
        return f"{self.feature}" + (f" camera {self.camera_id}" if self.camera_id is not None else "")

    def allow(self) -> bool:
        """May the component run now? False while backing off after a
        failure (the caller should skip this unit of work, not wait)."""
        with self._lock:
            if self.state == STOPPED:
                return False
            if self.consecutive_failures == 0:
                return True
            if self._clock() < self._retry_at:
                return False
            if self.state == DEGRADED:
                self.state = RECOVERING
            return True

    def success(self) -> None:
        with self._lock:
            recovered = self.consecutive_failures > 0
            self.consecutive_failures = 0
            self.state = RUNNING
            self.last_success_at = time.time()
        if recovered:
            log.warning("%s recovered", self._label())

    def failure(self, exc: BaseException | None = None) -> bool:
        """Record a failure. Returns True when the caller should rebuild the
        component (drop and reload its models) before the next attempt."""
        with self._lock:
            self.consecutive_failures += 1
            self.total_failures += 1
            n = self.consecutive_failures
            delay = min(self.maximum, self.base * (2 ** (n - 1)))
            self._retry_at = self._clock() + delay
            self.last_failure_at = time.time()
            self.last_error = f"{type(exc).__name__}: {exc}"[:300] if exc is not None else None
            self.state = FAILED if n >= self.rebuild_after else DEGRADED
            rebuild = n % self.rebuild_after == 0
        # Full traceback on the first failure of a streak; after that one
        # line per retry, so a persistent fault can't flood the log.
        if n == 1 and exc is not None:
            log.error("%s failed; retrying in %.0fs", self._label(), delay, exc_info=exc)
        else:
            log.error("%s failed again (%d in a row, %s); retrying in %.0fs%s", self._label(), n,
                      self.last_error, delay, ", rebuilding" if rebuild else "")
        return rebuild

    def stop(self) -> None:
        with self._lock:
            self.state = STOPPED

    def reset(self) -> None:
        with self._lock:
            self.state = RUNNING
            self.consecutive_failures = 0
            self._retry_at = 0.0

    @property
    def failed(self) -> bool:
        """Back-compat with the old boolean flag: True while in a failure
        streak (the component is not currently producing results)."""
        return self.consecutive_failures > 0

    def snapshot(self) -> dict:
        with self._lock:
            retry_in = max(0.0, self._retry_at - self._clock()) if self.consecutive_failures else 0.0
            return {
                "feature": self.feature,
                "camera_id": self.camera_id,
                "state": self.state,
                "consecutive_failures": self.consecutive_failures,
                "total_failures": self.total_failures,
                "retry_in_seconds": round(retry_in, 1),
                "last_error": self.last_error,
                "last_failure_at": self.last_failure_at,
                "last_success_at": self.last_success_at,
            }


# Registry so /ready can report every component's state in one place.
_registry: dict[tuple[str, int | None], FeatureHealth] = {}
_registry_lock = threading.Lock()


def health_for(feature: str, camera_id: int | None = None) -> FeatureHealth:
    """A new tracker for one component instance, registered for reporting
    (replacing any previous tracker for the same feature/camera, e.g. after
    that component was rebuilt)."""
    h = FeatureHealth(feature, camera_id)
    with _registry_lock:
        _registry[(feature, camera_id)] = h
    return h


def unregister(feature: str, camera_id: int | None = None) -> None:
    with _registry_lock:
        _registry.pop((feature, camera_id), None)


def all_health() -> list[dict]:
    with _registry_lock:
        items = list(_registry.values())
    return [h.snapshot() for h in items]


def backoff_delay(attempt: int, base: float, maximum: float, jitter: float = 0.2, rand=None) -> float:
    """Exponential backoff with +/- jitter, for reconnect loops. attempt >= 1."""
    import random

    delay = min(maximum, base * (2 ** max(0, attempt - 1)))
    r = (rand or random.random)()
    return max(0.0, delay * (1 + jitter * (2 * r - 1)))
