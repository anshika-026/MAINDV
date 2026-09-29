"""
ratelimit.py

Login throttling: a sliding-window counter of FAILED attempts, kept per
client IP and per username. Successful logins don't count, so a legitimate
user is only ever slowed down by someone else guessing their password, and
only for the length of the window (never a permanent lockout).

In-process state is correct for this app because it runs as a single
uvicorn worker (face_db.capture_limit_lock and the camera threads already
depend on that — see DEPLOYMENT.md). Multiple workers would each get their
own counters and the effective limit would multiply.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

from . import config

log = logging.getLogger("auth.ratelimit")


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float, clock=time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque:
        q = self._hits.get(key)
        if q is None:
            q = self._hits[key] = deque()
        while q and q[0] <= now - self.window:
            q.popleft()
        return q

    def retry_after(self, key: str) -> float:
        """Seconds until `key` may try again (0 = allowed now)."""
        with self._lock:
            now = self._clock()
            q = self._prune(key, now)
            if len(q) < self.limit:
                return 0.0
            return max(0.0, q[0] + self.window - now)

    def hit(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            self._prune(key, now).append(now)
            # Bounded memory: drop keys whose window has fully expired.
            if len(self._hits) > 10_000:
                for k in [k for k, v in self._hits.items() if not v or v[-1] <= now - self.window]:
                    del self._hits[k]

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


class LoginGuard:
    """Combines the per-IP and per-username limiters for one login endpoint."""

    def __init__(self, name: str, per_ip: int | None = None, per_user: int | None = None, window: float | None = None):
        window = window or config.LOGIN_RATE_WINDOW_SECONDS
        self.name = name
        self.by_ip = SlidingWindowLimiter(per_ip or config.LOGIN_RATE_LIMIT_PER_IP, window)
        self.by_user = SlidingWindowLimiter(per_user or config.LOGIN_RATE_LIMIT_PER_USER, window)

    def check(self, ip: str, username: str) -> None:
        wait = max(self.by_ip.retry_after(ip), self.by_user.retry_after(username.lower()))
        if wait > 0:
            log.warning("%s: throttled login attempt ip=%s user=%s", self.name, ip, username.lower()[:64])
            raise HTTPException(status_code=429, detail="Too many failed login attempts. Try again later.",
                                headers={"Retry-After": str(int(wait) + 1)})

    def failed(self, ip: str, username: str) -> None:
        self.by_ip.hit(ip)
        self.by_user.hit(username.lower())
        log.warning("%s: failed login ip=%s user=%s", self.name, ip, username.lower()[:64])

    def succeeded(self, ip: str, username: str) -> None:
        self.by_user.reset(username.lower())


def client_ip(request: Request) -> str:
    if config.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()[:64]
        real = request.headers.get("x-real-ip")
        if real:
            return real.strip()[:64]
    return request.client.host if request.client else "unknown"


admin_login_guard = LoginGuard("admin-login")
client_login_guard = LoginGuard("client-login")
