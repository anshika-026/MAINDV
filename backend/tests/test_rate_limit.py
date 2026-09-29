"""Login throttling (app/ratelimit.py)."""

from app import ratelimit
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_sliding_window_allows_up_to_the_limit_then_blocks_until_expiry():
    clock = FakeClock()
    lim = ratelimit.SlidingWindowLimiter(3, 60, clock=clock)
    for _ in range(3):
        assert lim.retry_after("k") == 0
        lim.hit("k")
    assert lim.retry_after("k") > 0
    clock.t += 61
    assert lim.retry_after("k") == 0


def test_keys_are_independent():
    lim = ratelimit.SlidingWindowLimiter(1, 60, clock=FakeClock())
    lim.hit("a")
    assert lim.retry_after("a") > 0
    assert lim.retry_after("b") == 0


def test_admin_login_is_throttled_per_username(api, admin_token):
    for _ in range(5):
        assert api.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "nope-nope-1"}).status_code == 401
    r = api.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0
    # Another account isn't affected by attacks on this one.
    assert api.post("/api/auth/login", json={"email": "other@example.com", "password": "x-x-x-x-1"}).status_code == 401


def test_successful_login_clears_the_username_counter(api, admin_token):
    for _ in range(4):
        api.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "nope-nope-1"})
    assert api.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}).status_code == 200
    for _ in range(4):
        assert api.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "nope-nope-1"}).status_code == 401


def test_per_ip_limit_stops_spraying_across_usernames(api, admin_token, monkeypatch):
    monkeypatch.setattr(ratelimit, "admin_login_guard", ratelimit.LoginGuard("admin-login", per_ip=3, per_user=100, window=300))
    for i in range(3):
        api.post("/api/auth/login", json={"email": f"u{i}@example.com", "password": "guess-guess-1"})
    assert api.post("/api/auth/login", json={"email": "u9@example.com", "password": "guess-guess-1"}).status_code == 429


def test_client_login_is_throttled(api):
    for _ in range(5):
        assert api.post("/api/licenses/client-login", json={"username": "acme", "password": "bad-pass"}).status_code == 401
    assert api.post("/api/licenses/client-login", json={"username": "acme", "password": "bad-pass"}).status_code == 429


def test_forwarded_for_is_ignored_unless_trusted(monkeypatch):
    from starlette.requests import Request

    from app import config

    scope = {"type": "http", "headers": [(b"x-forwarded-for", b"9.9.9.9")], "client": ("1.2.3.4", 1)}
    monkeypatch.setattr(config, "TRUST_PROXY_HEADERS", False)
    assert ratelimit.client_ip(Request(scope)) == "1.2.3.4"
    monkeypatch.setattr(config, "TRUST_PROXY_HEADERS", True)
    assert ratelimit.client_ip(Request(scope)) == "9.9.9.9"
