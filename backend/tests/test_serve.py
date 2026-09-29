"""The supported server entry point (app/serve.py) keeps the settings the
app depends on: one worker, no reload, a BOUNDED graceful shutdown."""

from app import serve


def test_server_settings():
    cfg = serve.build_config()
    assert cfg.workers == 1 and cfg.reload is False
    # Without a bound, uvicorn waits forever for open live-video websockets
    # and never runs the lifespan shutdown (cameras/readers left running).
    assert cfg.timeout_graceful_shutdown is not None and 0 < cfg.timeout_graceful_shutdown <= 30
    assert cfg.lifespan == "on"
