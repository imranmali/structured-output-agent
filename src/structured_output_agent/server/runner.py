"""Run the service with uvicorn: one worker process, scaled by replicas."""

from __future__ import annotations

from .app import create_app
from .logging_config import configure_logging
from .settings import ServerSettings

GRACEFUL_SHUTDOWN_S = 30


def run(settings: ServerSettings | None = None, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Serve the app until SIGTERM/SIGINT, then drain for up to 30 seconds.

    ``settings`` default to :meth:`ServerSettings.from_env`. Proxy headers
    (``X-Forwarded-For``/``-Proto``) are trusted from any address because the
    service is meant to run behind a load balancer or ingress; do not expose
    it directly to the internet.
    """
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "serving requires uvicorn: pip install 'structured-output-agent[server]'"
        ) from exc
    settings = settings if settings is not None else ServerSettings.from_env()
    configure_logging(settings)
    app = create_app(settings)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
        server_header=False,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
    )
