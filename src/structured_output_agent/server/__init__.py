"""HTTP service for structured-output runs (install the ``server`` extra).

``create_app`` builds the FastAPI app, ``ServerSettings`` reads its ``SOA_*``
configuration and ``PrometheusHooks`` exports run metrics (usable on any
agent). ``structured-agent serve`` runs it with uvicorn.
"""

from __future__ import annotations

try:
    import fastapi  # noqa: F401
    import prometheus_client  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "structured_output_agent.server requires the 'server' extra: "
        "pip install 'structured-output-agent[server]'"
    ) from exc

from .app import create_app
from .metrics import PrometheusHooks
from .settings import ServerSettings

__all__ = ["PrometheusHooks", "ServerSettings", "create_app"]
