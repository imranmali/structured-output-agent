"""Structured logging for the service: JSON lines with request correlation.

``request_id_var`` holds the current request's identifier; the request
middleware sets it and every log record emitted while handling the request
carries it. Prompts and model outputs are never logged by the service; the
``failure_event`` extra emitted by ``ValidationLogger`` is logged without the
raw model output and offending input values (those go, redacted, to the
optional failure JSONL file instead).
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, TextIO

from .settings import ServerSettings

request_id_var: ContextVar[str | None] = ContextVar("soa_request_id", default=None)

# Extras copied from log records into the JSON document when present.
_EXTRAS = ("failure_event", "run_outcome", "access")

_HANDLER_MARK = "_soa_handler"


def _scrub_failure_event(event: Any) -> Any:
    if not isinstance(event, dict):
        return event
    scrubbed = {key: value for key, value in event.items() if key != "raw"}
    errors = scrubbed.get("errors")
    if isinstance(errors, list):
        scrubbed["errors"] = [
            {k: v for k, v in error.items() if k != "input"} if isinstance(error, dict) else error
            for error in errors
        ]
    return scrubbed


class RequestIdFilter(logging.Filter):
    """Sets ``record.request_id`` from the current request context."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "request_id", None) is None:
            record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Formats each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        document: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
        }
        for key in _EXTRAS:
            value = getattr(record, key, None)
            if value is not None:
                document[key] = _scrub_failure_event(value) if key == "failure_event" else value
        if record.exc_info:
            document["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            document["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(document, default=str, ensure_ascii=False)


TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"


def configure_logging(settings: ServerSettings, *, stream: TextIO | None = None) -> logging.Handler:
    """Install one stream handler on the root logger, per ``settings``.

    Calling it again replaces the handler it installed before (other handlers
    are left alone). Returns the installed handler.
    """
    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_MARK, False):
            root.removeHandler(existing)
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    setattr(handler, _HANDLER_MARK, True)
    handler.addFilter(RequestIdFilter())
    formatter = JsonFormatter() if settings.log_format == "json" else logging.Formatter(TEXT_FORMAT)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(settings.log_level)
    # The service writes its own access log; uvicorn's would duplicate it.
    logging.getLogger("uvicorn.access").disabled = True
    return handler
