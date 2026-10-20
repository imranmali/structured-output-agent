"""Service configuration read from ``SOA_*`` environment variables."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from ..llm import DEFAULT_ANTHROPIC_MODEL

LogFormat = Literal["json", "text"]

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _split(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"expected one of {sorted(_TRUE | _FALSE)}, got {value!r}")


def _parse_int(value: str) -> int:
    return int(value.strip())


def _parse_float(value: str) -> float:
    number = float(value.strip())
    if not math.isfinite(number):
        raise ValueError(f"expected a finite number, got {value!r}")
    return number


def _optional_str(value: str) -> str | None:
    return value.strip() or None


def _str(value: str) -> str:
    return value.strip()


# Environment variable -> (field name, parser). Every ServerSettings field has
# exactly one variable; ``tests/test_server_settings.py`` checks the mapping
# stays complete.
ENV_VARS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "SOA_MODELS": ("models", _split),
    "SOA_TOOLS": ("tools", _optional_str),
    "SOA_LLM_MODEL": ("llm_model", _str),
    "SOA_LLM_TIMEOUT": ("llm_timeout", _parse_float),
    "SOA_PROVIDER_RETRIES": ("provider_retries", _parse_int),
    "SOA_MAX_ATTEMPTS": ("max_attempts", _parse_int),
    "SOA_RUN_TIMEOUT": ("run_timeout", _parse_float),
    "SOA_MAX_IN_FLIGHT": ("max_in_flight", _parse_int),
    "SOA_MAX_PROMPT_CHARS": ("max_prompt_chars", _parse_int),
    "SOA_MAX_BATCH": ("max_batch", _parse_int),
    "SOA_BATCH_CONCURRENCY": ("batch_concurrency", _parse_int),
    "SOA_STRICT": ("strict", _parse_bool),
    "SOA_SYSTEM_PROMPT": ("system_prompt", _optional_str),
    "SOA_API_KEYS": ("api_keys", _split),
    "SOA_LOG_FORMAT": ("log_format", lambda value: value.strip().lower()),
    "SOA_LOG_LEVEL": ("log_level", lambda value: value.strip().upper()),
    "SOA_FAILURE_LOG": ("failure_log", _optional_str),
    "SOA_REDACT_EMAILS": ("redact_emails", _parse_bool),
    "SOA_MAX_EVENTS": ("max_events", _parse_int),
}

_FIELD_TO_ENV = {name: env for env, (name, _) in ENV_VARS.items()}


@dataclass(frozen=True)
class ServerSettings:
    """Configuration of the HTTP service.

    Build it from the environment with :meth:`from_env` or construct it
    directly; values are validated either way and an invalid value raises
    ``ValueError`` naming its environment variable. API keys never appear in
    ``repr``.
    """

    models: tuple[str, ...] = ()
    """Output contracts to serve, as ``module:ClassName`` specs."""
    tools: str | None = None
    """Optional ``module:attribute`` spec of a ``ToolRegistry``."""
    llm_model: str = DEFAULT_ANTHROPIC_MODEL
    llm_timeout: float = 120.0
    """Seconds allowed for one provider call."""
    provider_retries: int = 3
    """Retries of transient provider failures per model call."""
    max_attempts: int = 3
    """Generation attempts per run (repairs of invalid output)."""
    run_timeout: float = 300.0
    """Deadline for one run in seconds; ``0`` disables it."""
    max_in_flight: int = 32
    """Runs in flight per replica before requests are shed with 503."""
    max_prompt_chars: int = 100_000
    max_batch: int = 50
    batch_concurrency: int = 8
    strict: bool = False
    system_prompt: str | None = None
    api_keys: tuple[str, ...] = field(default=(), repr=False)
    """Accepted API keys; empty disables authentication."""
    log_format: LogFormat = "json"
    log_level: str = "INFO"
    failure_log: str | None = None
    """Optional JSON lines file receiving every validation failure event."""
    redact_emails: bool = True
    max_events: int = 1000
    """Failure events kept in memory by the validation logger."""

    def __post_init__(self) -> None:
        self._check(self.llm_model != "", "llm_model", "must not be empty")
        self._check(self.llm_timeout > 0, "llm_timeout", "must be > 0")
        self._check(self.provider_retries >= 0, "provider_retries", "must be >= 0")
        self._check(self.max_attempts >= 1, "max_attempts", "must be >= 1")
        self._check(self.run_timeout >= 0, "run_timeout", "must be >= 0 (0 disables it)")
        self._check(self.max_in_flight >= 1, "max_in_flight", "must be >= 1")
        self._check(self.max_prompt_chars >= 1, "max_prompt_chars", "must be >= 1")
        self._check(self.max_batch >= 1, "max_batch", "must be >= 1")
        self._check(self.batch_concurrency >= 1, "batch_concurrency", "must be >= 1")
        self._check(self.max_events >= 1, "max_events", "must be >= 1")
        self._check(self.log_format in ("json", "text"), "log_format", "must be 'json' or 'text'")
        self._check(
            isinstance(logging.getLevelName(self.log_level), int),
            "log_level",
            "must be a logging level name such as INFO or DEBUG",
        )
        for spec in (*self.models, *((self.tools,) if self.tools else ())):
            module, sep, attr = spec.partition(":")
            name = "models" if spec in self.models else "tools"
            self._check(bool(module and sep and attr), name, f"expects module:attr, got {spec!r}")

    def _check(self, ok: bool, name: str, message: str) -> None:
        if not ok:
            value = "<redacted>" if name == "api_keys" else repr(getattr(self, name))
            raise ValueError(f"{_FIELD_TO_ENV[name]} {message}, got {value}")

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)

    @property
    def run_deadline(self) -> float | None:
        """``run_timeout`` as the agent expects it (``None`` when disabled)."""
        return self.run_timeout or None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServerSettings:
        """Read settings from ``env`` (``os.environ`` by default).

        Unset or empty variables keep their defaults.
        """
        source = os.environ if env is None else env
        values: dict[str, Any] = {}
        for var, (name, parse) in ENV_VARS.items():
            raw = source.get(var)
            if raw is None or not raw.strip():
                continue
            try:
                values[name] = parse(raw)
            except ValueError as exc:
                shown = "<redacted>" if var == "SOA_API_KEYS" else repr(raw)
                raise ValueError(f"{var} is invalid ({exc}): {shown}") from None
        return cls(**values)


def setting_names() -> list[str]:
    """Names of every ``ServerSettings`` field, in declaration order."""
    return [f.name for f in fields(ServerSettings)]
