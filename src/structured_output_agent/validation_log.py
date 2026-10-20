"""Structured validation failure events and reliability statistics.

Every failure is kept in memory, optionally appended to a JSON lines file and
mirrored to the ``structured_output_agent`` stdlib logger, so reliability can
be measured rather than guessed.

Two options make the logger safe to leave on in production:

* ``max_events`` bounds how many events are kept in memory (oldest dropped
  first); statistics still count every failure.
* ``redact`` is a ``str -> str`` function applied to raw output, error
  messages and offending input values before an event is stored, written or
  logged. :func:`redact_patterns` builds one from regular expressions, for
  example ``ValidationLogger(redact=redact_patterns(EMAIL_PATTERN))``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import ErrorDetail, FailureKind, Usage

LOGGER_NAME = "structured_output_agent"

REDACTION_FAILED = "[REDACTION FAILED]"
"""Text stored in place of content whose redaction function raised or returned a non-string."""

EMAIL_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
"""A simple pattern matching email addresses, for use with :func:`redact_patterns`."""


def redact_patterns(
    *patterns: str | re.Pattern[str], replacement: str = "[REDACTED]"
) -> Callable[[str], str]:
    """Return a redaction function replacing every match of ``patterns``.

    ``patterns`` may be strings or compiled regular expressions; they are
    applied in order and each match is replaced by ``replacement``.
    """
    compiled = [re.compile(pattern) for pattern in patterns]

    def redact(text: str) -> str:
        for pattern in compiled:
            text = pattern.sub(replacement, text)
        return text

    return redact


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=repr))


@dataclass
class FailureEvent:
    """One recorded parse, validation or tool failure."""

    timestamp: str
    kind: FailureKind
    contract: str
    attempt: int
    errors: list[ErrorDetail]
    raw: str
    truncated: bool
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "timestamp": self.timestamp,
            "kind": self.kind.value,
            "contract": self.contract,
            "attempt": self.attempt,
            "errors": _json_safe([error.to_dict() for error in self.errors]),
            "raw": self.raw,
            "truncated": self.truncated,
            "run_id": self.run_id,
        }


@dataclass
class ReliabilityStats:
    """Aggregate counts over recorded runs and failures."""

    runs: int = 0
    successes: int = 0
    failures: int = 0
    attempts: int = 0
    first_attempt_successes: int = 0
    failures_by_kind: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    total_duration_s: float = 0.0

    @property
    def mean_duration_s(self) -> float:
        return self.total_duration_s / self.runs if self.runs else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.runs if self.runs else 0.0

    @property
    def first_attempt_success_rate(self) -> float:
        return self.first_attempt_successes / self.runs if self.runs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "successes": self.successes,
            "failures": self.failures,
            "attempts": self.attempts,
            "first_attempt_successes": self.first_attempt_successes,
            "failures_by_kind": dict(self.failures_by_kind),
            "success_rate": self.success_rate,
            "first_attempt_success_rate": self.first_attempt_success_rate,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_duration_s": self.total_duration_s,
            "mean_duration_s": self.mean_duration_s,
        }


class _Redactor:
    """Applies a user redaction function, never letting it fail recording."""

    def __init__(self, func: Callable[[str], str]) -> None:
        self._func = func
        self.failure: str | None = None

    def text(self, text: str) -> str:
        try:
            result = self._func(text)
        except Exception as exc:
            self.failure = type(exc).__name__
            return REDACTION_FAILED
        if not isinstance(result, str):
            self.failure = f"returned {type(result).__name__}"
            return REDACTION_FAILED
        return result

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {key: self.value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.value(item) for item in value)
        return value


class ValidationLogger:
    """Records failure events and run outcomes.

    Events are kept in :attr:`events`, appended as JSON lines to ``path`` when
    given, and emitted on ``logger`` (the library logger by default): failures
    at warning level, exhausted runs at error level. Raw output longer than
    ``max_raw_chars`` is truncated.

    :attr:`events` is a :class:`collections.deque` holding at most
    ``max_events`` events (``None`` keeps every event); the oldest are dropped
    first, while :attr:`stats` keeps counting every failure.

    ``redact``, when given, is applied before anything is stored, written or
    logged: to the raw output (before truncation, so ``truncated`` reflects
    the redacted text), to each error message and to each error input value,
    recursively through the values of dicts, lists and tuples (dict keys and
    non-string scalars are left alone). If the function raises or returns a
    non-string, the affected text is replaced with ``"[REDACTION FAILED]"``
    so the original content never leaks, and a warning naming only the failure
    (the exception type, never the content) is logged.

    All mutations, including file writes, happen under an internal lock, so
    one logger can be shared between threads.
    """

    def __init__(
        self,
        *,
        path: str | os.PathLike[str] | None = None,
        max_raw_chars: int = 2000,
        logger: logging.Logger | None = None,
        max_events: int | None = 1000,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        if max_raw_chars < 0:
            raise ValueError(f"max_raw_chars must be >= 0, got {max_raw_chars}")
        if max_events is not None and max_events < 1:
            raise ValueError(f"max_events must be >= 1 or None, got {max_events}")
        self.path = Path(path) if path is not None else None
        self.max_raw_chars = max_raw_chars
        self.max_events = max_events
        self.redact = redact
        self.logger = logger if logger is not None else logging.getLogger(LOGGER_NAME)
        self.events: deque[FailureEvent] = deque(maxlen=max_events)
        self._stats = ReliabilityStats()
        self._lock = threading.Lock()

    @staticmethod
    def new_run_id() -> str:
        """Return a fresh identifier for correlating the events of one run."""
        return uuid.uuid4().hex

    @property
    def stats(self) -> ReliabilityStats:
        """A snapshot of the current statistics."""
        with self._lock:
            s = self._stats
            return ReliabilityStats(
                runs=s.runs,
                successes=s.successes,
                failures=s.failures,
                attempts=s.attempts,
                first_attempt_successes=s.first_attempt_successes,
                failures_by_kind=dict(s.failures_by_kind),
                input_tokens=s.input_tokens,
                output_tokens=s.output_tokens,
                total_duration_s=s.total_duration_s,
            )

    def record(
        self,
        kind: FailureKind,
        *,
        contract: str,
        attempt: int,
        errors: Sequence[ErrorDetail],
        raw: str = "",
        run_id: str | None = None,
    ) -> FailureEvent:
        """Record one failure and return the stored event.

        Redaction (when configured) is applied before truncation, storage,
        the file write and the log record.
        """
        stored_errors = list(errors)
        if self.redact is not None:
            redactor = _Redactor(self.redact)
            raw = redactor.text(raw)
            stored_errors = [
                replace(error, msg=redactor.text(error.msg), input=redactor.value(error.input))
                for error in stored_errors
            ]
            if redactor.failure is not None:
                self.logger.warning(
                    "redaction function failed (%s); affected content replaced with %s",
                    redactor.failure,
                    REDACTION_FAILED,
                )
        truncated = len(raw) > self.max_raw_chars
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            kind=kind,
            contract=contract,
            attempt=attempt,
            errors=stored_errors,
            raw=raw[: self.max_raw_chars] if truncated else raw,
            truncated=truncated,
            run_id=run_id,
        )
        payload = event.to_dict()
        with self._lock:
            self.events.append(event)
            counts = self._stats.failures_by_kind
            counts[kind.value] = counts.get(kind.value, 0) + 1
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

        summary = "; ".join(str(error) for error in event.errors[:3])
        if len(event.errors) > 3:
            summary += f"; ... ({len(event.errors) - 3} more)"
        self.logger.warning(
            "%s for %s on attempt %d: %s",
            kind.value,
            contract,
            attempt,
            summary or "no details",
            extra={"failure_event": payload},
        )
        return event

    def record_run(
        self,
        *,
        contract: str,
        success: bool,
        attempts: int,
        run_id: str | None = None,
        usage: Usage | None = None,
        duration_s: float | None = None,
    ) -> None:
        """Record the outcome of a whole run of ``attempts`` generation attempts."""
        with self._lock:
            s = self._stats
            s.runs += 1
            s.attempts += attempts
            if usage is not None:
                s.input_tokens += usage.input_tokens
                s.output_tokens += usage.output_tokens
            if duration_s is not None:
                s.total_duration_s += duration_s
            if success:
                s.successes += 1
                if attempts == 1:
                    s.first_attempt_successes += 1
            else:
                s.failures += 1
        if not success:
            self.logger.error(
                "run failed for %s after %d attempts",
                contract,
                attempts,
                extra={
                    "run_outcome": {
                        "contract": contract,
                        "success": success,
                        "attempts": attempts,
                        "run_id": run_id,
                    }
                },
            )

    def clear(self) -> None:
        """Forget all events and reset statistics."""
        with self._lock:
            self.events.clear()
            self._stats = ReliabilityStats()
