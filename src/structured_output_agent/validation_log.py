"""Structured validation failure events and reliability statistics.

Every failure is kept in memory, optionally appended to a JSON lines file and
mirrored to the ``structured_output_agent`` stdlib logger, so reliability can
be measured rather than guessed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import ErrorDetail, FailureKind, Usage

LOGGER_NAME = "structured_output_agent"


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


class ValidationLogger:
    """Records failure events and run outcomes.

    Events are kept in :attr:`events`, appended as JSON lines to ``path`` when
    given, and emitted on ``logger`` (the library logger by default): failures
    at warning level, exhausted runs at error level. Raw output longer than
    ``max_raw_chars`` is truncated.
    """

    def __init__(
        self,
        *,
        path: str | os.PathLike[str] | None = None,
        max_raw_chars: int = 2000,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_raw_chars < 0:
            raise ValueError(f"max_raw_chars must be >= 0, got {max_raw_chars}")
        self.path = Path(path) if path is not None else None
        self.max_raw_chars = max_raw_chars
        self.logger = logger if logger is not None else logging.getLogger(LOGGER_NAME)
        self.events: list[FailureEvent] = []
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
        """Record one failure and return the stored event."""
        truncated = len(raw) > self.max_raw_chars
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            kind=kind,
            contract=contract,
            attempt=attempt,
            errors=list(errors),
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
