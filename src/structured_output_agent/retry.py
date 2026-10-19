"""Retry policy with capped exponential backoff and repair prompts.

A failed attempt is repaired conversationally: the agent sends the model's
previous answer back as an assistant turn followed by the message built here,
which itemises what was wrong and asks for a corrected JSON object.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .types import ErrorDetail, FailureKind

QUOTE_LIMIT = 500


@dataclass(frozen=True)
class RetryPolicy:
    """How many attempts to make and how long to wait between them.

    The delay before attempt ``n`` (``n >= 2``) is
    ``min(initial_delay * multiplier ** (n - 2), max_delay)``; the first
    attempt never waits.
    """

    max_attempts: int = 3
    initial_delay: float = 0.0
    multiplier: float = 2.0
    max_delay: float = 10.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.initial_delay < 0:
            raise ValueError(f"initial_delay must be >= 0, got {self.initial_delay}")
        if self.max_delay < 0:
            raise ValueError(f"max_delay must be >= 0, got {self.max_delay}")
        if self.multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {self.multiplier}")

    def delay_for(self, attempt: int) -> float:
        """Seconds to wait before 1-based attempt number ``attempt``."""
        if attempt < 1:
            raise ValueError(f"attempt must be >= 1, got {attempt}")
        if attempt == 1:
            return 0.0
        return float(min(self.initial_delay * self.multiplier ** (attempt - 2), self.max_delay))

    def wait(self, attempt: int, sleep: Callable[[float], None] = time.sleep) -> float:
        """Sleep for :meth:`delay_for` ``attempt`` (if positive) and return the delay."""
        delay = self.delay_for(attempt)
        if delay > 0:
            sleep(delay)
        return delay


def _quote(text: str, limit: int = QUOTE_LIMIT) -> str:
    if not text.strip():
        return "(empty response)"
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated, {len(text) - limit} more characters]"


def build_repair_message(
    previous_output: str,
    errors: Sequence[ErrorDetail],
    kind: FailureKind,
    *,
    contract_name: str | None = None,
) -> str:
    """Build the user turn asking the model to fix its previous response."""
    target = f"the `{contract_name}` output contract" if contract_name else "the output contract"
    if kind is FailureKind.PARSE_ERROR:
        problem = "could not be parsed as JSON"
    elif kind is FailureKind.VALIDATION_ERROR:
        problem = f"failed validation against {target}"
    else:
        problem = f"was rejected ({kind.value})"

    lines = [
        f"Your previous response {problem}.",
        "",
        "Previous response:",
        _quote(previous_output),
    ]
    lines += ["", "Errors:"]
    lines += [f"- {error.loc}: {error.msg}" for error in errors] or ["- $: unknown error"]
    lines += [
        "",
        f"Reply with ONLY a corrected JSON object that conforms to the schema of {target}. "
        "Do not include any explanation, prose or markdown outside the JSON.",
    ]
    return "\n".join(lines)
