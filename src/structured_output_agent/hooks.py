"""Run lifecycle hooks for metrics, tracing and auditing integrations.

Subclass ``RunHooks`` and override the notifications you need; every method is
a no-op by default. Agents call hooks through ``HookDispatcher``, which logs
and swallows hook exceptions so a faulty integration never changes the
outcome of a run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .types import Usage

if TYPE_CHECKING:
    from .tools import ToolResult
    from .validation_log import FailureEvent

_log = logging.getLogger("structured_output_agent")


class RunHooks:
    """Base class for run lifecycle notifications. All methods are optional."""

    def on_run_start(self, *, run_id: str, contract: str, prompt: str) -> None:
        """A run is about to make its first model call."""

    def on_llm_request(self, *, run_id: str, contract: str, attempt: int) -> None:
        """A model call is about to be made."""

    def on_llm_response(
        self, *, run_id: str, contract: str, attempt: int, latency_s: float, usage: Usage
    ) -> None:
        """A model call returned (successfully at the transport level)."""

    def on_tool_result(
        self, *, run_id: str, contract: str, result: ToolResult, latency_s: float
    ) -> None:
        """A tool call finished, successfully or with an error result."""

    def on_failure(self, *, run_id: str, contract: str, event: FailureEvent) -> None:
        """A parse, validation, tool or provider failure was recorded."""

    def on_run_end(
        self,
        *,
        run_id: str,
        contract: str,
        success: bool,
        attempts: int,
        usage: Usage,
        duration_s: float,
        error: BaseException | None,
    ) -> None:
        """A run finished; ``error`` is set when it failed."""


class HookDispatcher(RunHooks):
    """Fans notifications out to several hooks, isolating their failures."""

    def __init__(self, hooks: Iterable[RunHooks] = ()) -> None:
        self.hooks = list(hooks)

    def _dispatch(self, method: str, **kwargs: Any) -> None:
        for hook in self.hooks:
            try:
                getattr(hook, method)(**kwargs)
            except Exception:
                _log.exception("hook %s.%s failed", type(hook).__name__, method)

    def on_run_start(self, *, run_id: str, contract: str, prompt: str) -> None:
        self._dispatch("on_run_start", run_id=run_id, contract=contract, prompt=prompt)

    def on_llm_request(self, *, run_id: str, contract: str, attempt: int) -> None:
        self._dispatch("on_llm_request", run_id=run_id, contract=contract, attempt=attempt)

    def on_llm_response(
        self, *, run_id: str, contract: str, attempt: int, latency_s: float, usage: Usage
    ) -> None:
        self._dispatch(
            "on_llm_response",
            run_id=run_id,
            contract=contract,
            attempt=attempt,
            latency_s=latency_s,
            usage=usage,
        )

    def on_tool_result(
        self, *, run_id: str, contract: str, result: ToolResult, latency_s: float
    ) -> None:
        self._dispatch(
            "on_tool_result", run_id=run_id, contract=contract, result=result, latency_s=latency_s
        )

    def on_failure(self, *, run_id: str, contract: str, event: FailureEvent) -> None:
        self._dispatch("on_failure", run_id=run_id, contract=contract, event=event)

    def on_run_end(
        self,
        *,
        run_id: str,
        contract: str,
        success: bool,
        attempts: int,
        usage: Usage,
        duration_s: float,
        error: BaseException | None,
    ) -> None:
        self._dispatch(
            "on_run_end",
            run_id=run_id,
            contract=contract,
            success=success,
            attempts=attempts,
            usage=usage,
            duration_s=duration_s,
            error=error,
        )
