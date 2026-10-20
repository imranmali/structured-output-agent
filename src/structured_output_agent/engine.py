"""Transport-independent run engine shared by the sync and async agents.

``RunEngine`` owns the state of one run (conversation, attempts, tool rounds,
usage, failures, timing) and decides what happens next, but never performs
I/O: it does not call the model, run tools or sleep. Drivers loop::

    request = engine.request()          # raises RunTimeoutError past the deadline
    response = <call the model>         # on failure: engine.provider_failed(exc)
    step = engine.handle(response)      # ExecuteTools | Retry | Finished
    ...                                 # run tools -> engine.add_tool_results()
                                        # or sleep Retry.delay, or return result

so the sync and async agents cannot drift apart.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

from .errors import OutputError, RetryExhaustedError, RunTimeoutError, ToolRoundLimitError
from .hooks import HookDispatcher
from .parsing import parse_output
from .retry import RetryPolicy, build_repair_message
from .schema import OutputContract
from .tools import ToolRegistry, ToolResult
from .types import (
    Attempt,
    ErrorDetail,
    FailureKind,
    LLMResponse,
    Message,
    RunResult,
    ToolCall,
    ToolSpec,
    Usage,
)
from .validation_log import ValidationLogger

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMRequest:
    """Arguments for the next model call."""

    system: str
    messages: list[Message]
    tools: list[ToolSpec]


@dataclass(frozen=True)
class ExecuteTools:
    """Run these tool calls and pass the results to ``add_tool_results``."""

    calls: list[ToolCall]


@dataclass(frozen=True)
class Retry:
    """Wait ``delay`` seconds, then make the next model call."""

    delay: float


@dataclass(frozen=True)
class Finished(Generic[T]):
    """The run succeeded."""

    result: RunResult[T]


@dataclass(frozen=True)
class TimedToolResult:
    """A tool result plus how long the tool took."""

    result: ToolResult
    latency_s: float


class RunEngine(Generic[T]):
    """State machine for one structured-output run."""

    def __init__(
        self,
        prompt: str,
        contract: OutputContract[T],
        *,
        tools: ToolRegistry,
        retry: RetryPolicy,
        logger: ValidationLogger,
        hooks: HookDispatcher,
        system_prompt: str | None = None,
        max_tool_rounds: int = 8,
        timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.contract = contract
        self.tools = tools
        self.retry = retry
        self.logger = logger
        self.hooks = hooks
        self.max_tool_rounds = max_tool_rounds
        self.timeout = timeout
        self.clock = clock
        self.run_id = logger.new_run_id()
        self.system = contract.system_prompt(system_prompt)
        self.specs = tools.specs()
        self.messages: list[Message] = [Message(role="user", content=prompt)]
        self.failures: list[Attempt] = []
        self.usage = Usage()
        self.attempt = 1
        self.tool_rounds = 0
        self.tool_calls = 0
        self.finished = False
        self.started = clock()
        self._request_started = self.started
        hooks.on_run_start(run_id=self.run_id, contract=contract.name, prompt=prompt)

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started

    def request(self) -> LLMRequest:
        """Return the next model call, failing the run if the deadline passed."""
        if self.timeout is not None and self.elapsed >= self.timeout:
            raise self.timed_out()
        self.hooks.on_llm_request(
            run_id=self.run_id, contract=self.contract.name, attempt=self.attempt
        )
        self._request_started = self.clock()
        return LLMRequest(system=self.system, messages=list(self.messages), tools=self.specs)

    def handle(self, response: LLMResponse) -> ExecuteTools | Retry | Finished[T]:
        """Process a model response and decide the next step."""
        self.usage = self.usage + response.usage
        self.hooks.on_llm_response(
            run_id=self.run_id,
            contract=self.contract.name,
            attempt=self.attempt,
            latency_s=self.clock() - self._request_started,
            usage=response.usage,
        )
        if response.tool_calls:
            self.tool_rounds += 1
            if self.tool_rounds > self.max_tool_rounds:
                raise self._fail(
                    ToolRoundLimitError(f"Exceeded the tool round limit of {self.max_tool_rounds}")
                )
            self.messages.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                    raw=response.raw,
                )
            )
            return ExecuteTools(calls=list(response.tool_calls))

        try:
            output = parse_output(response.text, self.contract)
        except OutputError as exc:
            self.failures.append(
                Attempt(number=self.attempt, raw=response.text, kind=exc.kind, errors=exc.errors)
            )
            self._record(exc.kind, self.contract.name, exc.errors, response.text)
            if self.attempt >= self.retry.max_attempts:
                raise self._fail(
                    RetryExhaustedError(
                        f"No valid {self.contract.name} after {self.attempt} attempt(s): {exc}",
                        attempts=self.failures,
                    )
                ) from exc
            self.messages.append(Message(role="assistant", content=response.text, raw=response.raw))
            self.messages.append(
                Message(
                    role="user",
                    content=build_repair_message(
                        response.text, exc.errors, exc.kind, contract_name=self.contract.name
                    ),
                )
            )
            self.attempt += 1
            return Retry(delay=self.retry.delay_for(self.attempt))

        self.messages.append(Message(role="assistant", content=response.text, raw=response.raw))
        self._finalize(success=True, error=None)
        return Finished(
            RunResult(
                output=output,
                attempts=self.attempt,
                failures=self.failures,
                tool_calls=self.tool_calls,
                messages=self.messages,
                run_id=self.run_id,
                usage=self.usage,
                duration_s=self.elapsed,
            )
        )

    def add_tool_results(self, results: Sequence[TimedToolResult]) -> None:
        """Append tool results (in call order) to the conversation."""
        for timed in results:
            result = timed.result
            self.tool_calls += 1
            if result.error is not None:
                self._record(result.error.kind, result.name, result.error.errors, result.content)
            self.hooks.on_tool_result(
                run_id=self.run_id,
                contract=self.contract.name,
                result=result,
                latency_s=timed.latency_s,
            )
            self.messages.append(
                Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=result.call_id,
                    is_error=not result.ok,
                )
            )

    def provider_failed(self, exc: BaseException) -> None:
        """Record a failed model call and end the run."""
        errors = [ErrorDetail(loc="$", msg=str(exc) or type(exc).__name__, type=type(exc).__name__)]
        self._record(FailureKind.PROVIDER_ERROR, self.contract.name, errors, "")
        self._finalize(success=False, error=exc)

    def timed_out(self) -> RunTimeoutError:
        """End the run as timed out and return the error to raise."""
        error = RunTimeoutError(
            f"Run for {self.contract.name} exceeded its {self.timeout}s timeout "
            f"after {self.attempt} attempt(s)"
        )
        self._finalize(success=False, error=error)
        return error

    def abort(self, exc: BaseException) -> None:
        """End the run because of an unexpected error or cancellation."""
        self._finalize(success=False, error=exc)

    def _record(
        self, kind: FailureKind, contract: str, errors: Sequence[ErrorDetail], raw: str
    ) -> None:
        event = self.logger.record(
            kind,
            contract=contract,
            attempt=self.attempt,
            errors=errors,
            raw=raw,
            run_id=self.run_id,
        )
        self.hooks.on_failure(run_id=self.run_id, contract=self.contract.name, event=event)

    def _fail(self, error: Exception) -> Exception:
        self._finalize(success=False, error=error)
        return error

    def _finalize(self, *, success: bool, error: BaseException | None) -> None:
        if self.finished:
            return
        self.finished = True
        duration = self.elapsed
        self.logger.record_run(
            contract=self.contract.name,
            success=success,
            attempts=self.attempt,
            run_id=self.run_id,
            usage=self.usage,
            duration_s=duration,
        )
        self.hooks.on_run_end(
            run_id=self.run_id,
            contract=self.contract.name,
            success=success,
            attempts=self.attempt,
            usage=self.usage,
            duration_s=duration,
            error=error,
        )
