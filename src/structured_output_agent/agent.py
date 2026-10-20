"""Agents: contract -> generate -> (tools) -> parse -> validate -> retry.

``StructuredOutputAgent`` drives runs synchronously and
``AsyncStructuredOutputAgent`` drives them on an event loop. Both delegate
every decision to ``RunEngine``, so they behave identically.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from pydantic import BaseModel

from .engine import ExecuteTools, Finished, RunEngine, TimedToolResult
from .hooks import HookDispatcher, RunHooks
from .llm import AsyncLLMClient, LLMClient
from .retry import RetryPolicy
from .schema import OutputContract
from .tools import ToolRegistry
from .types import RunResult, ToolCall
from .validation_log import ValidationLogger

T = TypeVar("T", bound=BaseModel)


class _AgentBase:
    def __init__(
        self,
        *,
        tools: ToolRegistry | None,
        retry: RetryPolicy | None,
        logger: ValidationLogger | None,
        hooks: Iterable[RunHooks],
        system_prompt: str | None,
        strict: bool,
        max_tool_rounds: int,
        run_timeout: float | None,
    ) -> None:
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be >= 0")
        if run_timeout is not None and run_timeout <= 0:
            raise ValueError("run_timeout must be > 0")
        self.tools = tools if tools is not None else ToolRegistry()
        self.retry = retry if retry is not None else RetryPolicy()
        self.logger = logger if logger is not None else ValidationLogger()
        self.hooks = HookDispatcher(hooks)
        self.system_prompt = system_prompt
        self.strict = strict
        self.max_tool_rounds = max_tool_rounds
        self.run_timeout = run_timeout

    def _engine(
        self,
        prompt: str,
        output_model: type[T],
        strict: bool | None,
        system_prompt: str | None,
    ) -> RunEngine[T]:
        contract = OutputContract(output_model, strict=self.strict if strict is None else strict)
        return RunEngine(
            prompt,
            contract,
            tools=self.tools,
            retry=self.retry,
            logger=self.logger,
            hooks=self.hooks,
            system_prompt=system_prompt or self.system_prompt,
            max_tool_rounds=self.max_tool_rounds,
            timeout=self.run_timeout,
        )


class StructuredOutputAgent(_AgentBase):
    """Runs prompts against a sync LLM and returns validated Pydantic objects.

    Every answer is parsed and validated against the requested model. Invalid
    answers are sent back to the model with itemised errors until the retry
    policy is exhausted. Tool calls are validated in both directions and do not
    consume retry attempts; they are capped by ``max_tool_rounds`` instead.
    ``run_timeout`` is checked before every model call; bound individual calls
    with the client's own timeout.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        tools: ToolRegistry | None = None,
        retry: RetryPolicy | None = None,
        logger: ValidationLogger | None = None,
        hooks: Iterable[RunHooks] = (),
        system_prompt: str | None = None,
        strict: bool = False,
        max_tool_rounds: int = 8,
        run_timeout: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            tools=tools,
            retry=retry,
            logger=logger,
            hooks=hooks,
            system_prompt=system_prompt,
            strict=strict,
            max_tool_rounds=max_tool_rounds,
            run_timeout=run_timeout,
        )
        self.llm = llm
        self._sleep = sleep

    def run(
        self,
        prompt: str,
        output_model: type[T],
        *,
        strict: bool | None = None,
        system_prompt: str | None = None,
    ) -> RunResult[T]:
        """Return ``output_model`` parsed from the model's answer to ``prompt``.

        Raises ``RetryExhaustedError`` when no attempt yields valid output,
        ``ToolRoundLimitError`` when the model keeps calling tools,
        ``RunTimeoutError`` past ``run_timeout`` and the provider's error when
        a model call fails.
        """
        engine = self._engine(prompt, output_model, strict, system_prompt)
        try:
            while True:
                request = engine.request()
                try:
                    response = self.llm.complete(
                        system=request.system, messages=request.messages, tools=request.tools
                    )
                except Exception as exc:
                    engine.provider_failed(exc)
                    raise
                step = engine.handle(response)
                if isinstance(step, Finished):
                    return step.result
                if isinstance(step, ExecuteTools):
                    engine.add_tool_results([self._execute(call) for call in step.calls])
                elif step.delay > 0:
                    self._sleep(step.delay)
        except BaseException as exc:
            engine.abort(exc)
            raise

    def run_many(
        self,
        prompts: Sequence[str],
        output_model: type[T],
        *,
        max_workers: int = 8,
        strict: bool | None = None,
        system_prompt: str | None = None,
    ) -> list[RunResult[T] | Exception]:
        """Run several prompts on a thread pool; results keep input order.

        A failed item is returned as its exception instead of aborting the
        batch. The LLM client, tools and hooks must be thread-safe.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        def one(prompt: str) -> RunResult[T] | Exception:
            try:
                return self.run(prompt, output_model, strict=strict, system_prompt=system_prompt)
            except Exception as exc:
                return exc

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="soa-run") as pool:
            return list(pool.map(one, prompts))

    def _execute(self, call: ToolCall) -> TimedToolResult:
        started = time.monotonic()
        result = self.tools.execute(call)
        return TimedToolResult(result=result, latency_s=time.monotonic() - started)


class AsyncStructuredOutputAgent(_AgentBase):
    """Async ``StructuredOutputAgent`` for high-concurrency services.

    Tool calls requested in one turn run concurrently (coroutine tools are
    awaited, sync tools run in worker threads). ``run_timeout`` bounds the
    whole run, including in-flight model and tool calls.
    """

    def __init__(
        self,
        llm: AsyncLLMClient,
        *,
        tools: ToolRegistry | None = None,
        retry: RetryPolicy | None = None,
        logger: ValidationLogger | None = None,
        hooks: Iterable[RunHooks] = (),
        system_prompt: str | None = None,
        strict: bool = False,
        max_tool_rounds: int = 8,
        run_timeout: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(
            tools=tools,
            retry=retry,
            logger=logger,
            hooks=hooks,
            system_prompt=system_prompt,
            strict=strict,
            max_tool_rounds=max_tool_rounds,
            run_timeout=run_timeout,
        )
        self.llm = llm
        self._sleep = sleep

    async def run(
        self,
        prompt: str,
        output_model: type[T],
        *,
        strict: bool | None = None,
        system_prompt: str | None = None,
    ) -> RunResult[T]:
        """Async ``StructuredOutputAgent.run``; same errors and semantics."""
        engine = self._engine(prompt, output_model, strict, system_prompt)
        try:
            if engine.timeout is None:
                return await self._drive(engine)
            try:
                return await asyncio.wait_for(self._drive(engine), timeout=engine.timeout)
            except (asyncio.TimeoutError, TimeoutError) as exc:
                if engine.finished:
                    raise
                raise engine.timed_out() from exc
        except BaseException as exc:
            engine.abort(exc)
            raise

    async def run_many(
        self,
        prompts: Sequence[str],
        output_model: type[T],
        *,
        concurrency: int = 8,
        strict: bool | None = None,
        system_prompt: str | None = None,
    ) -> list[RunResult[T] | Exception]:
        """Run several prompts with at most ``concurrency`` in flight.

        Results keep input order; a failed item is returned as its exception
        instead of aborting the batch.
        """
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        semaphore = asyncio.Semaphore(concurrency)

        async def one(prompt: str) -> RunResult[T] | Exception:
            async with semaphore:
                try:
                    return await self.run(
                        prompt, output_model, strict=strict, system_prompt=system_prompt
                    )
                except Exception as exc:
                    return exc

        return list(await asyncio.gather(*(one(prompt) for prompt in prompts)))

    async def _drive(self, engine: RunEngine[T]) -> RunResult[T]:
        while True:
            request = engine.request()
            try:
                response = await self.llm.complete(
                    system=request.system, messages=request.messages, tools=request.tools
                )
            except Exception as exc:
                engine.provider_failed(exc)
                raise
            step = engine.handle(response)
            if isinstance(step, Finished):
                return step.result
            if isinstance(step, ExecuteTools):
                results = await asyncio.gather(*(self._execute(call) for call in step.calls))
                engine.add_tool_results(results)
            elif step.delay > 0:
                await self._sleep(step.delay)

    async def _execute(self, call: ToolCall) -> TimedToolResult:
        started = time.monotonic()
        result = await self.tools.execute_async(call)
        return TimedToolResult(result=result, latency_s=time.monotonic() - started)
