import asyncio
import time

import pytest
from pydantic import BaseModel

from structured_output_agent import (
    AsyncScriptedLLM,
    AsyncStructuredOutputAgent,
    FailureKind,
    LLMError,
    LLMResponse,
    RetryExhaustedError,
    RetryPolicy,
    RunResult,
    RunTimeoutError,
    ScriptedLLM,
    StructuredOutputAgent,
    ToolCall,
    ToolRegistry,
    ToolRoundLimitError,
    Usage,
    ValidationLogger,
)


class Person(BaseModel):
    name: str
    age: int


class Answer(BaseModel):
    summary: str


async def no_sleep(_: float) -> None:
    return None


def call(id: str, name: str, **arguments: object) -> ToolCall:
    return ToolCall(id=id, name=name, arguments=dict(arguments))


async def test_async_repair_matches_sync_semantics() -> None:
    script = ['{"name": "Ada"}', '{"name": "Ada", "age": 36}']
    async_agent = AsyncStructuredOutputAgent(AsyncScriptedLLM(script), sleep=no_sleep)
    sync_agent = StructuredOutputAgent(ScriptedLLM(script), sleep=lambda _: None)

    a = await async_agent.run("Ada is 36", Person)
    s = sync_agent.run("Ada is 36", Person)

    assert a.output == s.output == Person(name="Ada", age=36)
    assert a.attempts == s.attempts == 2
    assert [(f.number, f.kind, [str(e) for e in f.errors]) for f in a.failures] == [
        (f.number, f.kind, [str(e) for e in f.errors]) for f in s.failures
    ]


async def test_async_backoff_uses_injected_sleep() -> None:
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    agent = AsyncStructuredOutputAgent(
        AsyncScriptedLLM(["x", '{"name": "A", "age": 1}']),
        retry=RetryPolicy(initial_delay=0.25),
        sleep=record,
    )
    await agent.run("?", Person)
    assert delays == [0.25]


async def test_async_retry_exhausted_records_failed_run() -> None:
    logger = ValidationLogger()
    agent = AsyncStructuredOutputAgent(
        AsyncScriptedLLM(["bad"]), retry=RetryPolicy(max_attempts=1), logger=logger
    )
    with pytest.raises(RetryExhaustedError):
        await agent.run("?", Person)
    assert logger.stats.failures == 1


async def test_parallel_tools_run_concurrently_and_keep_order() -> None:
    tools = ToolRegistry()

    @tools.register
    async def slow_async(label: str) -> dict[str, str]:
        """Slow coroutine tool."""
        await asyncio.sleep(0.2)
        return {"label": label}

    @tools.register
    def slow_sync(label: str) -> dict[str, str]:
        """Slow blocking tool, run in a thread."""
        time.sleep(0.2)
        return {"label": label}

    llm = AsyncScriptedLLM(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    call("a", "slow_async", label="first"),
                    call("b", "slow_sync", label="second"),
                ],
            ),
            '{"summary": "done"}',
        ]
    )
    agent = AsyncStructuredOutputAgent(llm, tools=tools)

    started = time.monotonic()
    result = await agent.run("?", Answer)
    elapsed = time.monotonic() - started

    assert result.tool_calls == 2
    assert elapsed < 0.35
    tool_messages = [m for m in llm.calls[1].messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["a", "b"]
    assert '"first"' in tool_messages[0].content


async def test_tool_timeout_returns_error_and_run_continues() -> None:
    tools = ToolRegistry()

    @tools.register(timeout=0.05)
    async def sluggish() -> dict[str, int]:
        """Too slow."""
        await asyncio.sleep(1)
        return {"x": 1}

    logger = ValidationLogger()
    llm = AsyncScriptedLLM(
        [LLMResponse(text="", tool_calls=[call("t", "sluggish")]), '{"summary": "no data"}']
    )
    agent = AsyncStructuredOutputAgent(llm, tools=tools, logger=logger)

    result = await agent.run("?", Answer)

    assert result.output.summary == "no data"
    tool_msg = llm.calls[1].messages[-1]
    assert tool_msg.is_error is True
    assert "timed out" in tool_msg.content
    assert logger.events[0].kind is FailureKind.TOOL_EXECUTION_ERROR


async def test_async_tool_round_limit() -> None:
    tools = ToolRegistry()

    @tools.register
    def ping() -> dict[str, bool]:
        """Ping."""
        return {"ok": True}

    turn = LLMResponse(text="", tool_calls=[call("p", "ping")])
    agent = AsyncStructuredOutputAgent(
        AsyncScriptedLLM([turn, turn]), tools=tools, max_tool_rounds=1
    )
    with pytest.raises(ToolRoundLimitError):
        await agent.run("?", Answer)
    assert agent.logger.stats.failures == 1


async def test_batch_bounded_concurrency_order_and_errors() -> None:
    in_flight = 0
    peak = 0

    class TrackingLLM(AsyncScriptedLLM):
        async def complete(self, **kwargs: object) -> LLMResponse:  # type: ignore[override]
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            prompt = kwargs["messages"][0].content  # type: ignore[index]
            if prompt == "bad":
                return LLMResponse(text="not json")
            return LLMResponse(text=f'{{"name": "{prompt}", "age": 1}}')

    agent = AsyncStructuredOutputAgent(TrackingLLM([]), retry=RetryPolicy(max_attempts=1))

    results = await agent.run_many(["a", "bad", "c"], Person, concurrency=2)

    assert peak <= 2
    assert isinstance(results[0], RunResult)
    assert results[0].output.name == "a"
    assert isinstance(results[1], RetryExhaustedError)
    assert isinstance(results[2], RunResult)
    assert results[2].output.name == "c"


async def test_batch_scales_with_concurrency() -> None:
    llm = AsyncScriptedLLM(['{"name": "A", "age": 1}'] * 20, delay=0.05)
    agent = AsyncStructuredOutputAgent(llm)

    started = time.monotonic()
    results = await agent.run_many([str(i) for i in range(20)], Person, concurrency=10)
    elapsed = time.monotonic() - started

    assert all(isinstance(r, RunResult) for r in results)
    assert elapsed < 0.5  # 20 x 50ms sequentially would take 1s
    assert agent.logger.stats.runs == 20


async def test_batch_rejects_bad_concurrency() -> None:
    agent = AsyncStructuredOutputAgent(AsyncScriptedLLM([]))
    with pytest.raises(ValueError, match="concurrency"):
        await agent.run_many(["x"], Person, concurrency=0)


async def test_run_timeout_fails_slow_model() -> None:
    logger = ValidationLogger()
    agent = AsyncStructuredOutputAgent(
        AsyncScriptedLLM(['{"name": "A", "age": 1}'], delay=1.0), logger=logger, run_timeout=0.05
    )
    with pytest.raises(RunTimeoutError, match="timeout"):
        await agent.run("?", Person)
    assert logger.stats.failures == 1
    assert logger.stats.runs == 1


async def test_provider_timeout_error_is_not_reported_as_run_timeout() -> None:
    agent = AsyncStructuredOutputAgent(AsyncScriptedLLM([TimeoutError("socket")]), run_timeout=5)
    with pytest.raises(TimeoutError, match="socket"):
        await agent.run("?", Person)
    assert agent.logger.events[0].kind is FailureKind.PROVIDER_ERROR


async def test_provider_failure_recorded() -> None:
    logger = ValidationLogger()
    agent = AsyncStructuredOutputAgent(AsyncScriptedLLM([LLMError("invalid key")]), logger=logger)
    with pytest.raises(LLMError, match="invalid key"):
        await agent.run("?", Person)
    assert logger.events[0].kind is FailureKind.PROVIDER_ERROR
    assert logger.events[0].contract == "Person"
    assert logger.stats.failures == 1


async def test_cancelled_run_is_recorded_as_failed() -> None:
    agent = AsyncStructuredOutputAgent(AsyncScriptedLLM(["{}"], delay=1.0))
    task = asyncio.create_task(agent.run("?", Person))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.logger.stats.failures == 1


async def test_usage_and_run_metadata() -> None:
    llm = AsyncScriptedLLM(
        [
            LLMResponse(text='{"name": "A"}', usage=Usage(100, 20)),
            LLMResponse(text='{"name": "A", "age": 1}', usage=Usage(120, 25)),
        ]
    )
    agent = AsyncStructuredOutputAgent(llm, sleep=no_sleep)

    result = await agent.run("?", Person)

    assert result.usage == Usage(220, 45)
    assert result.duration_s > 0
    assert result.run_id
    assert agent.logger.events[0].run_id == result.run_id
    assert agent.logger.stats.input_tokens == 220


def test_invalid_run_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="run_timeout"):
        AsyncStructuredOutputAgent(AsyncScriptedLLM([]), run_timeout=0)


def test_sync_run_timeout_checked_before_model_calls() -> None:
    clock = iter([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    agent = StructuredOutputAgent(ScriptedLLM(["bad", "unused"]), run_timeout=1.0)
    engine = agent._engine("?", Person, None, None)
    engine.clock = lambda: next(clock)
    engine.started = 0.0
    engine.request()
    engine.handle(LLMResponse(text="bad"))
    with pytest.raises(RunTimeoutError):
        engine.request()
    assert engine.finished


def test_sync_run_many_keeps_order_and_captures_errors() -> None:
    def answer(system: str, messages: object) -> str:
        prompt = messages[0].content  # type: ignore[index]
        return "nope" if prompt == "bad" else f'{{"name": "{prompt}", "age": 2}}'

    agent = StructuredOutputAgent(ScriptedLLM([answer] * 3), retry=RetryPolicy(max_attempts=1))
    results = agent.run_many(["x", "bad", "z"], Person, max_workers=2)

    assert isinstance(results[0], RunResult)
    assert isinstance(results[1], RetryExhaustedError)
    assert isinstance(results[2], RunResult)
    with pytest.raises(ValueError, match="max_workers"):
        agent.run_many(["x"], Person, max_workers=0)


def test_sync_provider_failure_and_interrupt_are_recorded() -> None:
    agent = StructuredOutputAgent(ScriptedLLM([LLMError("down")]))
    with pytest.raises(LLMError):
        agent.run("?", Person)
    assert agent.logger.events[0].kind is FailureKind.PROVIDER_ERROR

    def interrupt(system: str, messages: object) -> str:
        raise KeyboardInterrupt

    agent = StructuredOutputAgent(ScriptedLLM([LLMResponse(text="x"), interrupt]))
    agent.retry = RetryPolicy(max_attempts=3)
    with pytest.raises(KeyboardInterrupt):
        agent.run("?", Person)
    assert agent.logger.stats.failures == 1
