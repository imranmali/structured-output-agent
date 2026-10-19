import json
import logging
from pathlib import Path

import pytest
from pydantic import BaseModel

from structured_output_agent import (
    FailureKind,
    LLMResponse,
    RetryExhaustedError,
    ScriptedLLM,
    ToolCall,
    ToolRoundLimitError,
)
from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.retry import RetryPolicy
from structured_output_agent.tools import ToolRegistry
from structured_output_agent.validation_log import ValidationLogger


class Person(BaseModel):
    name: str
    age: int


class Weather(BaseModel):
    temp_c: float


class Answer(BaseModel):
    summary: str


def weather_tools(response: object = None) -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register(response_model=Weather)
    def get_weather(city: str) -> object:
        """Current weather for a city."""
        return response if response is not None else {"temp_c": 21.0}

    return registry


def make_agent(script: list[object], **kwargs: object) -> tuple[StructuredOutputAgent, ScriptedLLM]:
    llm = ScriptedLLM(script)  # type: ignore[arg-type]
    agent = StructuredOutputAgent(llm, sleep=lambda _: None, **kwargs)  # type: ignore[arg-type]
    return agent, llm


def test_first_attempt_success_injects_schema() -> None:
    agent, llm = make_agent(['{"name": "Ada", "age": 36}'], system_prompt="You extract people.")

    result = agent.run("Ada is 36", Person)

    assert result.output == Person(name="Ada", age=36)
    assert result.attempts == 1
    assert result.failures == []
    system = llm.calls[0].system
    assert system.startswith("You extract people.")
    assert '"age"' in system
    assert "JSON" in system


def test_repair_after_invalid_output_sends_errors_back() -> None:
    agent, llm = make_agent(['{"name": "Ada"}', '{"name": "Ada", "age": 36}'])

    result = agent.run("Ada is 36", Person)

    assert result.output.age == 36
    assert result.attempts == 2
    assert len(result.failures) == 1
    assert result.failures[0].kind is FailureKind.VALIDATION_ERROR
    second = llm.calls[1].messages
    assert second[-2].role == "assistant"
    assert second[-2].content == '{"name": "Ada"}'
    assert second[-1].role == "user"
    assert "age" in second[-1].content


def test_parse_error_is_retried() -> None:
    agent, _ = make_agent(["I cannot answer that.", '{"name": "Ada", "age": 36}'])

    result = agent.run("Ada", Person)

    assert result.failures[0].kind is FailureKind.PARSE_ERROR
    assert result.attempts == 2


def test_all_attempts_fail_raises_with_history(caplog: pytest.LogCaptureFixture) -> None:
    logger = ValidationLogger()
    agent, llm = make_agent(
        ["nope", '{"name": 1}', '{"age": "x"}'], retry=RetryPolicy(max_attempts=3), logger=logger
    )

    with caplog.at_level(logging.WARNING), pytest.raises(RetryExhaustedError) as info:
        agent.run("?", Person)

    assert len(llm.calls) == 3
    assert [a.number for a in info.value.attempts] == [1, 2, 3]
    assert info.value.attempts[0].raw == "nope"
    assert logger.stats.failures == 1
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_single_attempt_does_not_retry() -> None:
    agent, llm = make_agent(["bad", "unused"], retry=RetryPolicy(max_attempts=1))

    with pytest.raises(RetryExhaustedError):
        agent.run("?", Person)

    assert len(llm.calls) == 1


def test_backoff_delays_are_applied() -> None:
    delays: list[float] = []
    llm = ScriptedLLM(["x", "y", "z", '{"name": "A", "age": 1}'])
    agent = StructuredOutputAgent(
        llm,
        retry=RetryPolicy(max_attempts=4, initial_delay=0.5, multiplier=2, max_delay=1.5),
        sleep=delays.append,
    )

    agent.run("?", Person)

    assert delays == [0.5, 1.0, 1.5]


def test_strict_mode_rejects_extra_fields() -> None:
    payload = '{"name": "a", "age": 1, "extra": true}'
    lenient, _ = make_agent([payload])
    strict, _ = make_agent([payload], strict=True, retry=RetryPolicy(max_attempts=1))

    assert lenient.run("?", Person).output.name == "a"
    with pytest.raises(RetryExhaustedError) as info:
        strict.run("?", Person)
    assert any("extra" in e.loc for e in info.value.attempts[0].errors)


def test_strict_override_per_run() -> None:
    agent, _ = make_agent(['{"name": "a", "age": 1, "x": 0}'], retry=RetryPolicy(max_attempts=1))
    with pytest.raises(RetryExhaustedError):
        agent.run("?", Person, strict=True)


def test_tool_then_answer() -> None:
    script = [
        LLMResponse(
            text="Let me check.",
            tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "Paris"})],
        ),
        '{"summary": "21C in Paris"}',
    ]
    agent, llm = make_agent(script, tools=weather_tools())

    result = agent.run("Weather in Paris?", Answer)

    assert result.output.summary == "21C in Paris"
    assert result.tool_calls == 1
    assert result.attempts == 1
    assert llm.calls[0].tools[0].name == "get_weather"
    tool_msg = llm.calls[1].messages[-1]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "c1"
    assert json.loads(tool_msg.content) == {"temp_c": 21.0}
    assert tool_msg.is_error is False


def test_invalid_tool_arguments_are_returned_as_errors_and_logged() -> None:
    logger = ValidationLogger()
    script = [
        LLMResponse(
            text="", tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"town": "P"})]
        ),
        LLMResponse(text="", tool_calls=[ToolCall(id="c2", name="nope", arguments={})]),
        '{"summary": "gave up on tools"}',
    ]
    agent, llm = make_agent(script, tools=weather_tools(), logger=logger)

    result = agent.run("?", Answer)

    assert result.tool_calls == 2
    first_tool_msg = llm.calls[1].messages[-1]
    assert first_tool_msg.is_error is True
    assert "city" in first_tool_msg.content
    kinds = [e.kind for e in logger.events]
    assert kinds == [FailureKind.TOOL_ARGUMENT_ERROR, FailureKind.UNKNOWN_TOOL]
    assert logger.events[0].contract == "get_weather"


def test_non_conforming_tool_response_is_not_passed_through() -> None:
    logger = ValidationLogger()
    script = [
        LLMResponse(
            text="", tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "P"})]
        ),
        '{"summary": "unknown"}',
    ]
    agent, llm = make_agent(script, tools=weather_tools({"temp": "hot"}), logger=logger)

    agent.run("?", Answer)

    tool_msg = llm.calls[1].messages[-1]
    assert tool_msg.is_error is True
    assert "hot" not in json.loads(tool_msg.content).get("temp", "")
    assert logger.events[0].kind is FailureKind.TOOL_RESPONSE_ERROR


def test_tool_round_limit() -> None:
    call = LLMResponse(
        text="", tool_calls=[ToolCall(id="c", name="get_weather", arguments={"city": "P"})]
    )
    agent, llm = make_agent([call, call, call], tools=weather_tools(), max_tool_rounds=2)

    with pytest.raises(ToolRoundLimitError, match="2"):
        agent.run("?", Answer)
    assert len(llm.calls) == 3
    assert agent.logger.stats.failures == 1


def test_negative_tool_round_limit_rejected() -> None:
    with pytest.raises(ValueError, match="max_tool_rounds"):
        StructuredOutputAgent(ScriptedLLM([]), max_tool_rounds=-1)


def test_reliability_stats_and_jsonl_log(tmp_path: Path) -> None:
    log_file = tmp_path / "failures.jsonl"
    logger = ValidationLogger(path=log_file)
    agent, _ = make_agent(
        ['{"name": "A", "age": 1}', '{"name": "B"}', '{"name": "B", "age": 2}'], logger=logger
    )

    agent.run("a", Person)
    agent.run("b", Person)

    stats = logger.stats
    assert stats.runs == 2
    assert stats.successes == 2
    assert stats.attempts == 3
    assert stats.first_attempt_success_rate == 0.5
    assert stats.success_rate == 1.0
    lines = log_file.read_text().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["kind"] == "validation_error"
    assert event["contract"] == "Person"
    assert event["attempt"] == 1
