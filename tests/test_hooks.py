import logging

import pytest
from pydantic import BaseModel

from structured_output_agent import (
    HookDispatcher,
    LLMResponse,
    RunHooks,
    ScriptedLLM,
    StructuredOutputAgent,
    ToolCall,
    ToolRegistry,
    Usage,
)
from structured_output_agent.tools import ToolResult
from structured_output_agent.validation_log import FailureEvent


class Person(BaseModel):
    name: str
    age: int


class Recorder(RunHooks):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def on_run_start(self, **kw: object) -> None:  # type: ignore[override]
        self.calls.append(("start", kw))

    def on_llm_request(self, **kw: object) -> None:  # type: ignore[override]
        self.calls.append(("request", kw))

    def on_llm_response(self, **kw: object) -> None:  # type: ignore[override]
        self.calls.append(("response", kw))

    def on_tool_result(self, **kw: object) -> None:  # type: ignore[override]
        self.calls.append(("tool", kw))

    def on_failure(self, **kw: object) -> None:  # type: ignore[override]
        self.calls.append(("failure", kw))

    def on_run_end(self, **kw: object) -> None:  # type: ignore[override]
        self.calls.append(("end", kw))

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


class Broken(RunHooks):
    def __getattribute__(self, name: str) -> object:
        if name.startswith("on_"):

            def boom(**kwargs: object) -> None:
                raise RuntimeError(f"broken {name}")

            return boom
        return super().__getattribute__(name)


def test_hooks_see_the_whole_lifecycle() -> None:
    hook = Recorder()
    llm = ScriptedLLM(
        [
            LLMResponse(text='{"name": "A"}', usage=Usage(10, 2)),
            LLMResponse(text='{"name": "A", "age": 1}', usage=Usage(12, 3)),
        ]
    )
    agent = StructuredOutputAgent(llm, hooks=[hook], sleep=lambda _: None)

    result = agent.run("?", Person)

    assert hook.names() == ["start", "request", "response", "failure", "request", "response", "end"]
    start = hook.calls[0][1]
    assert start["contract"] == "Person"
    assert start["run_id"] == result.run_id
    response = hook.calls[2][1]
    assert response["usage"] == Usage(10, 2)
    assert isinstance(response["latency_s"], float)
    failure = hook.calls[3][1]
    assert isinstance(failure["event"], FailureEvent)
    end = hook.calls[-1][1]
    assert end["success"] is True
    assert end["attempts"] == 2
    assert end["usage"] == Usage(22, 5)
    assert end["error"] is None


def test_tool_hook_and_failed_run_end() -> None:
    hook = Recorder()
    tools = ToolRegistry()

    @tools.register
    def ping() -> dict[str, bool]:
        """Ping."""
        return {"ok": True}

    llm = ScriptedLLM([LLMResponse(text="", tool_calls=[ToolCall("1", "ping", {})]), "bad"])
    agent = StructuredOutputAgent(llm, tools=tools, hooks=[hook])
    agent.retry = agent.retry.__class__(max_attempts=1)

    with pytest.raises(Exception, match="No valid Person"):
        agent.run("?", Person)

    tool = next(kw for name, kw in hook.calls if name == "tool")
    assert isinstance(tool["result"], ToolResult)
    assert tool["result"].ok is True
    end = hook.calls[-1][1]
    assert end["success"] is False
    assert end["error"] is not None


def test_broken_hook_never_changes_the_outcome(caplog: pytest.LogCaptureFixture) -> None:
    recorder = Recorder()
    agent = StructuredOutputAgent(
        ScriptedLLM(['{"name": "A", "age": 1}']), hooks=[Broken(), recorder]
    )

    with caplog.at_level(logging.ERROR, logger="structured_output_agent"):
        result = agent.run("?", Person)

    assert result.output.name == "A"
    assert recorder.names() == ["start", "request", "response", "end"]
    assert any("hook Broken.on_run_start failed" in r.getMessage() for r in caplog.records)


def test_base_hooks_are_no_ops() -> None:
    hooks = RunHooks()
    dispatcher = HookDispatcher([hooks])
    dispatcher.on_run_start(run_id="r", contract="C", prompt="p")
    dispatcher.on_llm_request(run_id="r", contract="C", attempt=1)
    dispatcher.on_llm_response(run_id="r", contract="C", attempt=1, latency_s=0.1, usage=Usage())
    dispatcher.on_run_end(
        run_id="r",
        contract="C",
        success=True,
        attempts=1,
        usage=Usage(),
        duration_s=0.1,
        error=None,
    )
    assert dispatcher.hooks == [hooks]


def test_usage_helpers() -> None:
    total = Usage(1, 2) + Usage(3, 4)
    assert total == Usage(4, 6)
    assert total.total_tokens == 10
    assert total.to_dict() == {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}
