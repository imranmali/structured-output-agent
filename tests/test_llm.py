from types import SimpleNamespace
from typing import Any

import pytest

from structured_output_agent import LLMError, LLMResponse, Message, ScriptedLLM, ToolCall, ToolSpec
from structured_output_agent.llm import (
    AnthropicClient,
    LLMClient,
    from_anthropic_response,
    to_anthropic_messages,
)


def test_scripted_llm_plays_back_in_order_and_records_calls() -> None:
    llm = ScriptedLLM(["first", LLMResponse(text="second"), lambda system, msgs: f"n={len(msgs)}"])
    msgs = [Message(role="user", content="hi")]
    tools = [ToolSpec(name="t", description="d", input_schema={})]

    assert llm.complete(system="s", messages=msgs, tools=tools).text == "first"
    assert llm.complete(system="s", messages=msgs).text == "second"
    assert llm.complete(system="s", messages=msgs).text == "n=1"
    assert llm.remaining == 0
    assert len(llm.calls) == 3
    assert llm.calls[0].system == "s"
    assert llm.calls[0].tools == tools


def test_scripted_llm_raises_when_exhausted() -> None:
    llm = ScriptedLLM([])
    with pytest.raises(LLMError, match="exhausted"):
        llm.complete(system="", messages=[])


def test_scripted_llm_satisfies_protocol() -> None:
    assert isinstance(ScriptedLLM([]), LLMClient)


def test_to_anthropic_messages_merges_tool_results() -> None:
    messages = [
        Message(role="user", content="weather?"),
        Message(
            role="assistant",
            content="checking",
            tool_calls=[
                ToolCall(id="a", name="w", arguments={"city": "Paris"}),
                ToolCall(id="b", name="w", arguments={"city": "Rome"}),
            ],
        ),
        Message(role="tool", content='{"t": 1}', tool_call_id="a"),
        Message(role="tool", content="boom", tool_call_id="b", is_error=True),
        Message(role="assistant", content="", raw=[{"type": "text", "text": "raw"}]),
        Message(role="user", content="fix it"),
    ]
    out = to_anthropic_messages(messages)
    assert out[0] == {"role": "user", "content": "weather?"}
    assert out[1]["content"][0] == {"type": "text", "text": "checking"}
    assert out[1]["content"][1]["type"] == "tool_use"
    assert out[2]["role"] == "user"
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["a", "b"]
    assert out[2]["content"][1]["is_error"] is True
    assert out[3] == {"role": "assistant", "content": [{"type": "text", "text": "raw"}]}
    assert out[4] == {"role": "user", "content": "fix it"}


def test_from_anthropic_response_extracts_text_and_tool_calls() -> None:
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text='{"a": '),
        SimpleNamespace(type="text", text="1}"),
        SimpleNamespace(type="tool_use", id="t1", name="w", input={"city": "Oslo"}),
        SimpleNamespace(type="tool_use", id="t2", name="w", input='{"city": "Rome"}'),
    ]
    resp = from_anthropic_response(SimpleNamespace(stop_reason="tool_use", content=content))
    assert resp.text == '{"a": 1}'
    assert resp.tool_calls == [
        ToolCall(id="t1", name="w", arguments={"city": "Oslo"}),
        ToolCall(id="t2", name="w", arguments={"city": "Rome"}),
    ]
    assert resp.raw is content


def test_from_anthropic_response_raises_on_refusal() -> None:
    resp = SimpleNamespace(
        stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"), content=[]
    )
    with pytest.raises(LLMError, match="cyber"):
        from_anthropic_response(resp)


class _FakeMessages:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text="{}")]
        )


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()
        self.beta = SimpleNamespace(messages=_FakeMessages())


def test_anthropic_client_uses_fallbacks_by_default() -> None:
    fake = _FakeClient()
    client = AnthropicClient(client=fake, effort="low")
    tools = [ToolSpec(name="w", description="weather", input_schema={"type": "object"})]
    resp = client.complete(system="sys", messages=[Message(role="user", content="q")], tools=tools)

    sent = fake.beta.messages.kwargs
    assert resp.text == "{}"
    assert sent["model"] == "claude-opus-5-5"
    assert sent["system"] == "sys"
    assert sent["betas"] == ["server-side-fallback-2026-07-01"]
    assert sent["extra_body"] == {"fallbacks": "default"}
    assert sent["output_config"] == {"effort": "low"}
    assert sent["tools"][0]["name"] == "w"
    assert fake.messages.kwargs == {}


def test_anthropic_client_without_fallbacks_uses_stable_endpoint() -> None:
    fake = _FakeClient()
    client = AnthropicClient("claude-sonnet-5-5", client=fake, fallbacks=None)
    client.complete(system="sys", messages=[Message(role="user", content="q")])

    sent = fake.messages.kwargs
    assert sent["model"] == "claude-sonnet-5-5"
    assert "tools" not in sent
    assert "betas" not in sent
    assert "output_config" not in sent


async def test_async_scripted_llm_plays_back_and_raises_exception_steps() -> None:
    from structured_output_agent import AsyncLLMClient, AsyncScriptedLLM, TransientLLMError

    llm = AsyncScriptedLLM([TransientLLMError("busy", retry_after=1.5), "ok"], delay=0.001)
    assert isinstance(llm, AsyncLLMClient)
    with pytest.raises(TransientLLMError) as info:
        await llm.complete(system="s", messages=[])
    assert info.value.retryable is True
    assert info.value.retry_after == 1.5
    assert (await llm.complete(system="s", messages=[])).text == "ok"
    assert len(llm.calls) == 2


def test_llm_error_defaults_to_permanent() -> None:
    err = LLMError("bad request", status_code=400)
    assert err.retryable is False
    assert err.status_code == 400
    assert LLMError("x", retryable=True).retryable is True
