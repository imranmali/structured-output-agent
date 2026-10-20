from types import SimpleNamespace
from typing import Any

import pytest

from structured_output_agent import (
    LLMError,
    LLMResponse,
    Message,
    ScriptedLLM,
    ToolCall,
    ToolSpec,
    TransientLLMError,
    Usage,
)
from structured_output_agent.llm import (
    AnthropicClient,
    AsyncAnthropicClient,
    AsyncLLMClient,
    LLMClient,
    from_anthropic_response,
    map_anthropic_error,
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


# --- usage ------------------------------------------------------------------


def _text_response(**extra: Any) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason="end_turn", content=[SimpleNamespace(type="text", text="{}")], **extra
    )


def test_from_anthropic_response_extracts_usage() -> None:
    usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=5)
    resp = from_anthropic_response(_text_response(usage=usage))
    assert resp.usage == Usage(input_tokens=100, output_tokens=20)


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"usage": None},
        {"usage": SimpleNamespace()},
        {"usage": SimpleNamespace(input_tokens=None, output_tokens=True)},
    ],
)
def test_from_anthropic_response_tolerates_missing_usage(extra: dict[str, Any]) -> None:
    assert from_anthropic_response(_text_response(**extra)).usage == Usage()


# --- error mapping ----------------------------------------------------------


class AnthropicError(Exception):
    """Mimics the SDK's base exception (matched by class name)."""


class APIError(AnthropicError):
    pass


class APIStatusError(APIError):
    def __init__(self, message: str, *, status_code: Any, headers: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response: Any = SimpleNamespace(headers={} if headers is None else headers)


class RateLimitError(APIStatusError):
    pass


class APIConnectionError(APIError):
    pass


class APITimeoutError(APIConnectionError):
    pass


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 529])
def test_map_transient_statuses(status: int) -> None:
    exc = APIStatusError("boom", status_code=status)
    mapped = map_anthropic_error(exc)
    assert type(mapped) is TransientLLMError
    assert mapped.retryable is True
    assert mapped.status_code == status
    assert mapped.retry_after is None
    assert str(status) in str(mapped)
    assert "APIStatusError: boom" in str(mapped)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_map_permanent_statuses(status: int) -> None:
    mapped = map_anthropic_error(APIStatusError("nope", status_code=status))
    assert type(mapped) is LLMError
    assert mapped.retryable is False
    assert mapped.status_code == status


def test_map_api_error_without_status_is_permanent() -> None:
    mapped = map_anthropic_error(APIError("weird"))
    assert type(mapped) is LLMError
    assert mapped.status_code is None
    mapped = map_anthropic_error(APIStatusError("weird", status_code="500"))
    assert type(mapped) is LLMError
    assert mapped.status_code is None


@pytest.mark.parametrize("cls", [APIConnectionError, APITimeoutError])
def test_map_connection_and_timeout_errors(cls: type[Exception]) -> None:
    mapped = map_anthropic_error(cls("down"))
    assert type(mapped) is TransientLLMError
    assert mapped.status_code is None
    assert "connection" in str(mapped)


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "3"}, 3.0),
        ({"retry-after": "1.5"}, 1.5),
        ({"retry-after-ms": "2500"}, 2.5),
        ({"retry-after-ms": "bad", "retry-after": "4"}, 4.0),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({"retry-after": "-1"}, None),
        ({"retry-after": "inf"}, None),
        ({}, None),
    ],
)
def test_map_parses_retry_after(headers: dict[str, str], expected: float | None) -> None:
    mapped = map_anthropic_error(RateLimitError("slow down", status_code=429, headers=headers))
    assert isinstance(mapped, TransientLLMError)
    assert mapped.retry_after == expected


def test_map_tolerates_odd_responses() -> None:
    exc = APIStatusError("x", status_code=503)
    exc.response = SimpleNamespace(headers=object())  # no .get
    assert isinstance(map_anthropic_error(exc), TransientLLMError)
    exc.response = None
    mapped = map_anthropic_error(exc)
    assert isinstance(mapped, TransientLLMError)
    assert mapped.retry_after is None


def test_map_keeps_retry_after_on_permanent_errors() -> None:
    exc = APIStatusError("x", status_code=400, headers={"retry-after": "2"})
    mapped = map_anthropic_error(exc)
    assert type(mapped) is LLMError
    assert mapped.retry_after == 2.0


def test_map_returns_non_sdk_exceptions_unchanged() -> None:
    class HTTPStatusError(Exception):
        status_code = 429

    for exc in (ValueError("x"), HTTPStatusError("y"), LLMError("z")):
        assert map_anthropic_error(exc) is exc


def test_map_real_sdk_exceptions() -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    conn = map_anthropic_error(anthropic.APIConnectionError(request=request))
    assert type(conn) is TransientLLMError
    timeout = map_anthropic_error(anthropic.APITimeoutError(request=request))
    assert type(timeout) is TransientLLMError

    def status_error(cls: Any, status: int, headers: dict[str, str]) -> Any:
        response = httpx.Response(status, headers=headers, request=request)
        return cls("error", response=response, body=None)

    rate = map_anthropic_error(status_error(anthropic.RateLimitError, 429, {"Retry-After": "3"}))
    assert isinstance(rate, TransientLLMError)
    assert (rate.status_code, rate.retry_after) == (429, 3.0)
    overloaded = map_anthropic_error(status_error(anthropic.APIStatusError, 529, {}))
    assert isinstance(overloaded, TransientLLMError)
    assert overloaded.status_code == 529
    auth = map_anthropic_error(status_error(anthropic.AuthenticationError, 401, {}))
    assert type(auth) is LLMError
    assert (auth.retryable, auth.status_code) == (False, 401)


class _RaisingMessages:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        raise self.exc


def test_anthropic_client_maps_sdk_errors_and_chains_cause() -> None:
    sdk_exc = RateLimitError("slow down", status_code=429, headers={"retry-after": "3"})
    fake = SimpleNamespace(messages=_RaisingMessages(sdk_exc))
    client = AnthropicClient(client=fake, fallbacks=None)
    with pytest.raises(TransientLLMError) as info:
        client.complete(system="s", messages=[Message(role="user", content="q")])
    assert info.value.__cause__ is sdk_exc
    assert info.value.retry_after == 3.0


def test_anthropic_client_reraises_non_sdk_errors_unchanged() -> None:
    bug = KeyError("bug")
    fake = SimpleNamespace(beta=SimpleNamespace(messages=_RaisingMessages(bug)))
    with pytest.raises(KeyError) as info:
        AnthropicClient(client=fake).complete(system="s", messages=[])
    assert info.value is bug


def test_anthropic_client_with_retrying_wrapper_retries_mapped_errors() -> None:
    from structured_output_agent.resilience import ProviderRetryPolicy, RetryingLLM

    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise APIStatusError("overloaded", status_code=529)
            return _text_response(usage=SimpleNamespace(input_tokens=3, output_tokens=4))

    messages = Flaky()
    sleeps: list[float] = []
    llm = RetryingLLM(
        AnthropicClient(client=SimpleNamespace(messages=messages), fallbacks=None),
        ProviderRetryPolicy(jitter=False),
        sleep=sleeps.append,
    )
    resp = llm.complete(system="s", messages=[])
    assert resp.usage == Usage(input_tokens=3, output_tokens=4)
    assert messages.calls == 2
    assert sleeps == [0.5]


# --- async client -----------------------------------------------------------


class _FakeAsyncMessages:
    def __init__(self, exc: Exception | None = None) -> None:
        self.kwargs: dict[str, Any] = {}
        self.exc = exc

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return _text_response(usage=SimpleNamespace(input_tokens=7, output_tokens=2))


class _FakeAsyncClient:
    def __init__(self, exc: Exception | None = None) -> None:
        self.messages = _FakeAsyncMessages(exc)
        self.beta = SimpleNamespace(messages=_FakeAsyncMessages(exc))


async def test_async_anthropic_client_uses_fallbacks_by_default() -> None:
    fake = _FakeAsyncClient()
    client = AsyncAnthropicClient(client=fake, effort="high")
    tools = [ToolSpec(name="w", description="weather", input_schema={"type": "object"})]
    resp = await client.complete(
        system="sys", messages=[Message(role="user", content="q")], tools=tools
    )

    sent = fake.beta.messages.kwargs
    assert isinstance(client, AsyncLLMClient)
    assert resp.text == "{}"
    assert resp.usage == Usage(input_tokens=7, output_tokens=2)
    assert sent["model"] == "claude-opus-5-5"
    assert sent["messages"] == [{"role": "user", "content": "q"}]
    assert sent["betas"] == ["server-side-fallback-2026-07-01"]
    assert sent["extra_body"] == {"fallbacks": "default"}
    assert sent["output_config"] == {"effort": "high"}
    assert sent["tools"][0]["name"] == "w"
    assert fake.messages.kwargs == {}


async def test_async_anthropic_client_without_fallbacks_uses_stable_endpoint() -> None:
    fake = _FakeAsyncClient()
    client = AsyncAnthropicClient("claude-sonnet-5-5", client=fake, fallbacks=None)
    await client.complete(system="sys", messages=[Message(role="user", content="q")])

    sent = fake.messages.kwargs
    assert sent["model"] == "claude-sonnet-5-5"
    assert "betas" not in sent
    assert "extra_body" not in sent
    assert fake.beta.messages.kwargs == {}


async def test_async_anthropic_client_maps_errors() -> None:
    sdk_exc = APIStatusError("unauthorized", status_code=401)
    client = AsyncAnthropicClient(client=_FakeAsyncClient(sdk_exc))
    with pytest.raises(LLMError) as info:
        await client.complete(system="s", messages=[])
    assert info.value.retryable is False
    assert info.value.__cause__ is sdk_exc

    bug = RuntimeError("bug")
    with pytest.raises(RuntimeError) as raw:
        await AsyncAnthropicClient(client=_FakeAsyncClient(bug)).complete(system="s", messages=[])
    assert raw.value is bug


async def test_async_anthropic_client_raises_on_refusal() -> None:
    class Refusing:
        async def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(stop_reason="refusal", stop_details=None, content=[])

    client = AsyncAnthropicClient(client=SimpleNamespace(messages=Refusing()), fallbacks=None)
    with pytest.raises(LLMError, match="declined") as info:
        await client.complete(system="s", messages=[])
    assert info.value.retryable is False


# --- SDK construction -------------------------------------------------------


@pytest.mark.parametrize(
    ("client_cls", "sdk_name"),
    [(AnthropicClient, "Anthropic"), (AsyncAnthropicClient, "AsyncAnthropic")],
)
def test_sdk_client_constructor_kwargs(
    monkeypatch: pytest.MonkeyPatch, client_cls: Any, sdk_name: str
) -> None:
    anthropic = pytest.importorskip("anthropic")
    created: list[dict[str, Any]] = []

    class SDK:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    monkeypatch.setattr(anthropic, sdk_name, SDK)

    assert isinstance(client_cls()._client, SDK)
    assert isinstance(client_cls(timeout=30.0, max_retries=0)._client, SDK)
    passed = object()
    assert client_cls(client=passed, timeout=1.0, max_retries=5)._client is passed

    assert created == [{}, {"timeout": 30.0, "max_retries": 0}]
