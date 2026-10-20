"""LLM client interface and implementations.

``LLMClient`` (sync) and ``AsyncLLMClient`` (async) are the only things the
agents need from a provider. Clients shipped with the library:

* ``ScriptedLLM`` / ``AsyncScriptedLLM`` play back canned responses. They are
  deterministic, need no network, and record every request, which makes them
  ideal for tests, demos and load tests.
* ``AnthropicClient`` / ``AsyncAnthropicClient`` call Claude through the
  official ``anthropic`` SDK (install the ``anthropic`` extra). SDK errors are
  mapped onto ``TransientLLMError`` / ``LLMError`` so ``RetryingLLM`` can tell
  retryable failures from permanent ones, and token usage is reported on
  every response.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .errors import LLMError, TransientLLMError
from .types import LLMResponse, Message, ToolCall, ToolSpec, Usage

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic chat completion interface."""

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse: ...


@runtime_checkable
class AsyncLLMClient(Protocol):
    """Async counterpart of ``LLMClient``."""

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse: ...


ScriptStep = (
    str | LLMResponse | BaseException | Callable[[str, Sequence[Message]], str | LLMResponse]
)


@dataclass
class RecordedCall:
    """A snapshot of one request made to ``ScriptedLLM``."""

    system: str
    messages: list[Message]
    tools: list[ToolSpec]


class _Script:
    def __init__(self, script: Iterable[ScriptStep]) -> None:
        self.script = script
        self.calls: list[RecordedCall] = []
        self._steps = list(script)

    @property
    def remaining(self) -> int:
        return len(self._steps)

    def _next(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        self.calls.append(RecordedCall(system=system, messages=list(messages), tools=list(tools)))
        if not self._steps:
            raise LLMError(f"{type(self).__name__} exhausted after {len(self.calls) - 1} responses")
        step = self._steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            step = step(system, messages)
        if isinstance(step, str):
            return LLMResponse(text=step)
        return step


class ScriptedLLM(_Script):
    """Deterministic client that returns scripted responses in order.

    Each step is a string (answer text), an ``LLMResponse`` (for tool calls or
    usage), an exception instance (raised, to simulate provider failures), or a
    callable receiving ``(system, messages)`` and returning a string or
    ``LLMResponse``.
    """

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse:
        return self._next(system, messages, tools)


class AsyncScriptedLLM(_Script):
    """Async ``ScriptedLLM``; ``delay`` seconds simulate model latency."""

    def __init__(self, script: Iterable[ScriptStep], *, delay: float = 0.0) -> None:
        super().__init__(script)
        self.delay = delay

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse:
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._next(system, messages, tools)


_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class _AnthropicMessagesAPI:
    """Request building and SDK setup shared by the sync and async clients."""

    _sdk_class = "Anthropic"

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        max_tokens: int = 16000,
        effort: str | None = None,
        fallbacks: str | None = "default",
        client: Any = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        if client is None:
            client = _make_sdk_client(self._sdk_class, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.fallbacks = fallbacks
        self._client = client

    def _prepare(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Return the SDK ``create`` method to call and its keyword arguments."""
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": to_anthropic_messages(messages),
        }
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            params["betas"] = [_FALLBACK_BETA]
            params["extra_body"] = {"fallbacks": self.fallbacks}
            return self._client.beta.messages.create, params
        return self._client.messages.create, params


def _make_sdk_client(class_name: str, *, timeout: float | None, max_retries: int | None) -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise LLMError(
            f"{class_name}Client requires the 'anthropic' package: "
            "pip install 'structured-output-agent[anthropic]'"
        ) from exc
    kwargs: dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return getattr(anthropic, class_name)(**kwargs)


class AnthropicClient(_AnthropicMessagesAPI):
    """``LLMClient`` backed by the Claude Messages API.

    Assistant turns are replayed with their original content blocks (kept on
    ``Message.raw``) so the conversation stays append-only. Server-side refusal
    fallbacks are enabled by default; pass ``fallbacks=None`` to disable them.

    ``timeout`` and ``max_retries`` configure the SDK client created when
    ``client`` is not given (``None`` keeps the SDK defaults). Pass
    ``max_retries=0`` when wrapping the client in ``RetryingLLM``. SDK errors
    are raised as ``TransientLLMError`` (rate limits, overload, 5xx, timeouts,
    connection failures) or permanent ``LLMError``; see ``map_anthropic_error``.
    """

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse:
        create, params = self._prepare(system, messages, tools)
        try:
            response = create(**params)
        except Exception as exc:
            mapped = map_anthropic_error(exc)
            if mapped is exc:
                raise
            raise mapped from exc
        return from_anthropic_response(response)


class AsyncAnthropicClient(_AnthropicMessagesAPI):
    """``AsyncLLMClient`` backed by the Claude Messages API.

    Same behaviour and parameters as ``AnthropicClient``, using
    ``anthropic.AsyncAnthropic``.
    """

    _sdk_class = "AsyncAnthropic"

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse:
        create, params = self._prepare(system, messages, tools)
        try:
            response = await create(**params)
        except Exception as exc:
            mapped = map_anthropic_error(exc)
            if mapped is exc:
                raise
            raise mapped from exc
        return from_anthropic_response(response)


_SDK_BASE_NAMES = frozenset({"AnthropicError", "APIError"})
_CONNECTION_ERROR_NAMES = frozenset({"APIConnectionError", "APITimeoutError"})
_TRANSIENT_STATUSES = frozenset({408, 409, 429})


def map_anthropic_error(exc: BaseException) -> BaseException:
    """Translate an ``anthropic`` SDK exception into a library ``LLMError``.

    Timeouts, connection failures and HTTP 408, 409, 429 and 5xx (including
    529 overloaded) become ``TransientLLMError`` with ``retry_after`` taken
    from the ``retry-after-ms`` / ``retry-after`` response headers; other API
    errors become permanent ``LLMError``. Exceptions that do not come from the
    SDK are returned unchanged. Works by duck typing, so the SDK need not be
    importable.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    if not names & _SDK_BASE_NAMES:
        return exc
    detail = f"{type(exc).__name__}: {exc}"
    if names & _CONNECTION_ERROR_NAMES:
        return TransientLLMError(f"Anthropic API connection failed ({detail})")
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = None
    retry_after = _parse_retry_after(exc)
    if status is not None and (status in _TRANSIENT_STATUSES or status >= 500):
        return TransientLLMError(
            f"Anthropic API transient error {status} ({detail})",
            retry_after=retry_after,
            status_code=status,
        )
    label = f"error {status}" if status is not None else "error"
    return LLMError(
        f"Anthropic API {label} ({detail})",
        retryable=False,
        retry_after=retry_after,
        status_code=status,
    )


def _parse_retry_after(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        try:
            seconds = float(headers.get(name)) / scale
        except (TypeError, ValueError, AttributeError):
            continue  # missing, HTTP-date or otherwise unparseable
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return None


def to_anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert library messages to Messages API ``messages``.

    Consecutive tool results are merged into one user turn, as the API expects
    all results for a parallel tool call in a single message.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content,
                "is_error": msg.is_error,
            }
            if out and out[-1]["role"] == "user" and _is_tool_result_turn(out[-1]):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif msg.role == "assistant":
            if msg.raw is not None:
                out.append({"role": "assistant", "content": msg.raw})
                continue
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            content.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in msg.tool_calls
            )
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": "user", "content": msg.content})
    return out


def _is_tool_result_turn(turn: dict[str, Any]) -> bool:
    content = turn["content"]
    return isinstance(content, list) and all(b.get("type") == "tool_result" for b in content)


def from_anthropic_response(response: Any) -> LLMResponse:
    """Convert a Messages API response into an ``LLMResponse``.

    A refusal (``stop_reason == "refusal"``) raises a permanent ``LLMError``.
    """
    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details is not None else None
        raise LLMError(f"Model declined the request (category: {category})")
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            texts.append(block.text)
        elif block_type == "tool_use":
            args = block.input
            if isinstance(args, str):
                args = json.loads(args)
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(args)))
    return LLMResponse(
        text="".join(texts), tool_calls=calls, raw=response.content, usage=_usage(response)
    )


def _usage(response: Any) -> Usage:
    """Token counts from ``response.usage``; zeros when the provider omits them."""
    usage = getattr(response, "usage", None)
    return Usage(
        input_tokens=_count(getattr(usage, "input_tokens", None)),
        output_tokens=_count(getattr(usage, "output_tokens", None)),
    )


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
