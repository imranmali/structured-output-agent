"""LLM client interface and implementations.

``LLMClient`` (sync) and ``AsyncLLMClient`` (async) are the only things the
agents need from a provider. Clients shipped with the library:

* ``ScriptedLLM`` / ``AsyncScriptedLLM`` play back canned responses. They are
  deterministic, need no network, and record every request, which makes them
  ideal for tests, demos and load tests.
* ``AnthropicClient`` calls Claude through the official ``anthropic`` SDK
  (install the ``anthropic`` extra).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .errors import LLMError
from .types import LLMResponse, Message, ToolCall, ToolSpec

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


class AnthropicClient:
    """``LLMClient`` backed by the Claude Messages API.

    Assistant turns are replayed with their original content blocks (kept on
    ``Message.raw``) so the conversation stays append-only. Server-side refusal
    fallbacks are enabled by default; pass ``fallbacks=None`` to disable them.
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        max_tokens: int = 16000,
        effort: str | None = None,
        fallbacks: str | None = "default",
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise LLMError(
                    "AnthropicClient requires the 'anthropic' package: "
                    "pip install 'structured-output-agent[anthropic]'"
                ) from exc
            client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.fallbacks = fallbacks
        self._client = client

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse:
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
            response = self._client.beta.messages.create(
                **params,
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": self.fallbacks},
            )
        else:
            response = self._client.messages.create(**params)
        return from_anthropic_response(response)


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
    """Convert a Messages API response into an ``LLMResponse``."""
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
    return LLMResponse(text="".join(texts), tool_calls=calls, raw=response.content)
