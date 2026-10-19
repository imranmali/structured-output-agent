"""Shared data types used across the library.

These are plain dataclasses so every module (schema, parsing, retry, tools,
logging, agent) can depend on them without depending on each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T")

Role = Literal["user", "assistant", "tool"]


class FailureKind(str, Enum):
    """Category of a recorded failure."""

    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_TOOL = "unknown_tool"
    TOOL_ARGUMENT_ERROR = "tool_argument_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    TOOL_RESPONSE_ERROR = "tool_response_error"


@dataclass(frozen=True)
class ErrorDetail:
    """One itemised problem with a payload.

    ``loc`` is a dotted path to the offending field (``"$"`` for the whole
    payload), ``msg`` a human readable message, ``type`` a machine readable
    error code and ``input`` the offending value when known.
    """

    loc: str
    msg: str
    type: str
    input: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"loc": self.loc, "msg": self.msg, "type": self.type, "input": self.input}

    def __str__(self) -> str:
        return f"{self.loc}: {self.msg}"


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to run a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    """A tool definition advertised to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Message:
    """One conversation turn.

    ``raw`` carries a provider-native representation of the turn (for example
    the content blocks of an Anthropic response) so provider clients can send
    assistant turns back unchanged.
    """

    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    is_error: bool = False
    raw: Any = None


@dataclass
class LLMResponse:
    """What an LLM client returns: candidate answer text and/or tool calls."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


@dataclass
class Attempt:
    """A failed generation attempt kept for diagnostics and retry history."""

    number: int
    raw: str
    kind: FailureKind
    errors: list[ErrorDetail]


@dataclass
class RunResult(Generic[T]):
    """Successful outcome of an agent run."""

    output: T
    attempts: int
    failures: list[Attempt] = field(default_factory=list)
    tool_calls: int = 0
    messages: list[Message] = field(default_factory=list)
