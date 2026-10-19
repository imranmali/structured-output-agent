"""Exception hierarchy. Every library error derives from StructuredOutputError."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .types import Attempt, ErrorDetail, FailureKind


class StructuredOutputError(Exception):
    """Base class for all errors raised by this library."""


class OutputError(StructuredOutputError):
    """Model output could not be turned into a valid object."""

    kind: FailureKind

    def __init__(self, message: str, *, raw: str, errors: Sequence[ErrorDetail]) -> None:
        super().__init__(message)
        self.raw = raw
        self.errors = list(errors)


class ParseError(OutputError):
    """No syntactically valid JSON could be extracted from the model output."""

    kind = FailureKind.PARSE_ERROR

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(
            message, raw=raw, errors=[ErrorDetail(loc="$", msg=message, type="json_parse")]
        )


class SchemaValidationError(OutputError):
    """JSON was extracted but does not satisfy the output contract."""

    kind = FailureKind.VALIDATION_ERROR

    def __init__(
        self, message: str, *, raw: str, errors: Sequence[ErrorDetail], payload: Any = None
    ) -> None:
        super().__init__(message, raw=raw, errors=errors)
        self.payload = payload


class RetryExhaustedError(StructuredOutputError):
    """Every allowed attempt produced invalid output."""

    def __init__(self, message: str, *, attempts: Sequence[Attempt]) -> None:
        super().__init__(message)
        self.attempts = list(attempts)


class ToolRegistrationError(StructuredOutputError):
    """A tool could not be registered (for example a duplicate name)."""


class ToolError(StructuredOutputError):
    """A tool call failed. Returned to the model rather than raised in the loop."""

    kind: FailureKind

    def __init__(self, message: str, *, tool_name: str, errors: Sequence[ErrorDetail] = ()) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.errors = list(errors) or [ErrorDetail(loc="$", msg=message, type=self.kind.value)]


class UnknownToolError(ToolError):
    kind = FailureKind.UNKNOWN_TOOL


class ToolArgumentError(ToolError):
    kind = FailureKind.TOOL_ARGUMENT_ERROR


class ToolExecutionError(ToolError):
    kind = FailureKind.TOOL_EXECUTION_ERROR


class ToolResponseError(ToolError):
    kind = FailureKind.TOOL_RESPONSE_ERROR


class ToolRoundLimitError(StructuredOutputError):
    """The model kept calling tools past the configured round limit."""


class LLMError(StructuredOutputError):
    """The LLM provider failed or declined to answer."""
