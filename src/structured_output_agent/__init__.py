"""Structured Output Agent: make LLMs reliable, not random."""

from .errors import (
    LLMError,
    OutputError,
    ParseError,
    RetryExhaustedError,
    SchemaValidationError,
    StructuredOutputError,
    ToolArgumentError,
    ToolError,
    ToolExecutionError,
    ToolRegistrationError,
    ToolResponseError,
    ToolRoundLimitError,
    UnknownToolError,
)
from .llm import AnthropicClient, LLMClient, ScriptedLLM
from .types import (
    Attempt,
    ErrorDetail,
    FailureKind,
    LLMResponse,
    Message,
    RunResult,
    ToolCall,
    ToolSpec,
)

__version__ = "0.1.0"

__all__ = [
    "AnthropicClient",
    "Attempt",
    "ErrorDetail",
    "FailureKind",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "Message",
    "OutputError",
    "ParseError",
    "RetryExhaustedError",
    "RunResult",
    "SchemaValidationError",
    "ScriptedLLM",
    "StructuredOutputError",
    "ToolArgumentError",
    "ToolCall",
    "ToolError",
    "ToolExecutionError",
    "ToolRegistrationError",
    "ToolResponseError",
    "ToolRoundLimitError",
    "ToolSpec",
    "UnknownToolError",
    "__version__",
]
