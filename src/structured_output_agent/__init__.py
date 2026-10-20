"""Structured Output Agent: make LLMs reliable, not random."""

import logging

from .agent import AsyncStructuredOutputAgent, StructuredOutputAgent
from .errors import (
    LLMError,
    OutputError,
    ParseError,
    RetryExhaustedError,
    RunTimeoutError,
    SchemaValidationError,
    StructuredOutputError,
    ToolArgumentError,
    ToolError,
    ToolExecutionError,
    ToolRegistrationError,
    ToolResponseError,
    ToolRoundLimitError,
    TransientLLMError,
    UnknownToolError,
)
from .hooks import HookDispatcher, RunHooks
from .llm import AnthropicClient, AsyncLLMClient, AsyncScriptedLLM, LLMClient, ScriptedLLM
from .parsing import errors_from_pydantic, extract_json, parse_output
from .retry import RetryPolicy, build_repair_message
from .schema import OutputContract
from .tools import Tool, ToolRegistry, ToolResult
from .types import (
    Attempt,
    ErrorDetail,
    FailureKind,
    LLMResponse,
    Message,
    RunResult,
    ToolCall,
    ToolSpec,
    Usage,
)
from .validation_log import LOGGER_NAME, FailureEvent, ReliabilityStats, ValidationLogger

# Library etiquette: stay silent unless the application configures logging.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())

__version__ = "0.2.0"

__all__ = [
    "LOGGER_NAME",
    "AnthropicClient",
    "AsyncLLMClient",
    "AsyncScriptedLLM",
    "AsyncStructuredOutputAgent",
    "Attempt",
    "ErrorDetail",
    "FailureEvent",
    "FailureKind",
    "HookDispatcher",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "Message",
    "OutputContract",
    "OutputError",
    "ParseError",
    "ReliabilityStats",
    "RetryExhaustedError",
    "RetryPolicy",
    "RunHooks",
    "RunResult",
    "RunTimeoutError",
    "SchemaValidationError",
    "ScriptedLLM",
    "StructuredOutputAgent",
    "StructuredOutputError",
    "Tool",
    "ToolArgumentError",
    "ToolCall",
    "ToolError",
    "ToolExecutionError",
    "ToolRegistrationError",
    "ToolRegistry",
    "ToolResponseError",
    "ToolResult",
    "ToolRoundLimitError",
    "ToolSpec",
    "TransientLLMError",
    "UnknownToolError",
    "Usage",
    "ValidationLogger",
    "__version__",
    "build_repair_message",
    "errors_from_pydantic",
    "extract_json",
    "parse_output",
]
