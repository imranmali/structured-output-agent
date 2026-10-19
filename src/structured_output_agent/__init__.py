"""Structured Output Agent: make LLMs reliable, not random."""

import logging

from .agent import StructuredOutputAgent
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
)
from .validation_log import LOGGER_NAME, FailureEvent, ReliabilityStats, ValidationLogger

# Library etiquette: stay silent unless the application configures logging.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())

__version__ = "0.1.0"

__all__ = [
    "LOGGER_NAME",
    "AnthropicClient",
    "Attempt",
    "ErrorDetail",
    "FailureEvent",
    "FailureKind",
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
    "RunResult",
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
    "UnknownToolError",
    "ValidationLogger",
    "__version__",
    "build_repair_message",
    "errors_from_pydantic",
    "extract_json",
    "parse_output",
]
