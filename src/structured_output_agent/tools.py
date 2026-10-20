"""Tool registry with validated arguments and responses.

Tools are plain Python callables with a Pydantic argument model (generated from
the signature when not given) and an optional response model. ``ToolRegistry``
advertises their JSON Schemas to the model and executes tool calls, validating
arguments before the call and the return value after it. Tool problems are
never raised from ``execute``; they come back as an error ``ToolResult`` whose
content can be sent to the model so it can recover.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, get_type_hints, overload

from pydantic import BaseModel, ValidationError, create_model
from pydantic_core import to_jsonable_python

from .errors import (
    ToolArgumentError,
    ToolError,
    ToolExecutionError,
    ToolRegistrationError,
    ToolResponseError,
    UnknownToolError,
)
from .parsing import errors_from_pydantic
from .types import ErrorDetail, ToolCall, ToolSpec

F = TypeVar("F", bound=Callable[..., Any])

_UNSUPPORTED_PARAMS = {
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
    inspect.Parameter.POSITIONAL_ONLY: "positional-only parameters",
}


@dataclass(frozen=True)
class Tool:
    """A callable plus the contracts for its arguments and response."""

    name: str
    description: str
    func: Callable[..., Any]
    args_model: type[BaseModel]
    response_model: type[BaseModel] | None = None
    timeout: float | None = None

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.func)

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        args_model: type[BaseModel] | None = None,
        response_model: type[BaseModel] | None = None,
        timeout: float | None = None,
    ) -> Tool:
        """Build a tool, deriving missing pieces from the function.

        The name defaults to ``func.__name__``, the description to the first
        docstring paragraph, the argument model to one generated from the
        signature, and the response model to the return annotation when it is
        a Pydantic model.
        """
        if timeout is not None and timeout <= 0:
            raise ToolRegistrationError(f"Tool timeout must be > 0, got {timeout}")
        tool_name = name or getattr(func, "__name__", None)
        if not tool_name:
            raise ToolRegistrationError("Tool name is required for callables without __name__")
        if description is None:
            description = _first_paragraph(inspect.getdoc(func))
        if args_model is None or response_model is None:
            hints = _type_hints(func, tool_name)
            if args_model is None:
                args_model = _args_model_from_signature(func, tool_name, hints)
            if response_model is None:
                returned = hints.get("return")
                if isinstance(returned, type) and issubclass(returned, BaseModel):
                    response_model = returned
        return cls(
            name=tool_name,
            description=description,
            func=func,
            args_model=args_model,
            response_model=response_model,
            timeout=timeout,
        )

    def spec(self) -> ToolSpec:
        """The definition advertised to the model."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.args_model.model_json_schema(),
        )


@dataclass
class ToolResult:
    """Outcome of one tool call.

    ``content`` is the JSON string sent back to the model: the validated
    response on success, or ``{"error": ..., "details": [...]}`` on failure.
    ``value`` holds the validated response object and ``error`` the failure.
    """

    call_id: str
    name: str
    ok: bool
    content: str
    value: Any = None
    error: ToolError | None = None
    arguments: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Named collection of tools that validates every call it executes."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    @overload
    def register(
        self,
        func: F,
        /,
        *,
        name: str | None = ...,
        description: str | None = ...,
        args_model: type[BaseModel] | None = ...,
        response_model: type[BaseModel] | None = ...,
        timeout: float | None = ...,
    ) -> F: ...

    @overload
    def register(
        self,
        func: None = ...,
        /,
        *,
        name: str | None = ...,
        description: str | None = ...,
        args_model: type[BaseModel] | None = ...,
        response_model: type[BaseModel] | None = ...,
        timeout: float | None = ...,
    ) -> Callable[[F], F]: ...

    def register(
        self,
        func: F | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        args_model: type[BaseModel] | None = None,
        response_model: type[BaseModel] | None = None,
        timeout: float | None = None,
    ) -> F | Callable[[F], F]:
        """Register a function as a tool; usable bare, with options, or directly.

        Returns the original function unchanged. See ``Tool.from_function`` for
        how missing options are derived.
        """

        def decorator(fn: F) -> F:
            self.add(
                Tool.from_function(
                    fn,
                    name=name,
                    description=description,
                    args_model=args_model,
                    response_model=response_model,
                    timeout=timeout,
                )
            )
            return fn

        return decorator if func is None else decorator(func)

    def add(self, tool: Tool) -> None:
        """Add a prebuilt tool. Raises ``ToolRegistrationError`` on a duplicate name."""
        if tool.name in self._tools:
            raise ToolRegistrationError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        """Registered tool names in registration order."""
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        """Definitions of every tool, for sending to the model."""
        return [tool.spec() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def execute(self, call: ToolCall) -> ToolResult:
        """Run a tool call, returning an error result instead of raising.

        Coroutine tools are run to completion with ``asyncio.run``. A tool with
        a ``timeout`` runs in a worker thread; on timeout the call fails fast
        but the thread finishes in the background.
        """
        prepared = self._prepare(call)
        if isinstance(prepared, ToolResult):
            return prepared
        tool, kwargs = prepared
        try:
            if tool.is_async:
                returned = asyncio.run(_await_with_timeout(tool, tool.func(**kwargs)))
            elif tool.timeout is not None:
                future = _executor().submit(tool.func, **kwargs)
                returned = future.result(timeout=tool.timeout)
            else:
                returned = tool.func(**kwargs)
        except Exception as exc:
            return _failure(call, _execution_error(tool, exc))
        return _finish(call, tool, returned)

    async def execute_async(self, call: ToolCall) -> ToolResult:
        """Async ``execute``: coroutine tools are awaited, sync tools run in a thread."""
        prepared = self._prepare(call)
        if isinstance(prepared, ToolResult):
            return prepared
        tool, kwargs = prepared
        try:
            if tool.is_async:
                returned = await _await_with_timeout(tool, tool.func(**kwargs))
            else:
                returned = await _await_with_timeout(
                    tool, asyncio.to_thread(functools.partial(tool.func, **kwargs))
                )
        except Exception as exc:
            return _failure(call, _execution_error(tool, exc))
        return _finish(call, tool, returned)

    def _prepare(self, call: ToolCall) -> tuple[Tool, dict[str, Any]] | ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self.names) or "none"
            return _failure(
                call,
                UnknownToolError(
                    f"Unknown tool '{call.name}'. Available tools: {available}",
                    tool_name=call.name,
                ),
            )
        try:
            validated = tool.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            errors = errors_from_pydantic(exc)
            return _failure(
                call,
                ToolArgumentError(
                    f"Invalid arguments for tool '{tool.name}': {_summary(errors)}",
                    tool_name=tool.name,
                    errors=errors,
                ),
            )
        return tool, {attr: getattr(validated, attr) for attr in tool.args_model.model_fields}


_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None


def _executor() -> concurrent.futures.ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = concurrent.futures.ThreadPoolExecutor(thread_name_prefix="soa-tool")
    return _EXECUTOR


async def _await_with_timeout(tool: Tool, awaitable: Awaitable[Any]) -> Any:
    if tool.timeout is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=tool.timeout)


def _execution_error(tool: Tool, exc: Exception) -> ToolExecutionError:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, concurrent.futures.TimeoutError)):
        return ToolExecutionError(
            f"Tool '{tool.name}' timed out after {tool.timeout}s", tool_name=tool.name
        )
    return ToolExecutionError(f"{type(exc).__name__}: {exc}", tool_name=tool.name)


def _finish(call: ToolCall, tool: Tool, returned: Any) -> ToolResult:
    try:
        value, content = _serialise_response(tool, returned)
    except ToolResponseError as exc:
        return _failure(call, exc)
    return ToolResult(
        call_id=call.id,
        name=call.name,
        ok=True,
        content=content,
        value=value,
        arguments=call.arguments,
    )


def _serialise_response(tool: Tool, returned: Any) -> tuple[Any, str]:
    if tool.response_model is not None:
        try:
            model = tool.response_model.model_validate(returned)
        except ValidationError as exc:
            errors = errors_from_pydantic(exc)
            raise ToolResponseError(
                f"Tool '{tool.name}' returned an invalid response: {_summary(errors)}",
                tool_name=tool.name,
                errors=errors,
            ) from exc
        return model, json.dumps(model.model_dump(mode="json"))
    try:
        return returned, json.dumps(to_jsonable_python(returned))
    except (TypeError, ValueError) as exc:
        raise ToolResponseError(
            f"Tool '{tool.name}' returned a value that is not JSON serialisable: "
            f"{type(returned).__name__}",
            tool_name=tool.name,
        ) from exc


def _failure(call: ToolCall, error: ToolError) -> ToolResult:
    content = json.dumps(
        {
            "error": str(error),
            "details": [{"loc": detail.loc, "msg": detail.msg} for detail in error.errors],
        }
    )
    return ToolResult(
        call_id=call.id,
        name=call.name,
        ok=False,
        content=content,
        error=error,
        arguments=call.arguments,
    )


def _summary(errors: list[ErrorDetail]) -> str:
    return "; ".join(str(error) for error in errors)


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    return " ".join(doc.split("\n\n", 1)[0].split())


def _type_hints(func: Callable[..., Any], tool_name: str) -> dict[str, Any]:
    try:
        return get_type_hints(func)
    except Exception as exc:
        raise ToolRegistrationError(
            f"Cannot resolve type hints for tool '{tool_name}': {exc}"
        ) from exc


def _args_model_from_signature(
    func: Callable[..., Any], tool_name: str, hints: dict[str, Any]
) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for param in inspect.signature(func).parameters.values():
        if param.kind in _UNSUPPORTED_PARAMS:
            raise ToolRegistrationError(
                f"Tool '{tool_name}' cannot use {_UNSUPPORTED_PARAMS[param.kind]} "
                f"('{param.name}'); pass an explicit args_model instead"
            )
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (hints.get(param.name, Any), default)
    model_name = "".join(part.capitalize() for part in tool_name.split("_")) + "Arguments"
    try:
        model: type[BaseModel] = create_model(model_name, **fields)
        model.model_json_schema()
    except Exception as exc:
        raise ToolRegistrationError(
            f"Cannot build an argument model for tool '{tool_name}': {exc}"
        ) from exc
    return model
