"""The agent loop: contract -> generate -> (tools) -> parse -> validate -> retry."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from .errors import OutputError, RetryExhaustedError, ToolRoundLimitError
from .llm import LLMClient
from .parsing import parse_output
from .retry import RetryPolicy, build_repair_message
from .schema import OutputContract
from .tools import ToolRegistry
from .types import Attempt, Message, RunResult
from .validation_log import ValidationLogger

T = TypeVar("T", bound=BaseModel)


class StructuredOutputAgent:
    """Runs prompts against an LLM and returns validated Pydantic objects.

    Every answer is parsed and validated against the requested model. Invalid
    answers are sent back to the model with itemised errors until the retry
    policy is exhausted. Tool calls are validated in both directions and do not
    consume retry attempts; they are capped by ``max_tool_rounds`` instead.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        tools: ToolRegistry | None = None,
        retry: RetryPolicy | None = None,
        logger: ValidationLogger | None = None,
        system_prompt: str | None = None,
        strict: bool = False,
        max_tool_rounds: int = 8,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be >= 0")
        self.llm = llm
        self.tools = tools if tools is not None else ToolRegistry()
        self.retry = retry if retry is not None else RetryPolicy()
        self.logger = logger if logger is not None else ValidationLogger()
        self.system_prompt = system_prompt
        self.strict = strict
        self.max_tool_rounds = max_tool_rounds
        self._sleep = sleep

    def run(
        self,
        prompt: str,
        output_model: type[T],
        *,
        strict: bool | None = None,
        system_prompt: str | None = None,
    ) -> RunResult[T]:
        """Return ``output_model`` parsed from the model's answer to ``prompt``.

        Raises ``RetryExhaustedError`` when no attempt yields valid output and
        ``ToolRoundLimitError`` when the model keeps calling tools.
        """
        contract = OutputContract(output_model, strict=self.strict if strict is None else strict)
        system = contract.system_prompt(system_prompt or self.system_prompt)
        specs = self.tools.specs()
        run_id = self.logger.new_run_id()
        messages: list[Message] = [Message(role="user", content=prompt)]
        failures: list[Attempt] = []
        attempt = 1
        tool_rounds = 0
        tool_calls = 0

        while True:
            response = self.llm.complete(system=system, messages=messages, tools=specs)

            if response.tool_calls:
                tool_rounds += 1
                if tool_rounds > self.max_tool_rounds:
                    self.logger.record_run(
                        contract=contract.name, success=False, attempts=attempt, run_id=run_id
                    )
                    raise ToolRoundLimitError(
                        f"Exceeded the tool round limit of {self.max_tool_rounds}"
                    )
                messages.append(
                    Message(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls,
                        raw=response.raw,
                    )
                )
                for call in response.tool_calls:
                    tool_calls += 1
                    result = self.tools.execute(call)
                    if result.error is not None:
                        self.logger.record(
                            result.error.kind,
                            contract=call.name,
                            attempt=attempt,
                            errors=result.error.errors,
                            raw=result.content,
                            run_id=run_id,
                        )
                    messages.append(
                        Message(
                            role="tool",
                            content=result.content,
                            tool_call_id=call.id,
                            is_error=not result.ok,
                        )
                    )
                continue

            try:
                output = parse_output(response.text, contract)
            except OutputError as exc:
                failures.append(
                    Attempt(number=attempt, raw=response.text, kind=exc.kind, errors=exc.errors)
                )
                self.logger.record(
                    exc.kind,
                    contract=contract.name,
                    attempt=attempt,
                    errors=exc.errors,
                    raw=response.text,
                    run_id=run_id,
                )
                if attempt >= self.retry.max_attempts:
                    self.logger.record_run(
                        contract=contract.name, success=False, attempts=attempt, run_id=run_id
                    )
                    raise RetryExhaustedError(
                        f"No valid {contract.name} after {attempt} attempt(s): {exc}",
                        attempts=failures,
                    ) from exc
                messages.append(Message(role="assistant", content=response.text, raw=response.raw))
                messages.append(
                    Message(
                        role="user",
                        content=build_repair_message(
                            response.text, exc.errors, exc.kind, contract_name=contract.name
                        ),
                    )
                )
                attempt += 1
                self.retry.wait(attempt, sleep=self._sleep)
                continue

            messages.append(Message(role="assistant", content=response.text, raw=response.raw))
            self.logger.record_run(
                contract=contract.name, success=True, attempts=attempt, run_id=run_id
            )
            return RunResult(
                output=output,
                attempts=attempt,
                failures=failures,
                tool_calls=tool_calls,
                messages=messages,
            )
