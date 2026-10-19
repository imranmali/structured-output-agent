"""``structured-agent`` command line interface.

Subcommands:

* ``demo``   - offline walkthrough with a scripted model that makes mistakes.
* ``schema`` - print the JSON Schema the model is shown for a Pydantic model.
* ``run``    - run a prompt against Claude and print the validated JSON.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from typing import Any, TextIO

from pydantic import BaseModel, Field

from . import __version__
from .agent import StructuredOutputAgent
from .errors import StructuredOutputError
from .llm import DEFAULT_ANTHROPIC_MODEL, AnthropicClient, LLMClient, ScriptedLLM
from .retry import RetryPolicy
from .schema import OutputContract
from .tools import ToolRegistry
from .types import LLMResponse, ToolCall
from .validation_log import ValidationLogger


class WeatherReport(BaseModel):
    """Demo output contract."""

    city: str
    temperature_c: float
    conditions: str
    advice: str = Field(description="One sentence of practical advice")


class Weather(BaseModel):
    """Demo tool response contract."""

    temperature_c: float
    conditions: str


def demo_tools() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register
    def get_weather(city: str) -> Weather:
        """Look up the current weather for a city."""
        return Weather(temperature_c=18.5, conditions="light rain")

    return registry


def demo_llm() -> ScriptedLLM:
    """A scripted model that makes the mistakes real models make."""
    return ScriptedLLM(
        [
            # Run 1: prose around broken JSON -> parse error.
            'Sure! Here is the report: {"city": "Paris", "temperature_c": 18.5,, }',
            # Missing and mistyped fields -> validation error.
            '{"city": "Paris", "temperature_c": "warm"}',
            # Tool call with the wrong argument name -> rejected before execution.
            LLMResponse(text="", tool_calls=[ToolCall("c1", "get_weather", {"town": "Paris"})]),
            # Corrected tool call -> executed, response validated.
            LLMResponse(text="", tool_calls=[ToolCall("c2", "get_weather", {"city": "Paris"})]),
            # Fenced but valid JSON -> success.
            '```json\n{"city": "Paris", "temperature_c": 18.5, "conditions": "light rain", '
            '"advice": "Take an umbrella."}\n```',
            # Run 2: valid on the first attempt.
            '{"city": "Oslo", "temperature_c": -3, "conditions": "snow", "advice": "Wear boots."}',
        ]
    )


def _naive_parse(text: str) -> str:
    try:
        WeatherReport.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValueError) as exc:
        return f"FAILED ({type(exc).__name__})"
    return "ok"


def cmd_demo(args: argparse.Namespace, out: TextIO) -> int:
    llm = demo_llm()
    naive = [s for s in llm.script if isinstance(s, str)]
    logger = ValidationLogger(path=args.log_file)
    agent = StructuredOutputAgent(
        llm, tools=demo_tools(), retry=RetryPolicy(max_attempts=4), logger=logger
    )

    out.write("Structured Output Agent demo (offline, scripted model)\n\n")
    out.write("Naive json.loads + model_validate on the same raw outputs:\n")
    for i, text in enumerate(naive, 1):
        out.write(f"  output {i}: {_naive_parse(text)}\n")

    for prompt in ("Weather report for Paris", "Weather report for Oslo"):
        result = agent.run(prompt, WeatherReport)
        out.write(f"\n> {prompt}\n")
        for failure in result.failures:
            out.write(f"  attempt {failure.number}: {failure.kind.value}\n")
            for err in failure.errors:
                out.write(f"    - {err}\n")
        out.write(
            f"  OK after {result.attempts} attempt(s), {result.tool_calls} tool call(s): "
            f"{result.output.model_dump_json()}\n"
        )

    out.write("\nLogged failures:\n")
    for event in logger.events:
        out.write(f"  [{event.kind.value}] {event.contract} attempt {event.attempt}\n")
    out.write("\nReliability stats:\n")
    out.write(json.dumps(logger.stats.to_dict(), indent=2) + "\n")
    if args.log_file:
        out.write(f"\nFailure events written to {args.log_file}\n")
    return 0


def load_model(spec: str) -> type[BaseModel]:
    """Import ``package.module:ClassName`` and check it is a Pydantic model."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"Expected 'module:ClassName', got {spec!r}")
    obj: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise TypeError(f"{spec} is not a Pydantic model class")
    return obj


def cmd_schema(args: argparse.Namespace, out: TextIO) -> int:
    contract = OutputContract(load_model(args.model), strict=args.strict)
    out.write(json.dumps(contract.json_schema(), indent=2, sort_keys=True) + "\n")
    return 0


def cmd_run(args: argparse.Namespace, out: TextIO, llm: LLMClient | None = None) -> int:
    model = load_model(args.model)
    client = llm if llm is not None else AnthropicClient(args.llm_model)
    agent = StructuredOutputAgent(
        client,
        retry=RetryPolicy(max_attempts=args.max_attempts),
        logger=ValidationLogger(path=args.log_file),
        strict=args.strict,
    )
    result = agent.run(args.prompt, model)
    out.write(result.output.model_dump_json(indent=2) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structured-agent",
        description="Make LLMs reliable, not random: schema-enforced, validated, retried output.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="offline demo with a scripted model")
    demo.add_argument("--log-file", help="append failure events to this JSONL file")
    demo.set_defaults(func=cmd_demo)

    schema = sub.add_parser("schema", help="print the JSON Schema for a Pydantic model")
    schema.add_argument("model", help="model to load, as module:ClassName")
    schema.add_argument("--strict", action="store_true", help="forbid unknown fields")
    schema.set_defaults(func=cmd_schema)

    run = sub.add_parser("run", help="run a prompt against Claude (needs the anthropic extra)")
    run.add_argument("model", help="output model, as module:ClassName")
    run.add_argument("prompt", help="the user prompt")
    run.add_argument("--llm-model", default=DEFAULT_ANTHROPIC_MODEL, help="Claude model id")
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--strict", action="store_true", help="forbid unknown fields")
    run.add_argument("--log-file", help="append failure events to this JSONL file")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args, sys.stdout)
    except (StructuredOutputError, ValueError, TypeError, ImportError, AttributeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
