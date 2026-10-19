# Structured Output Agent

[![CI](https://github.com/imranmali/structured-output-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/imranmali/structured-output-agent/actions/workflows/ci.yml)
[![CD](https://github.com/imranmali/structured-output-agent/actions/workflows/cd.yml/badge.svg)](https://github.com/imranmali/structured-output-agent/actions/workflows/cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)

**Make LLMs reliable, not random.** Structured Output Agent wraps any chat LLM so
that every answer is a validated Pydantic object, or a precise, logged error.

- **Enforce Pydantic JSON schemas**: the model is shown your schema on every request.
- **Validate tool responses**: tool arguments are checked before a tool runs and
  tool results are checked before the model sees them.
- **Retry on parse errors**: bad output goes back to the model with itemised
  errors, under a bounded retry policy with exponential backoff.
- **Log validation failures**: every failure becomes a structured event (stdlib
  logging and JSON lines), with success-rate statistics.

```text
prompt ──► LLM ──► extract JSON ──► validate (Pydantic) ──► typed object ✔
            ▲          │ parse error        │ validation error
            │          ▼                    ▼
            └──── repair prompt with itemised errors (bounded retries, backoff)
            │
            └──► tool call ──► validate args ──► run ──► validate response ──┐
                 ▲                                                         │
                 └─────────────────── tool result / tool error ◄───────────┘
```

## Install

```bash
pip install "structured-output-agent @ git+https://github.com/imranmali/structured-output-agent"
# with the Claude client
pip install "structured-output-agent[anthropic] @ git+https://github.com/imranmali/structured-output-agent"
```

Requires Python 3.10+. The only runtime dependency is `pydantic>=2`.

## Quick start

```python
from pydantic import BaseModel

from structured_output_agent import AnthropicClient, StructuredOutputAgent


class Person(BaseModel):
    name: str
    age: int


agent = StructuredOutputAgent(AnthropicClient())  # uses ANTHROPIC_API_KEY
result = agent.run("Ada Lovelace was 36 when she died.", Person)

print(result.output)    # name='Ada Lovelace' age=36
print(result.attempts)  # 1, or more if the model needed repairing
```

No API key? Every feature works with the deterministic `ScriptedLLM`:

```python
from structured_output_agent import ScriptedLLM

llm = ScriptedLLM(['{"name": "Ada"}', '{"name": "Ada", "age": 36}'])
result = StructuredOutputAgent(llm).run("Ada is 36", Person)
assert result.attempts == 2               # first answer was missing "age"
print(result.failures[0].errors[0])       # age: Field required
```

Try the offline demo:

```bash
structured-agent demo
```

## Concepts

### Schema enforcement

`OutputContract` derives the JSON Schema from your model and builds the system
prompt. Strict mode rejects fields the schema does not declare:

```python
from structured_output_agent import OutputContract

contract = OutputContract(Person, strict=True)
contract.json_schema()["additionalProperties"]  # False
contract.system_prompt("You extract people.")    # your text + schema instructions

agent = StructuredOutputAgent(llm, strict=True)  # or agent.run(..., strict=True)
```

### Tolerant parsing, strict validation

`extract_json` accepts bare JSON, fenced code blocks and JSON surrounded by
prose. Validation is always strict Pydantic validation, and errors are itemised:

```python
from structured_output_agent import extract_json, parse_output

extract_json('Sure! Here it is: {"a": 1} Hope that helps.')  # {'a': 1}
parse_output('{"age": "old"}', OutputContract(Person))
# SchemaValidationError: name: Field required; age: Input should be a valid integer...
```

### Retry on parse and validation errors

```python
from structured_output_agent import RetryPolicy

agent = StructuredOutputAgent(
    llm,
    retry=RetryPolicy(max_attempts=4, initial_delay=0.5, multiplier=2, max_delay=4),
)
```

A failed attempt is appended to the conversation as the assistant turn, followed
by a repair request listing each error (`loc: msg`). When every attempt fails,
`RetryExhaustedError.attempts` holds each raw output and its errors.

### Validated tools

```python
from pydantic import BaseModel

from structured_output_agent import ToolRegistry


class Weather(BaseModel):
    temp_c: float
    conditions: str


tools = ToolRegistry()


@tools.register
def get_weather(city: str) -> Weather:
    """Current weather for a city."""
    return Weather(temp_c=18.5, conditions="light rain")


agent = StructuredOutputAgent(llm, tools=tools, max_tool_rounds=5)
```

The argument schema is generated from the signature (or pass `args_model=`), and
the response model comes from the return annotation (or `response_model=`).
Unknown tools, invalid arguments, exceptions and non-conforming responses are
returned to the model as error results, so it can recover, and are logged.

### Validation failure logging

```python
from structured_output_agent import ValidationLogger

logger = ValidationLogger(path="logs/failures.jsonl", max_raw_chars=2000)
agent = StructuredOutputAgent(llm, logger=logger)
...
logger.events          # list[FailureEvent]
logger.stats.to_dict() # runs, successes, attempts, failures_by_kind, success rates
```

Each failure is also emitted on the `structured_output_agent` logger (warning),
and retry exhaustion at error level. A JSONL line looks like:

```json
{"timestamp": "2026-10-19T12:00:00+00:00", "kind": "validation_error", "contract": "Person",
 "attempt": 1, "errors": [{"loc": "age", "msg": "Field required", "type": "missing", "input": {"name": "Ada"}}],
 "raw": "{\"name\": \"Ada\"}", "truncated": false, "run_id": "3f2a..."}
```

### Bring your own LLM

Implement one method:

```python
from collections.abc import Sequence

from structured_output_agent import LLMResponse, Message, ToolSpec


class MyClient:
    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> LLMResponse:
        ...
```

`AnthropicClient` defaults to `claude-opus-5-5`, replays assistant turns with
their original content blocks, raises `LLMError` on refusals, and enables
server-side refusal fallbacks (`fallbacks=None` turns them off).

## CLI

```bash
structured-agent demo [--log-file failures.jsonl]     # offline walkthrough
structured-agent schema mypkg.models:Person --strict  # print the JSON Schema
structured-agent run mypkg.models:Person "Ada is 36"  # call Claude (anthropic extra)
```

The Docker image runs the same CLI:

```bash
docker run --rm ghcr.io/imranmali/structured-output-agent:latest demo
```

## Examples

- [`examples/extract_invoice.py`](examples/extract_invoice.py): nested models,
  dates and constraints repaired after a validation error (`--live` to use Claude).
- [`examples/tool_validation.py`](examples/tool_validation.py): tool arguments
  rejected before execution, then a validated tool response.

## Project layout

```text
src/structured_output_agent/
  types.py           shared dataclasses (Message, ToolCall, ErrorDetail, RunResult, ...)
  errors.py          exception hierarchy
  llm.py             LLMClient protocol, ScriptedLLM, AnthropicClient
  schema.py          OutputContract: JSON Schema, prompt, strict validation
  parsing.py         JSON extraction and itemised validation errors
  retry.py           RetryPolicy and repair prompts
  tools.py           ToolRegistry with argument and response validation
  validation_log.py  failure events, JSONL sink, reliability stats
  agent.py           StructuredOutputAgent loop
  cli.py             structured-agent CLI
openspec/            specs and change history (OpenSpec)
```

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'

ruff check . && ruff format --check .
mypy
pytest --cov
npx @fission-ai/openspec@1.14.3 validate --all --strict
```

This project is planned with [OpenSpec](https://github.com/Fission-AI/OpenSpec):
behaviour lives in `openspec/specs/`, changes are proposed in
`openspec/changes/`, implemented, then archived. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the branching model, commit conventions
and release process.

## CI/CD

- **CI** (`.github/workflows/ci.yml`, every PR and push to `main`): ruff lint and
  format check, mypy strict, pytest on Python 3.10–3.13 with a 90% coverage gate,
  `openspec validate --all --strict`, wheel/sdist build with an installed smoke
  test, and a Docker build that runs the demo.
- **CD** (`.github/workflows/cd.yml`): pushes to `main` publish a container image
  to GHCR; `vX.Y.Z` tags also build the distributions and attach them to a GitHub
  Release.

## License

[MIT](LICENSE)
