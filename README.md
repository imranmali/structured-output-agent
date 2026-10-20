# Structured Output Agent

[![CI](https://github.com/imranmali/structured-output-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/imranmali/structured-output-agent/actions/workflows/ci.yml)
[![CD](https://github.com/imranmali/structured-output-agent/actions/workflows/cd.yml/badge.svg)](https://github.com/imranmali/structured-output-agent/actions/workflows/cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue)

**Make LLMs reliable, not random.** Structured Output Agent wraps any chat LLM so
that every answer is a validated Pydantic object, or a precise, logged error.

- **Enforce Pydantic JSON schemas**: the model is shown your schema on every request.
- **Validate tool responses**: tool arguments are checked before a tool runs and
  tool results are checked before the model sees them.
- **Retry on parse errors**: bad output goes back to the model with itemised
  errors, under a bounded retry policy with exponential backoff.
- **Log validation failures**: every failure becomes a structured event (stdlib
  logging and JSON lines), with success-rate statistics.
- **Production ready**: async agent with concurrent tool calls, bounded batches,
  run deadlines, jittered retries for provider outages, token usage and hooks,
  log redaction, and an HTTP service with probes, Prometheus metrics, auth and
  load shedding that scales horizontally on Kubernetes.

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
# with the HTTP service
pip install "structured-output-agent[anthropic,server] @ git+https://github.com/imranmali/structured-output-agent"
```

Requires Python 3.10+. The only runtime dependency of the library is
`pydantic>=2`; the `anthropic` and `server` extras are optional.

## Quick start

```python
from pydantic import BaseModel

from structured_output_agent import AnthropicClient, StructuredOutputAgent


class Person(BaseModel):
    name: str
    age: int


agent = StructuredOutputAgent(AnthropicClient())  # uses ANTHROPIC_API_KEY
result = agent.run("Ada Lovelace was 36 when she died.", Person)

print(result.output)  # name='Ada Lovelace' age=36
print(result.attempts)  # 1, or more if the model needed repairing
```

No API key? Every feature works with the deterministic `ScriptedLLM`:

```python
from structured_output_agent import ScriptedLLM

llm = ScriptedLLM(['{"name": "Ada"}', '{"name": "Ada", "age": 36}'])
result = StructuredOutputAgent(llm).run("Ada is 36", Person)
assert result.attempts == 2  # first answer was missing "age"
print(result.failures[0].errors[0])  # age: Field required
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
contract.system_prompt("You extract people.")  # your text + schema instructions

agent = StructuredOutputAgent(llm, strict=True)  # or agent.run(..., strict=True)
```

### Tolerant parsing, strict validation

`extract_json` accepts bare JSON, fenced code blocks and JSON surrounded by
prose. Validation is always strict Pydantic validation, and errors are itemised:

```python
from structured_output_agent import extract_json, parse_output

extract_json('Sure! Here it is: {"a": 1} Hope that helps.')  # {'a': 1}
parse_output('{"age": "old"}', OutputContract(Person))
# raises SchemaValidationError: 2 validation errors for Person: name: Field required; age: ...
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
logger.events  # most recent FailureEvents (bounded, see below)
logger.stats.to_dict()  # runs, successes, attempts, failures_by_kind, success rates
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
    ) -> LLMResponse: ...
```

`AnthropicClient` defaults to `claude-opus-5-5`, replays assistant turns with
their original content blocks, raises `LLMError` on refusals, and enables
server-side refusal fallbacks (`fallbacks=None` turns them off).

## Production use

### Async runs, concurrent tools and batches

`AsyncStructuredOutputAgent` has the same semantics as the sync agent (both
are thin drivers over one `RunEngine`), but awaits the model, runs all tool
calls of a turn concurrently and bounds whole runs with a deadline:

```python
import asyncio

from structured_output_agent import AsyncScriptedLLM, AsyncStructuredOutputAgent, RunResult

llm = AsyncScriptedLLM(['{"name": "Ada", "age": 36}'] * 4, delay=0.05)
agent = AsyncStructuredOutputAgent(llm, run_timeout=30)


async def main() -> None:
    one = await agent.run("Ada is 36", Person)
    many = await agent.run_many(["a", "b", "c"], Person, concurrency=8)
    ok = [r.output for r in many if isinstance(r, RunResult)]  # failures come back in place
    print(one.output, len(ok))


asyncio.run(main())
```

- Coroutine tools are awaited and sync tools run in worker threads; set
  `@tools.register(timeout=5)` to fail slow tools fast (the model gets a
  tool error and the run continues).
- `run_many` keeps input order and returns a failed item as its exception.
  The sync agent has `run_many(..., max_workers=8)` on a thread pool.
- `run_timeout` raises `RunTimeoutError`; cancelled runs are recorded as failed.

### Provider resilience

Rate limits, overload, 5xx, timeouts and connection errors are transient and
retried with capped exponential backoff, full jitter and `Retry-After`;
authentication errors, bad requests and refusals fail immediately:

```python
from structured_output_agent import (
    AsyncAnthropicClient,
    AsyncRetryingLLM,
    ProviderRetryPolicy,
)

llm = AsyncRetryingLLM(
    AsyncAnthropicClient(timeout=60, max_retries=0),  # let the wrapper own retries
    ProviderRetryPolicy(max_retries=4, initial_delay=0.5, max_delay=20),
)
agent = AsyncStructuredOutputAgent(llm)
```

`RetryingLLM` wraps sync clients the same way. Any client can opt in by raising
`TransientLLMError(retry_after=...)` for retryable failures. A provider failure
that ends a run is logged as a `provider_error` event.

### Usage, run IDs and hooks

Every `RunResult` carries `run_id`, `usage` (summed input/output tokens) and
`duration_s`; `ValidationLogger.stats` aggregates tokens and durations. Plug in
metrics or tracing with `RunHooks`; a hook that raises is logged and ignored:

```python
from structured_output_agent import RunHooks, Usage


class CostHook(RunHooks):
    def on_run_end(
        self,
        *,
        run_id: str,
        contract: str,
        success: bool,
        attempts: int,
        usage: Usage,
        duration_s: float,
        error: BaseException | None,
    ) -> None:
        print(f"{contract} {run_id}: {usage.total_tokens} tokens in {duration_s:.2f}s")


agent = StructuredOutputAgent(llm_sync, hooks=[CostHook()])
```

Other notifications: `on_run_start`, `on_llm_request`, `on_llm_response`
(latency and usage per call), `on_tool_result` and `on_failure`.

### Logging safely

```python
from structured_output_agent import ValidationLogger
from structured_output_agent.validation_log import EMAIL_PATTERN, redact_patterns

logger = ValidationLogger(
    max_events=1000,  # in-memory retention; statistics still count everything
    redact=redact_patterns(EMAIL_PATTERN, r"\b\d{16}\b"),  # scrub before storing/logging
)
```

Redaction is applied to raw output, error messages and offending inputs before
an event is stored, written to JSONL or logged. If the redactor itself fails,
the text is replaced with `[REDACTION FAILED]` rather than leaked.

## HTTP service

`structured-agent serve` (or the container image, which serves by default)
exposes your Pydantic models over HTTP as a stateless service that scales by
adding replicas:

```bash
docker run --rm -p 8000:8000 \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e SOA_API_KEYS="$(openssl rand -hex 32)" \
  -e SOA_MODELS=structured_output_agent.cli:WeatherReport \
  ghcr.io/imranmali/structured-output-agent:latest

curl -sS localhost:8000/v1/runs -H "Authorization: Bearer $SOA_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "WeatherReport", "prompt": "Weather report for Paris"}'
```

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/runs` | one run: validated `output`, `run_id`, `attempts`, `failures`, `usage`, `duration_ms` |
| `POST /v1/batches` | many prompts for one contract, bounded concurrency, results in order |
| `GET /v1/models`, `GET /v1/models/{name}` | served contracts and their JSON Schemas |
| `GET /healthz`, `GET /readyz` | liveness and readiness probes (never authenticated) |
| `GET /metrics` | Prometheus metrics: runs, attempts, failures by kind, latency, tokens, in-flight |

Production behaviour:

- **Errors mapped to status codes:** 401 auth, 404 unknown contract, 422 invalid
  request or retries exhausted (with the failure history), 502 provider error,
  503 overloaded (with `Retry-After`), 504 run timeout.
- **Load shedding:** each replica caps in-flight runs (`SOA_MAX_IN_FLIGHT`) and
  sheds excess load immediately instead of queueing it; the HPA adds replicas.
- **Credentials:** API keys (`SOA_API_KEYS`) are compared in constant time, and
  the LLM client is created lazily so probes work before credentials resolve.
- **Logs:** JSON logs carry request IDs. Prompts and outputs are never logged,
  and emails are redacted from the failure log.
- **Shutdown:** readiness fails during shutdown so in-flight requests drain.

Deployment assets: [`deploy/kubernetes/`](deploy/kubernetes) (Deployment with
hardened security context, Service, HPA, PDB, ConfigMap; `kubectl apply -k
deploy/kubernetes`) and [`deploy/docker-compose.yml`](deploy/docker-compose.yml)
(with optional Prometheus). The configuration reference, API reference,
scaling guidance and operations runbook are in
[docs/deployment.md](docs/deployment.md).

## CLI

```bash
structured-agent demo [--log-file failures.jsonl]     # offline walkthrough
structured-agent schema mypkg.models:Person --strict  # print the JSON Schema
structured-agent run mypkg.models:Person "Ada is 36"  # call Claude (anthropic extra)
structured-agent serve --host 0.0.0.0 --port 8000     # HTTP service (server extra)
```

The Docker image runs the same CLI and serves the API by default:

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
  llm.py             LLMClient/AsyncLLMClient protocols, scripted and Anthropic clients
  resilience.py      ProviderRetryPolicy, RetryingLLM, AsyncRetryingLLM
  schema.py          OutputContract: JSON Schema, prompt, strict validation
  parsing.py         JSON extraction and itemised validation errors
  retry.py           RetryPolicy and repair prompts
  tools.py           ToolRegistry with argument and response validation
  validation_log.py  failure events (bounded, redacted), JSONL sink, reliability stats
  engine.py          sans-IO RunEngine shared by both agents
  agent.py           StructuredOutputAgent and AsyncStructuredOutputAgent
  hooks.py           RunHooks lifecycle notifications
  loading.py         module:attr loader
  cli.py             structured-agent CLI
  server/            FastAPI service: settings, app, metrics, logging, runner
deploy/              Kubernetes manifests, docker-compose, Prometheus config
docs/deployment.md   configuration, API, scaling and operations guide
openspec/            specs and change history (OpenSpec)
```

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,server,anthropic]'

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
  format check, mypy strict, pytest on Python 3.10–3.14 with all extras and a
  90% coverage gate, `openspec validate --all --strict`, `pip-audit`, wheel/sdist
  build with an installed smoke test, and a Docker build that runs the demo,
  smoke-tests the HTTP service and scans the image with Trivy.
- **CodeQL** (`.github/workflows/codeql.yml`): Python and workflow analysis on
  PRs, `main` and weekly.
- **CD** (`.github/workflows/cd.yml`): pushes to `main` publish a multi-arch
  (amd64/arm64) image to GHCR with SBOM and provenance; `vX.Y.Z` tags also
  build the distributions, attach them to a GitHub Release and, when the
  repository variable `PUBLISH_TO_PYPI` is `true`, publish to PyPI with trusted
  publishing.

See [CHANGELOG.md](CHANGELOG.md) and [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
