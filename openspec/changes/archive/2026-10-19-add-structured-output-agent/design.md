# Design

## Context

Greenfield repository. See `proposal.md` for motivation and `specs/` for the
behavioural contract. The library must run offline in CI (no API keys), so
every behaviour is exercised through a deterministic scripted LLM client, and
the real provider client is an optional extra.

## Goals / Non-Goals

**Goals:**

- A small, dependency-light library (`pydantic` only at runtime) with a typed
  public API that passes `mypy --strict`.
- Each concern lives in its own module so it can be developed and tested in
  isolation (and in parallel).
- 90%+ line coverage enforced in CI.

**Non-Goals:**

- Async API, streaming, or partial-JSON validation.
- Provider-native "JSON mode"/structured-output features; the library works
  with any text-producing model and validates on our side.

## Decisions

### Module layout (`src/` layout, package `structured_output_agent`)

| Module | Responsibility | Owner branch |
| --- | --- | --- |
| `types.py` | Shared dataclasses: `ErrorDetail`, `FailureKind`, `Attempt`, `Message`, `ToolCall`, `ToolSpec`, `LLMResponse`, `RunResult` | scaffold |
| `errors.py` | Exception hierarchy rooted at `StructuredOutputError` | scaffold |
| `llm.py` | `LLMClient` protocol, `ScriptedLLM`, `AnthropicClient` | scaffold |
| `schema.py` | `OutputContract` — JSON Schema, prompt instructions, strict/lenient validation | `feature/schema-parsing` |
| `parsing.py` | `extract_json`, `parse_output`, `errors_from_pydantic` | `feature/schema-parsing` |
| `retry.py` | `RetryPolicy`, `build_repair_message` | `feature/retry-logging` |
| `validation_log.py` | `FailureEvent`, `ReliabilityStats`, `ValidationLogger` | `feature/retry-logging` |
| `tools.py` | `Tool`, `ToolRegistry`, `ToolResult` | `feature/tool-validation` |
| `agent.py` | `StructuredOutputAgent` orchestration loop | `feature/agent-cli` |
| `cli.py` | `structured-agent` entry point (`demo`, `schema`) | `feature/agent-cli` |

Shared types and errors are written first so the feature branches can be built
concurrently against a fixed interface.

### Provider-agnostic client protocol

`LLMClient.complete(*, system, messages, tools) -> LLMResponse`. A response is
either text (a candidate answer) or a list of tool calls. Alternative
considered: depending on a framework such as LangChain — rejected to keep the
dependency surface small and the retry semantics explicit.

### Tolerant extraction, strict validation

Extraction tries, in order: whole text as JSON, fenced code blocks, then a
brace-balanced scan for the first decodable object/array. Validation is always
done by Pydantic. Strict mode is implemented by validating against a
derived subclass with `extra="forbid"` so the caller's model stays untouched,
and the advertised schema then includes `additionalProperties: false`.

### Retry as conversation repair

A failed attempt appends the model's raw answer as an assistant turn and a user
turn listing itemised errors (`loc: msg`) plus the schema reminder. This gives
the model the exact information it needs to self-correct; alternative of
re-sending the original prompt unchanged was rejected because it tends to
reproduce the same error. Tool rounds do not consume retry attempts but are
capped separately by `max_tool_rounds`.

### Tool contracts

Tools are registered via a decorator. If no argument model is given it is
generated from the function signature with `pydantic.create_model`; the
response model defaults to the return annotation when it is a Pydantic model.
Tool failures are never raised into the loop — they are returned to the model
as an error tool result so it can recover, and they are logged.

### Logging

`ValidationLogger` stores events in memory, optionally appends JSON lines to a
file, and mirrors every event to the `structured_output_agent` stdlib logger.
Raw output is truncated (default 2000 chars) to bound log size.

### CI/CD

GitHub Actions: `ci.yml` (ruff lint + format check, mypy, pytest matrix on
Python 3.10–3.13 with coverage gate, `openspec validate --all --strict`,
package build, Docker build), `cd.yml` (on `main`: build and push the image to
GHCR; on `v*` tags: build distributions and attach them to a GitHub Release).

## Risks / Trade-offs

- [Brace-scan extraction may pick the wrong object when prose contains several
  JSON snippets] → take the first decodable one and rely on schema validation +
  retry to catch mismatches.
- [Retries increase cost/latency] → bounded attempts, configurable backoff, and
  statistics that make the cost visible.
- [Signature-derived tool schemas can be lossy for exotic annotations] →
  callers can always pass an explicit argument model.

## Migration Plan

Not applicable (new project). Releases are cut by pushing a `vX.Y.Z` tag.
