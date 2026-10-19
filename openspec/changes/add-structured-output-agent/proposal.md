# Proposal

## Why

LLM output is free-form text, so applications that need typed data get broken
JSON, missing fields, prose wrapped around payloads, and tool results that do
not match what the caller expects. This project shows that LLM calls can be
made reliable rather than random: every answer is checked against a declared
schema, failures are fed back to the model and retried, and every failure is
recorded so reliability can be measured.

## What Changes

- New Python library `structured_output_agent` that wraps any chat-style LLM
  behind a provider-agnostic client interface.
- Output contracts are declared as Pydantic models; their JSON Schema is
  injected into the prompt and every response is validated against it.
- Tolerant extraction of JSON from raw model text (code fences, surrounding
  prose) followed by strict validation.
- Bounded retry loop that sends the exact parse/validation errors back to the
  model as a repair request, with configurable attempts and backoff.
- Tool registry where each tool declares Pydantic argument and response
  models; tool-call arguments and tool responses are validated before use.
- Structured (JSON lines) logging of every validation failure plus in-memory
  reliability statistics.
- Built-in Anthropic client (optional dependency) and a deterministic scripted
  client for tests and offline demos.
- A `structured-agent` CLI with an offline demo that shows malformed outputs
  being repaired.
- Repository tooling: test suite, GitHub Actions CI (lint, type-check, tests,
  spec validation, build) and CD (container image and release artifacts),
  README, and MIT license.

## Capabilities

### New Capabilities

- `schema-enforcement`: declaring an output contract and instructing the model
  to follow its JSON Schema.
- `output-parsing`: extracting a JSON payload from raw model text and
  validating it into a typed object with precise error reporting.
- `parse-retry`: retrying failed generations with error feedback under a
  bounded, configurable policy.
- `tool-validation`: validating tool-call arguments and tool responses
  against declared models.
- `validation-logging`: recording validation failures as structured events and
  exposing reliability statistics.

### Modified Capabilities

None — this is a new project.

## Impact

- New package under `src/structured_output_agent/`, tests under `tests/`.
- Runtime dependency: `pydantic>=2`. Optional dependency: `anthropic`.
- New CI/CD workflows under `.github/workflows/` and a `Dockerfile`.
- Out of scope: streaming partial validation, non-JSON output formats, and
  hosting the agent as a network service.
