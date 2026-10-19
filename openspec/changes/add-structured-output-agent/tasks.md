# Tasks

## 1. Scaffold

- [x] 1.1 Add `pyproject.toml` (hatchling, pydantic runtime dep, `anthropic` extra, dev tools, ruff/mypy/pytest config) and verify `pip install -e .[dev]` succeeds
- [x] 1.2 Add shared `types.py` and `errors.py` and verify `mypy --strict src` passes
- [x] 1.3 Add `llm.py` with the `LLMClient` protocol, `ScriptedLLM` and `AnthropicClient`, with tests verifying scripted playback, call recording and Anthropic message conversion

## 2. Schema enforcement and output parsing (`feature/schema-parsing`)

- [x] 2.1 Implement `OutputContract` (schema derivation, prompt instructions, strict/lenient validation) and verify `tests/test_schema.py` covers every schema-enforcement scenario
- [x] 2.2 Implement `extract_json`, `parse_output`, `errors_from_pydantic` and verify `tests/test_parsing.py` covers every output-parsing scenario

## 3. Retry and validation logging (`feature/retry-logging`)

- [x] 3.1 Implement `RetryPolicy` and `build_repair_message` and verify `tests/test_retry.py` covers backoff and bounds
- [x] 3.2 Implement `ValidationLogger`, `FailureEvent` and `ReliabilityStats` and verify `tests/test_validation_log.py` covers events, truncation, JSONL sink, stdlib logging and statistics

## 4. Tool validation (`feature/tool-validation`)

- [x] 4.1 Implement `Tool`, `ToolRegistry` and `ToolResult` and verify `tests/test_tools.py` covers registration, argument validation, response validation, unknown tools and tool exceptions

## 5. Agent loop and CLI (`feature/agent-cli`)

- [x] 5.1 Implement `StructuredOutputAgent.run` combining contract, parsing, retry, tools and logging, and verify `tests/test_agent.py` covers the parse-retry and agent-loop scenarios end to end with `ScriptedLLM`
- [x] 5.2 Implement the `structured-agent` CLI (`demo`, `schema`) and verify `tests/test_cli.py` passes and `structured-agent demo` prints the reliability report
- [x] 5.3 Add runnable examples under `examples/` and verify they run offline

## 6. CI/CD (`ci/github-actions`)

- [ ] 6.1 Add `.github/workflows/ci.yml` (lint, format, mypy, test matrix with coverage gate, OpenSpec validation, build, Docker build) and verify it passes on the pull request
- [ ] 6.2 Add `.github/workflows/cd.yml`, `Dockerfile` and `.dockerignore` and verify the Docker image builds and runs `structured-agent demo` locally or in CI

## 7. Documentation and licensing

- [x] 7.1 Write the README (motivation, install, quick start, tools, retry, logging, CLI, development workflow, CI/CD) and verify every code snippet matches the public API
- [x] 7.2 Confirm MIT `LICENSE` is present and referenced from `pyproject.toml` and the README

## 8. Integration

- [x] 8.1 Run the full local gate (`ruff check`, `ruff format --check`, `mypy`, `pytest --cov`, `openspec validate --all --strict`) and verify all pass

## Workflow follow-up

- Archive the change with `openspec archive add-structured-output-agent` once implementation is verified, and confirm `openspec/specs/` holds the five capabilities.
- Merge the pull request into `main` after CI is green.
