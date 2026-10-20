# Tasks

## 1. Run engine and async execution (`feature/run-engine`)

- [x] 1.1 Add `Usage`, run metadata on `RunResult`, `FailureKind.PROVIDER_ERROR`, `TransientLLMError`, `RunTimeoutError`, `AsyncLLMClient` and `AsyncScriptedLLM`, and verify `mypy --strict` passes
- [x] 1.2 Extract the sans-IO `RunEngine` and rebuild `StructuredOutputAgent` on it, and verify the existing agent tests pass unchanged
- [x] 1.3 Add `AsyncStructuredOutputAgent` with concurrent tool execution, tool timeouts, run deadlines and `run_many` (both agents), and verify `tests/test_async_agent.py` covers every async-execution scenario
- [x] 1.4 Add `RunHooks` with isolated dispatch and usage/duration statistics, and verify `tests/test_hooks.py` covers every run-observability scenario

## 2. Provider resilience (`feature/provider-resilience`)

- [x] 2.1 Implement `ProviderRetryPolicy`, `RetryingLLM` and `AsyncRetryingLLM`, and verify `tests/test_resilience.py` covers every provider-resilience scenario
- [x] 2.2 Map Anthropic SDK errors to transient/permanent `LLMError`s, extract token usage, add `AsyncAnthropicClient`, and verify `tests/test_llm.py` covers the mapping and usage

## 3. Logging hardening (`feature/logging-hardening`)

- [x] 3.1 Add bounded event retention and redaction to `ValidationLogger`, and verify `tests/test_validation_log.py` covers the new validation-logging scenarios

## 4. HTTP service and deployment (`feature/http-service`)

- [x] 4.1 Implement the FastAPI service (settings, endpoints, auth, load shedding, request IDs, JSON logs, Prometheus hooks, graceful readiness) and verify `tests/test_server.py` covers every agent-service scenario
- [x] 4.2 Add `structured-agent serve`, update the Dockerfile to serve by default, and verify the container answers `/healthz` and `/readyz`
- [x] 4.3 Add Kubernetes manifests and docker-compose under `deploy/` and verify they parse (and pass `kubeconform` when available)
- [x] 4.4 Write `docs/deployment.md` (configuration reference, scaling, operations runbook) and verify every setting it lists exists in the settings model

## 5. CI/CD and supply chain (`ci/production-hardening`)

- [ ] 5.1 Extend CI with Python 3.14, all extras, `pip-audit`, CodeQL, container scan and a service smoke test, and verify the workflows pass `actionlint` and run green on the pull request
- [x] 5.2 Extend CD with multi-arch images, SBOM/provenance and optional PyPI trusted publishing, and verify `actionlint` passes
- [x] 5.3 Add SECURITY.md and CHANGELOG.md and update CONTRIBUTING.md, and verify links resolve

## 6. Documentation and integration

- [x] 6.1 Update the README (async, batch, resilience, hooks, usage, service, deployment) and verify every snippet runs
- [x] 6.2 Bump the version to 0.2.0 and run the full local gate (ruff, format, mypy, pytest with coverage, `openspec validate --all --strict`), and verify all pass

## Workflow follow-up

- Archive the change once CI is green on the pull request, then merge the pull request into `main`.
- Tag `v0.2.0` when the user wants a release.
