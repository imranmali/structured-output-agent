# Proposal

## Why

Version 0.1 proves the reliability loop, but only for one synchronous caller in
one process. A production deployment needs to serve many concurrent requests,
survive provider outages and rate limits, keep memory bounded, keep sensitive
model output out of logs, expose metrics and run behind a load balancer as
horizontally scalable, stateless replicas.

## What Changes

- Async agent (`AsyncStructuredOutputAgent`) with the same semantics as the
  sync agent, sharing one transport-independent run engine so the two cannot
  drift apart.
- Parallel tool calls in one model turn execute concurrently (async tools are
  awaited, sync tools run in worker threads) with optional per-tool timeouts.
- Batch runs (`run_many`) with bounded concurrency, results in input order and
  per-item errors.
- Run deadlines: a run that exceeds its configured timeout fails with a
  dedicated error.
- Provider resilience: transient provider failures (rate limits, overload,
  5xx, timeouts, connection errors) are retried with capped, jittered
  exponential backoff honouring `Retry-After`; permanent failures are not.
  Provider errors are recorded as failures.
- Token usage and latency are tracked per run, returned on results and
  aggregated in statistics.
- Lifecycle hooks for metrics/tracing integrations; a failing hook never fails
  a run.
- Validation logging hardened: bounded in-memory event retention, a redaction
  hook for raw output and error inputs, and usage/latency statistics.
- HTTP service (optional `server` extra): run endpoint, batch endpoint, model
  catalogue, health/readiness probes, Prometheus metrics, API-key auth,
  per-replica concurrency limiting with load shedding, request IDs and JSON
  logs. `structured-agent serve` starts it.
- Deployment assets: container image serving the API by default (multi-arch,
  SBOM and provenance), Kubernetes manifests (Deployment, Service, HPA, PDB)
  and a docker-compose file.
- Supply-chain and release hygiene: dependency audit and CodeQL in CI, image
  vulnerability scan, Python 3.14 in the test matrix, single-sourced version,
  optional PyPI trusted publishing, SECURITY.md and CHANGELOG.md.

## Capabilities

### New Capabilities

- `async-execution`: async runs, concurrent tool execution, batch runs and run
  deadlines.
- `provider-resilience`: classifying and retrying transient provider failures.
- `run-observability`: token usage and latency reporting and lifecycle hooks.
- `agent-service`: the HTTP API, its operational endpoints and its overload
  behaviour.

### Modified Capabilities

- `validation-logging`: bounded retention, redaction, provider-error events and
  usage/latency statistics.

## Impact

- New modules: `engine.py`, `hooks.py`, `resilience.py`, `server/`.
  `agent.py`, `llm.py`, `tools.py`, `validation_log.py`, `types.py`,
  `errors.py` and `cli.py` change; the public sync API stays compatible.
- New optional extra `server` (`fastapi`, `uvicorn`, `prometheus-client`).
- New `deploy/` directory; CI/CD workflows, Dockerfile and docs change.
- The default container command changes from `--help` to `serve`.
- Out of scope: persistent job queues, multi-tenant quotas, streaming
  responses, providers other than Anthropic (the client protocol stays open).
