# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-20

Production hardening (OpenSpec change `harden-for-production`). Backward
compatible for library users; see "Changed" for the container default.

### Added

- Async agent (`AsyncStructuredOutputAgent`) with the same contract as the
  sync agent, plus batch runs (`run_many`) with bounded concurrency for both.
- Concurrent execution of the tool calls in one turn, with per-tool timeouts;
  sync tools run in worker threads, coroutine tools are awaited.
- Run deadlines: a whole run fails fast with a timeout error instead of
  hanging on a slow provider or tool.
- Provider resilience: retries on transient provider errors with capped
  exponential backoff, full jitter and `Retry-After` support, and an async
  Anthropic client.
- Token usage accounting, run ids and lifecycle hooks for metrics and tracing
  integrations; a failing hook never fails a run.
- Validation logging hardening: bounded in-memory retention (`max_events`)
  and a redaction function (`ValidationLogger(redact=...)`) applied to raw
  output before it is stored or emitted.
- HTTP service (`structured-agent serve`, `server` extra): runs and batches
  over HTTP, model listing, liveness and readiness probes, Prometheus
  metrics, API-key authentication (`SOA_API_KEYS`) and load shedding with
  `503` + `Retry-After`.
- Deployment assets: Kubernetes manifests (Deployment, Service, HPA, PDB,
  ConfigMap, example Secret) and a Docker Compose file.
- Multi-arch (`linux/amd64`, `linux/arm64`) container images with SBOM and
  signed build-provenance attestations.
- Supply-chain checks in CI: CodeQL (Python and workflows), `pip-audit` on
  the runtime dependency set and a Trivy scan of the image; Dependabot now
  tracks the Docker base image.
- Optional PyPI publishing with trusted publishing (no API token), enabled
  by the `PUBLISH_TO_PYPI` repository variable.
- Python 3.14 support, tested in CI alongside 3.10-3.13.
- `SECURITY.md`, `CHANGELOG.md` and `CODEOWNERS`.

### Changed

- The container image installs the `anthropic` and `server` extras, exposes
  port 8000 and runs `serve` by default. Run `docker run IMAGE demo` for the
  demo that used to be reached through `--help`.
- The package version is single-sourced from
  `structured_output_agent.__version__`; the release workflow checks the tag
  against it.
- CI installs all extras so the service and Anthropic client are linted,
  type-checked and tested.

## [0.1.0] - 2026-10-19

Initial release (OpenSpec change `add-structured-output-agent`).

### Added

- Schema enforcement: `OutputContract` turns a Pydantic model into the JSON
  schema sent to the model and validates the reply, with an optional strict
  mode.
- Tolerant JSON parsing: extraction from fenced code blocks and surrounding
  prose, and itemised validation errors.
- Retry on parse and validation errors via `RetryPolicy`, with capped
  exponential backoff and repair prompts that quote the errors back to the
  model.
- Tool validation: a tool registry that validates tool-call arguments and
  tool responses against Pydantic models.
- Validation failure logging: structured failure events and reliability
  statistics.
- `StructuredOutputAgent` run loop, an Anthropic client, scripted test
  clients and examples.
- `structured-agent` CLI with an offline `demo`.
- CI/CD: lint, type-check, tests with a coverage gate, OpenSpec validation,
  package and Docker builds; container images on GHCR and GitHub Releases on
  version tags.

[Unreleased]: https://github.com/imranmali/structured-output-agent/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/imranmali/structured-output-agent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/imranmali/structured-output-agent/releases/tag/v0.1.0
