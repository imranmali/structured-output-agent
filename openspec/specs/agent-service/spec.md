# agent-service Specification

## Purpose
Exposes structured-output runs over HTTP as a stateless service that can be
scaled horizontally behind a load balancer and operated with standard probes
and metrics.

## Requirements

### Requirement: Model catalogue
The service SHALL expose the output contracts it was configured with,
including each contract's JSON Schema, and SHALL reject runs for unknown
contracts.

#### Scenario: List models
- **WHEN** the service is configured with contract `Invoice` and a client requests the model list
- **THEN** the response lists `Invoice` with its JSON Schema

#### Scenario: Unknown model
- **WHEN** a client requests a run for contract `Nope`
- **THEN** the service responds with HTTP 404 and an error naming the unknown contract

### Requirement: Run endpoint
The service SHALL accept a run request naming a contract and a prompt, and
respond with the validated output, the run identifier, attempts, failure
history, token usage and duration.

#### Scenario: Successful run
- **WHEN** a client submits a valid run request and the model answers validly
- **THEN** the service responds with HTTP 200 and the validated output object

#### Scenario: Retries exhausted
- **WHEN** every attempt of a run is invalid
- **THEN** the service responds with HTTP 422 and the itemised failure history

#### Scenario: Provider failure
- **WHEN** the model provider fails permanently or stays unavailable after retries
- **THEN** the service responds with HTTP 502

#### Scenario: Run timeout
- **WHEN** a run exceeds the configured run timeout
- **THEN** the service responds with HTTP 504

### Requirement: Request validation limits
The service SHALL reject requests with an empty prompt, a prompt longer than
the configured maximum, or a batch larger than the configured maximum, with
HTTP 422 and without calling the model.

#### Scenario: Oversized prompt
- **WHEN** the maximum prompt length is 1000 characters and a client sends 1001
- **THEN** the service responds with HTTP 422 and the model is not called

### Requirement: Batch endpoint
The service SHALL accept a batch of prompts for one contract and respond with
one result or error per prompt, in input order.

#### Scenario: Mixed batch
- **WHEN** a batch of two prompts is submitted and the second exhausts its retries
- **THEN** the response contains a success entry followed by an error entry

### Requirement: Load shedding
The service SHALL limit the number of runs in flight per replica and SHALL
respond with HTTP 503 and a `Retry-After` header when the limit is reached,
instead of queueing without bound.

#### Scenario: Replica saturated
- **WHEN** the in-flight limit is 1, one run is in progress and a second request arrives
- **THEN** the second request receives HTTP 503 with a `Retry-After` header

### Requirement: API key authentication
The service SHALL, when API keys are configured, reject run, batch and model
requests that do not present a configured key with HTTP 401, while health,
readiness and metrics endpoints stay unauthenticated.

#### Scenario: Missing key
- **WHEN** API keys are configured and a run request carries no key
- **THEN** the service responds with HTTP 401

#### Scenario: Probes stay open
- **WHEN** API keys are configured and an orchestrator calls the liveness endpoint without a key
- **THEN** the service responds with HTTP 200

### Requirement: Health and readiness probes
The service SHALL expose a liveness endpoint that succeeds while the process
is serving, and a readiness endpoint that fails while no contracts are
configured or the service is shutting down.

#### Scenario: No contracts configured
- **WHEN** the service starts with no contracts configured
- **THEN** liveness responds with HTTP 200 and readiness responds with HTTP 503

### Requirement: Prometheus metrics
The service SHALL expose Prometheus metrics covering run outcomes by contract,
attempts per run, failures by kind, model-call latency, token usage, run
duration and runs in flight.

#### Scenario: Metrics after a run
- **WHEN** one run succeeds and the metrics endpoint is scraped
- **THEN** the run counter for that contract with outcome `success` is 1

### Requirement: Request correlation
The service SHALL return a request identifier on every response, reusing a
valid identifier supplied by the client, and SHALL include it in its logs.

#### Scenario: Client-supplied request ID
- **WHEN** a client sends `X-Request-ID: abc-123`
- **THEN** the response carries `X-Request-ID: abc-123`
