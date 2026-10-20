# Spec Delta

## Purpose

Lets one process serve many structured-output runs concurrently without
changing the guarantees a single synchronous run provides.

## ADDED Requirements

### Requirement: Async runs with identical semantics
The system SHALL provide an asynchronous run entry point that applies the same
schema enforcement, parsing, retry, tool validation and logging behaviour as
the synchronous entry point.

#### Scenario: Async repair after invalid output
- **WHEN** an async run receives `{"name": "Ada"}` and then `{"name": "Ada", "age": 36}` for a contract requiring `name` and `age`
- **THEN** the run succeeds after 2 attempts with the same failure history a synchronous run would report

### Requirement: Concurrent tool execution
The system SHALL execute all tool calls requested in one model turn
concurrently in async runs and SHALL return their results to the model in the
order the calls were requested.

#### Scenario: Two slow tools
- **WHEN** the model requests two tool calls that each take 0.2 seconds
- **THEN** both results are available after roughly 0.2 seconds rather than 0.4 seconds
- **AND** the tool results are sent back in request order

### Requirement: Tool timeouts
The system SHALL fail a tool call that exceeds its configured timeout with a
tool execution error returned to the model, without failing the run.

#### Scenario: Tool exceeds timeout
- **WHEN** a tool with a 0.1 second timeout takes 1 second
- **THEN** the model receives a tool error stating that the tool timed out and the run continues

### Requirement: Bounded-concurrency batch runs
The system SHALL run a batch of prompts with at most a configured number of
runs in flight, return results in input order, and report a failed item as an
error in its position without aborting the other items.

#### Scenario: Batch with a failing item
- **WHEN** three prompts are run with a concurrency of 2 and the second exhausts its retries
- **THEN** no more than 2 runs are in flight at any time
- **AND** the result list has three entries in input order with the second being the retry-exhausted error

### Requirement: Run deadline
The system SHALL fail a run that exceeds its configured timeout with a run
timeout error and SHALL record the run as failed.

#### Scenario: Slow model
- **WHEN** a run has a 0.1 second timeout and the model takes 1 second to answer
- **THEN** the run fails with a run timeout error and statistics count one failed run
