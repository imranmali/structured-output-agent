# Spec Delta

## Purpose

Keeps runs alive through rate limits, overload and short provider outages,
while failing fast on errors that retrying cannot fix.

## ADDED Requirements

### Requirement: Transient provider failures are retried
The system SHALL retry a provider call that fails with a transient error (rate
limited, overloaded, server error, timeout or connection failure) up to a
configurable number of retries.

#### Scenario: Rate limited then success
- **WHEN** the provider rejects the first call as rate limited and accepts the second
- **THEN** the run continues with the second response and the caller sees no error

#### Scenario: Retries exhausted
- **WHEN** the maximum is 2 retries and the provider is overloaded on every call
- **THEN** exactly 3 provider calls are made and the transient error is raised

### Requirement: Permanent provider failures are not retried
The system SHALL NOT retry provider errors classified as permanent, such as
invalid requests, authentication failures and refusals.

#### Scenario: Authentication failure
- **WHEN** the provider rejects the call because the credentials are invalid
- **THEN** exactly 1 provider call is made and the error is raised immediately

### Requirement: Jittered exponential backoff honouring Retry-After
The system SHALL wait between provider retries using capped exponential
backoff with random jitter, and SHALL wait at least the provider's
`Retry-After` value when one is given, still capped at the maximum delay.

#### Scenario: Backoff bounds
- **WHEN** the initial delay is 1s, the multiplier is 2, the cap is 5s and jitter is enabled
- **THEN** the waits before retries 1, 2 and 3 lie within [0, 1], [0, 2] and [0, 4] seconds

#### Scenario: Retry-After respected
- **WHEN** a rate-limit error carries `Retry-After: 3` and the computed backoff is 1s
- **THEN** the system waits 3 seconds before retrying

### Requirement: Provider failures are recorded
The system SHALL record a provider failure that ends a run as a failure event
of kind `provider_error` and count the run as failed.

#### Scenario: Provider down
- **WHEN** a run's provider call fails permanently
- **THEN** a `provider_error` event is recorded for the run's contract and statistics count one failed run
