# Spec Delta

## ADDED Requirements

### Requirement: Bounded event retention
The system SHALL keep at most a configurable number of failure events in
memory, discarding the oldest first, while statistics keep counting every
failure.

#### Scenario: Retention limit reached
- **WHEN** the retention limit is 2 and three failures are recorded
- **THEN** the two most recent events are retained and statistics count three failures

### Requirement: Redaction of logged content
The system SHALL apply a configurable redaction function to raw output and to
offending input values before a failure event is stored, written or logged.

#### Scenario: Email redacted
- **WHEN** a redaction function replaces email addresses with `[REDACTED]` and the raw output contains `ada@example.com`
- **THEN** the stored event, the JSON lines file and the log record contain `[REDACTED]` and not the address

## MODIFIED Requirements

### Requirement: Reliability statistics
The system SHALL keep counts of total runs, successful runs, failed runs,
total attempts and failures by kind, and SHALL compute first-attempt success
rate and overall success rate. It SHALL also sum token usage and run duration
across runs and compute the mean run duration.

#### Scenario: Statistics after mixed runs
- **WHEN** one run succeeds on the first attempt and one succeeds on the second attempt
- **THEN** statistics report 2 runs, 2 successes, 3 attempts, a first-attempt success rate of 0.5 and an overall success rate of 1.0

#### Scenario: Usage and duration totals
- **WHEN** two runs report 100 and 50 input tokens and durations of 1.0s and 3.0s
- **THEN** statistics report 150 input tokens, a total duration of 4.0s and a mean duration of 2.0s
