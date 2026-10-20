# validation-logging Specification

## Purpose
Makes reliability observable by recording every validation failure as a
structured event and summarising success and failure rates.

## Requirements

### Requirement: Structured failure events
The system SHALL record each parse, validation, and tool validation failure as
an event containing a timestamp, the failure kind, the contract or tool name,
the attempt number, the itemised errors, and a truncated copy of the raw
output.

#### Scenario: Validation failure recorded
- **WHEN** an attempt fails validation for contract `Person`
- **THEN** an event with kind `validation_error`, contract `Person`, the attempt number and the error list is recorded

#### Scenario: Raw output truncated
- **WHEN** the raw output exceeds the configured maximum length
- **THEN** the recorded raw output is truncated to that length and marked as truncated

### Requirement: JSON lines sink
The system SHALL be able to append failure events to a file as one JSON object
per line.

#### Scenario: Events written to file
- **WHEN** a file sink is configured and two failures occur
- **THEN** the file contains two lines, each a valid JSON object describing one failure

### Requirement: Standard logging integration
The system SHALL emit each failure event through the standard Python logging
framework at warning level, and final exhaustion of retries at error level.

#### Scenario: Warning emitted
- **WHEN** a validation failure is recorded
- **THEN** a warning-level log record is emitted on the library's logger

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
