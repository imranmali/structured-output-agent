# parse-retry Specification

## Purpose
Recovers from bad model output automatically by telling the model what it got
wrong and asking again, within limits the caller controls.

## Requirements

### Requirement: Retry with error feedback
The system SHALL, when a response fails parsing or validation, send a follow-up
request that contains the model's previous response and the itemised errors,
asking for a corrected JSON object.

#### Scenario: Repair after invalid output
- **WHEN** the first response is `{"name": "Ada"}` for a contract that also requires `age`
- **THEN** the second request includes the previous response and an error mentioning `age`
- **AND** if the second response is valid the call succeeds with the typed object

### Requirement: Bounded attempts
The system SHALL stop after a configurable maximum number of attempts and
raise an error carrying the full history of attempts and their errors.

#### Scenario: All attempts fail
- **WHEN** the maximum is 3 attempts and every response is invalid
- **THEN** exactly 3 model calls are made
- **AND** an error is raised whose attempt history contains 3 entries with their raw outputs and errors

#### Scenario: Single attempt
- **WHEN** the maximum is 1 attempt and the response is invalid
- **THEN** no retry is made and the error is raised immediately

### Requirement: Configurable backoff between attempts
The system SHALL wait between attempts according to a configurable delay that
may grow exponentially and is capped at a maximum delay.

#### Scenario: Exponential backoff
- **WHEN** the initial delay is 0.5s, the multiplier is 2 and the cap is 1.5s
- **THEN** the delays before attempts 2, 3 and 4 are 0.5s, 1.0s and 1.5s

### Requirement: Attempt metadata on success
The system SHALL report, with every successful result, how many attempts were
needed and the errors encountered on the way.

#### Scenario: Success on second attempt
- **WHEN** the first response is invalid and the second is valid
- **THEN** the result reports 2 attempts and one recorded failure
