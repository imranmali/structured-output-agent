# Spec Delta

## Purpose

Makes cost and latency visible per run and lets operators plug metrics and
tracing into the run lifecycle without forking the library.

## ADDED Requirements

### Requirement: Token usage per run
The system SHALL sum the input and output token counts reported by the
provider across every model call in a run and return the totals with the run
result.

#### Scenario: Usage across a repaired run
- **WHEN** a run makes two model calls reporting 100/20 and 120/25 input/output tokens
- **THEN** the run result reports 220 input tokens and 45 output tokens

#### Scenario: Provider reports no usage
- **WHEN** the provider reports no token counts
- **THEN** the run result reports zero tokens rather than failing

### Requirement: Run identity and duration
The system SHALL assign every run a unique run identifier, use it on every
failure event of that run, and return it together with the run's wall-clock
duration on the result.

#### Scenario: Correlating events
- **WHEN** a run fails validation once and then succeeds
- **THEN** the recorded failure event and the run result carry the same run identifier and the result reports a positive duration

### Requirement: Lifecycle hooks
The system SHALL notify registered hooks when a run starts, before and after
each model call (with latency and usage), after each tool call, on each
recorded failure and when a run ends (with outcome, attempts, usage and
duration).

#### Scenario: Hook sees the run end
- **WHEN** a hook is registered and a run succeeds on its second attempt
- **THEN** the hook receives one run-start and one run-end notification, two model-call notifications and one failure notification

### Requirement: Hook failures are isolated
The system SHALL log and otherwise ignore an exception raised by a hook, so a
faulty hook never changes the outcome of a run.

#### Scenario: Broken hook
- **WHEN** a hook raises an exception on every notification
- **THEN** the run still returns its validated result and the exception is logged
