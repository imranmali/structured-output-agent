# Spec Delta

## Purpose

Ensures that tools invoked by the model receive well-formed arguments and
return data that matches what the rest of the system expects.

## ADDED Requirements

### Requirement: Tools declare argument and response contracts
The system SHALL let a tool be registered with a name, a description, an
argument model, and a response model, and SHALL expose each tool's argument
JSON Schema to the model.

#### Scenario: Tool advertised to the model
- **WHEN** a tool `get_weather` with argument `city: str` is registered
- **THEN** the tool definitions sent to the model include `get_weather` with a schema requiring `city`

#### Scenario: Duplicate registration
- **WHEN** a second tool with an already registered name is registered
- **THEN** registration fails with an error naming the duplicate

### Requirement: Tool arguments validated before execution
The system SHALL validate tool-call arguments against the tool's argument
model and SHALL NOT execute the tool when validation fails.

#### Scenario: Invalid arguments
- **WHEN** the model calls `get_weather` with `{"town": "Paris"}`
- **THEN** the tool function is not executed and a tool error describing the missing `city` is returned to the model

#### Scenario: Unknown tool
- **WHEN** the model calls a tool name that is not registered
- **THEN** a tool error naming the unknown tool and listing available tools is returned to the model

### Requirement: Tool responses validated after execution
The system SHALL validate the value returned by a tool against the tool's
response model and SHALL report a tool error when it does not conform.

#### Scenario: Non-conforming tool response
- **WHEN** `get_weather` returns `{"temp": "hot"}` but its response model requires `temp_c: float`
- **THEN** a tool response validation error is recorded and the model receives an error result instead of the raw value

#### Scenario: Tool raises an exception
- **WHEN** the tool function raises an exception
- **THEN** the exception message is returned to the model as a tool error and the agent continues

### Requirement: Agent loop with tools
The system SHALL run a loop in which the model may call tools any number of
times, bounded by a configurable maximum number of tool rounds, before
producing the final structured answer.

#### Scenario: Tool then answer
- **WHEN** the model calls `get_weather` once and then returns a valid final JSON answer
- **THEN** the tool result is fed back to the model and the final answer is returned as a typed object

#### Scenario: Too many tool rounds
- **WHEN** the model keeps calling tools beyond the configured maximum rounds
- **THEN** the run fails with an error stating that the tool round limit was exceeded
