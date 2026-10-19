# Spec Delta

## Purpose

Lets a caller declare the exact shape of the data they need from an LLM and
guarantees the model is told that shape on every request.

## ADDED Requirements

### Requirement: Output contract from a Pydantic model
The system SHALL accept a Pydantic model class as the output contract for a
request and derive its JSON Schema from that model.

#### Scenario: Schema derived from model
- **WHEN** a caller supplies a model with fields `name: str` and `age: int`
- **THEN** the derived JSON Schema lists `name` and `age` as required properties with types `string` and `integer`

### Requirement: Schema instructions injected into the prompt
The system SHALL include the output contract's JSON Schema and an instruction
to reply with a single JSON object conforming to it in the system prompt sent
to the model.

#### Scenario: Prompt contains schema
- **WHEN** a request is made with an output contract
- **THEN** the system prompt sent to the model contains the serialized JSON Schema and the instruction to return only JSON

#### Scenario: Caller system prompt preserved
- **WHEN** the caller supplies their own system prompt
- **THEN** the caller's text appears in the system prompt before the schema instructions

### Requirement: Strict mode rejects unknown fields
The system SHALL support a strict mode in which a payload containing fields
not declared in the output contract fails validation.

#### Scenario: Extra field in strict mode
- **WHEN** strict mode is enabled and the model returns `{"name": "a", "age": 1, "extra": true}`
- **THEN** validation fails with an error identifying `extra` as not permitted

#### Scenario: Extra field in lenient mode
- **WHEN** strict mode is disabled and the model returns the same payload
- **THEN** validation succeeds and the unknown field is ignored
