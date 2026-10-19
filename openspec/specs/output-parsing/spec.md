# output-parsing Specification

## Purpose
Turns raw LLM text into a validated, typed object, or into a precise error
that explains exactly what was wrong.

## Requirements

### Requirement: JSON extraction from model text
The system SHALL extract a JSON object from model output that is bare JSON,
wrapped in a Markdown code fence, or surrounded by prose.

#### Scenario: Bare JSON
- **WHEN** the model returns `{"a": 1}`
- **THEN** the extracted payload is `{"a": 1}`

#### Scenario: Fenced JSON
- **WHEN** the model returns a ```` ```json ```` fenced block containing `{"a": 1}`
- **THEN** the extracted payload is `{"a": 1}`

#### Scenario: JSON surrounded by prose
- **WHEN** the model returns `Sure! Here it is: {"a": 1} Hope that helps.`
- **THEN** the extracted payload is `{"a": 1}`

### Requirement: Parse errors are reported
The system SHALL report a parse error, including a description of the problem,
when no syntactically valid JSON object can be extracted.

#### Scenario: No JSON present
- **WHEN** the model returns `I cannot answer that.`
- **THEN** a parse error is raised stating that no JSON object was found

#### Scenario: Malformed JSON
- **WHEN** the model returns `{"a": 1,,}`
- **THEN** a parse error is raised that includes the decoder's error message

### Requirement: Validation errors are itemised
The system SHALL report every schema violation with the field location, the
error message, and the offending input when validation of an extracted payload
fails.

#### Scenario: Wrong type and missing field
- **WHEN** the contract requires `name: str` and `age: int` and the payload is `{"age": "old"}`
- **THEN** the validation error lists one issue at location `name` (missing) and one at location `age` (not a valid integer)

### Requirement: Successful parse yields a typed object
The system SHALL return an instance of the output contract model when the
payload validates.

#### Scenario: Valid payload
- **WHEN** the payload is `{"name": "Ada", "age": 36}`
- **THEN** the result is a model instance with `name == "Ada"` and `age == 36`
