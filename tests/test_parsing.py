import pytest
from pydantic import BaseModel, ValidationError

from structured_output_agent.errors import OutputError, ParseError, SchemaValidationError
from structured_output_agent.parsing import errors_from_pydantic, extract_json, parse_output
from structured_output_agent.schema import OutputContract
from structured_output_agent.types import ErrorDetail, FailureKind


class Person(BaseModel):
    name: str
    age: int


class Line(BaseModel):
    name: str


class Invoice(BaseModel):
    items: list[Line]


def test_errors_from_pydantic_itemises_locations_messages_and_inputs() -> None:
    with pytest.raises(ValidationError) as info:
        Invoice.model_validate({"items": [{"name": "ok"}, {"name": 5}]})
    assert errors_from_pydantic(info.value) == [
        ErrorDetail(
            loc="items.1.name",
            msg="Input should be a valid string",
            type="string_type",
            input=5,
        )
    ]


def test_errors_from_pydantic_uses_dollar_for_root() -> None:
    with pytest.raises(ValidationError) as info:
        Person.model_validate("nope")
    [detail] = errors_from_pydantic(info.value)
    assert detail.loc == "$"
    assert detail.type == "model_type"
    assert detail.input == "nope"


def test_bare_json() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_bare_json_with_surrounding_whitespace() -> None:
    assert extract_json('\n  {"a": 1}  \n') == {"a": 1}


def test_bare_scalar_is_returned_as_is() -> None:
    assert extract_json("42") == 42


def test_top_level_array() -> None:
    assert extract_json('[{"a": 1}, {"a": 2}]') == [{"a": 1}, {"a": 2}]


def test_fenced_json() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_fenced_json_with_prose_and_uppercase_tag() -> None:
    text = 'Here you go:\n```JSON\n{"a": {"b": [1, 2]}}\n```\nAnything else?'
    assert extract_json(text) == {"a": {"b": [1, 2]}}


def test_bare_fence() -> None:
    assert extract_json('Result:\n```\n{"a": 1}\n```') == {"a": 1}


def test_single_line_fence() -> None:
    assert extract_json('```{"a": 1}```') == {"a": 1}


def test_first_decodable_fence_wins() -> None:
    text = '```\nnot json\n```\nthen\n```json\n{"a": 2}\n```\nand\n```json\n{"a": 3}\n```'
    assert extract_json(text) == {"a": 2}


def test_json_surrounded_by_prose() -> None:
    assert extract_json('Sure! Here it is: {"a": 1} Hope that helps.') == {"a": 1}


def test_prose_with_non_json_braces_before_payload() -> None:
    text = 'Use {curly} braces like this: {"a": {"b": "}"}} and done.'
    assert extract_json(text) == {"a": {"b": "}"}}


def test_first_decodable_candidate_is_taken() -> None:
    assert extract_json('See [1] and then {"a": 1}') == [1]


def test_invalid_fence_falls_back_to_scan() -> None:
    text = '```python\nprint("hi")\n```\nThe answer is {"a": 1}.'
    assert extract_json(text) == {"a": 1}


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_empty_output(text: str) -> None:
    with pytest.raises(ParseError, match="Model output is empty") as info:
        extract_json(text)
    assert info.value.raw == text


def test_no_json_present() -> None:
    with pytest.raises(ParseError) as info:
        extract_json("I cannot answer that.")
    err = info.value
    assert str(err) == "No JSON object found in model output"
    assert isinstance(err, OutputError)
    assert err.kind is FailureKind.PARSE_ERROR
    assert err.raw == "I cannot answer that."
    assert err.errors == [ErrorDetail(loc="$", msg=str(err), type="json_parse")]


def test_malformed_json_includes_decoder_message() -> None:
    with pytest.raises(ParseError) as info:
        extract_json('{"a": 1,,}')
    assert str(info.value) == (
        "Invalid JSON: Expecting property name enclosed in double quotes: line 1 column 9 (char 8)"
    )


def test_malformed_json_reports_first_failed_candidate() -> None:
    with pytest.raises(ParseError, match=r"^Invalid JSON: Expecting property name .*\(char 8\)$"):
        extract_json("Result {oops} and [also, bad]")


def test_parse_output_valid_payload() -> None:
    result = parse_output('```json\n{"name": "Ada", "age": 36}\n```', OutputContract(Person))
    assert isinstance(result, Person)
    assert result.name == "Ada"
    assert result.age == 36


def test_parse_output_itemises_wrong_type_and_missing_field() -> None:
    text = '{"age": "old"}'
    with pytest.raises(SchemaValidationError) as info:
        parse_output(text, OutputContract(Person))
    err = info.value
    assert [(e.loc, e.type) for e in err.errors] == [
        ("name", "missing"),
        ("age", "int_parsing"),
    ]
    assert err.errors[1].msg.startswith("Input should be a valid integer")
    assert err.errors[1].input == "old"
    assert err.raw == text
    assert err.payload == {"age": "old"}
    assert err.kind is FailureKind.VALIDATION_ERROR


def test_parse_output_strict_contract() -> None:
    with pytest.raises(SchemaValidationError) as info:
        parse_output(
            'Here: {"name": "a", "age": 1, "extra": true}', OutputContract(Person, strict=True)
        )
    assert [e.loc for e in info.value.errors] == ["extra"]


def test_parse_output_propagates_parse_error() -> None:
    with pytest.raises(ParseError):
        parse_output("no json here", OutputContract(Person))
