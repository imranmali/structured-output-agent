import json

import pytest
from pydantic import BaseModel, ConfigDict

from structured_output_agent import schema as schema_module
from structured_output_agent.errors import SchemaValidationError
from structured_output_agent.schema import OutputContract


class Person(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    city: str


class Item(BaseModel):
    sku: str


class Order(BaseModel):
    address: Address
    items: list[Item]
    tags: dict[str, int] = {}


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str


class Stripping(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str


def test_defaults_and_custom_name() -> None:
    contract = OutputContract(Person)
    assert contract.model is Person
    assert contract.strict is False
    assert contract.name == "Person"
    assert OutputContract(Person, name="Human", strict=True).name == "Human"


def test_schema_derived_from_model() -> None:
    schema = OutputContract(Person).json_schema()
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"name", "age"}
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["age"]["type"] == "integer"
    assert "additionalProperties" not in schema


def test_strict_schema_closes_top_level_and_nested_objects() -> None:
    schema = OutputContract(Order, strict=True).json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Address"]["additionalProperties"] is False
    assert schema["$defs"]["Item"]["additionalProperties"] is False
    assert schema["properties"]["tags"]["additionalProperties"] == {"type": "integer"}


def test_strict_schema_overrides_model_that_allows_extra() -> None:
    assert OutputContract(Open).json_schema()["additionalProperties"] is True
    assert OutputContract(Open, strict=True).json_schema()["additionalProperties"] is False


def test_json_schema_returns_independent_copies() -> None:
    contract = OutputContract(Person, strict=True)
    contract.json_schema()["properties"].clear()
    assert "name" in contract.json_schema()["properties"]


def test_prompt_contains_serialized_schema_and_json_only_instruction() -> None:
    contract = OutputContract(Person)
    prompt = contract.system_prompt()
    assert json.dumps(contract.json_schema(), indent=2, sort_keys=True) in prompt
    assert "Return only JSON" in prompt
    assert "single JSON object" in prompt
    assert "code fences" in prompt
    assert "`Person`" in prompt


def test_instructions_are_deterministic() -> None:
    assert OutputContract(Order).instructions() == OutputContract(Order).instructions()


def test_strict_instructions_mention_unknown_properties() -> None:
    assert "not defined in the schema" in OutputContract(Person, strict=True).instructions()
    assert "not defined in the schema" not in OutputContract(Person).instructions()


def test_caller_system_prompt_precedes_schema_instructions() -> None:
    contract = OutputContract(Person)
    prompt = contract.system_prompt("You extract people.")
    assert prompt == f"You extract people.\n\n{contract.instructions()}"


@pytest.mark.parametrize("base", [None, ""])
def test_system_prompt_without_base_is_just_instructions(base: str | None) -> None:
    contract = OutputContract(Person)
    assert contract.system_prompt(base) == contract.instructions()


def test_valid_payload_yields_model_instance() -> None:
    result = OutputContract(Person).validate({"name": "Ada", "age": 36})
    assert isinstance(result, Person)
    assert result.name == "Ada"
    assert result.age == 36


def test_extra_field_rejected_in_strict_mode() -> None:
    payload = {"name": "a", "age": 1, "extra": True}
    with pytest.raises(SchemaValidationError) as info:
        OutputContract(Person, strict=True).validate(payload, raw="raw text")
    err = info.value
    assert [(e.loc, e.type) for e in err.errors] == [("extra", "extra_forbidden")]
    assert err.errors[0].input is True
    assert err.raw == "raw text"
    assert err.payload is payload
    assert str(err).startswith("1 validation error for Person:")
    assert "extra: Extra inputs are not permitted" in str(err)


def test_extra_field_ignored_in_lenient_mode() -> None:
    result = OutputContract(Person).validate({"name": "a", "age": 1, "extra": True})
    assert result.model_dump() == {"name": "a", "age": 1}


def test_strict_mode_rejects_nested_extra_fields() -> None:
    payload = {"address": {"city": "Paris", "zip": "75001"}, "items": [{"sku": "x", "qty": 2}]}
    with pytest.raises(SchemaValidationError) as info:
        OutputContract(Order, strict=True).validate(payload)
    assert sorted(e.loc for e in info.value.errors) == ["address.zip", "items.0.qty"]
    assert str(info.value).startswith("2 validation errors for Order:")


def test_strict_mode_returns_exact_model_without_mutating_it() -> None:
    result = OutputContract(Order, strict=True).validate(
        {"address": {"city": "Paris"}, "items": [{"sku": "x"}]}
    )
    assert type(result) is Order
    assert type(result.address) is Address
    assert result == Order(address=Address(city="Paris"), items=[Item(sku="x")])
    assert "extra" not in Order.model_config
    assert Person.model_validate({"name": "a", "age": 1, "extra": 1}).name == "a"


def test_strict_mode_overrides_model_that_allows_extra() -> None:
    with pytest.raises(SchemaValidationError):
        OutputContract(Open, strict=True).validate({"name": "a", "x": 1})
    assert OutputContract(Open).validate({"name": "a", "x": 1}).model_extra == {"x": 1}


def test_validation_error_uses_contract_name_and_root_location() -> None:
    with pytest.raises(SchemaValidationError, match="for Human") as info:
        OutputContract(Person, name="Human").validate([1, 2])
    assert info.value.errors[0].loc == "$"
    assert info.value.errors[0].type == "model_type"


def test_strict_fallback_without_runtime_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schema_module, "_RUNTIME_EXTRA", False)
    contract = OutputContract(Stripping, strict=True)
    with pytest.raises(SchemaValidationError) as info:
        contract.validate({"name": "a", "extra": 1})
    assert [e.loc for e in info.value.errors] == ["extra"]
    result = contract.validate({"name": "  a  "})
    assert isinstance(result, Stripping)
    assert result.name == "a"
    assert contract._forbidding() is contract._forbidding()
    assert Stripping.model_config == {"str_strip_whitespace": True}
