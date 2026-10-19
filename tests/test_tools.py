from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field

from structured_output_agent import (
    FailureKind,
    ToolArgumentError,
    ToolCall,
    ToolExecutionError,
    ToolRegistrationError,
    ToolResponseError,
    ToolSpec,
    UnknownToolError,
)
from structured_output_agent.tools import Tool, ToolRegistry


class Weather(BaseModel):
    temp_c: float


class Point(BaseModel):
    x: int
    y: int


class Opaque:
    pass


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="call_1", name=name, arguments=arguments)


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


def test_tool_advertised_with_schema_requiring_city(registry: ToolRegistry) -> None:
    @registry.register
    def get_weather(city: str) -> Weather:
        """Current weather for a city.

        Uses an imaginary provider.
        """
        return Weather(temp_c=21.5)

    [spec] = registry.specs()
    assert isinstance(spec, ToolSpec)
    assert spec.name == "get_weather"
    assert spec.description == "Current weather for a city."
    assert spec.input_schema["required"] == ["city"]
    assert spec.input_schema["properties"]["city"]["type"] == "string"
    assert spec.input_schema["title"] == "GetWeatherArguments"


def test_duplicate_registration_names_the_duplicate(registry: ToolRegistry) -> None:
    def get_weather(city: str) -> str:
        return city

    registry.register(get_weather)
    with pytest.raises(ToolRegistrationError, match="'get_weather' is already registered"):
        registry.register(get_weather)
    assert registry.names == ["get_weather"]


def test_invalid_arguments_are_not_executed(registry: ToolRegistry) -> None:
    executed: list[str] = []

    @registry.register
    def get_weather(city: str) -> Weather:
        executed.append(city)
        return Weather(temp_c=20)

    result = registry.execute(call("get_weather", town="Paris"))

    assert executed == []
    assert not result.ok
    assert result.call_id == "call_1"
    assert result.arguments == {"town": "Paris"}
    assert isinstance(result.error, ToolArgumentError)
    assert result.error.kind is FailureKind.TOOL_ARGUMENT_ERROR
    assert [(e.loc, e.type) for e in result.error.errors] == [("city", "missing")]
    content = json.loads(result.content)
    assert "city" in content["error"]
    assert content["details"] == [{"loc": "city", "msg": "Field required"}]


def test_unknown_tool_lists_available_tools(registry: ToolRegistry) -> None:
    registry.register(lambda city: city, name="get_weather")
    registry.register(lambda: "now", name="get_time")

    result = registry.execute(call("get_forecast"))

    assert isinstance(result.error, UnknownToolError)
    assert result.error.tool_name == "get_forecast"
    message = json.loads(result.content)["error"]
    assert "'get_forecast'" in message
    assert "Available tools: get_weather, get_time" in message


def test_unknown_tool_with_empty_registry(registry: ToolRegistry) -> None:
    result = registry.execute(call("anything"))
    assert "Available tools: none" in json.loads(result.content)["error"]


def test_non_conforming_response_is_an_error_not_the_raw_value(registry: ToolRegistry) -> None:
    @registry.register(response_model=Weather)
    def get_weather(city: str) -> dict[str, Any]:
        return {"temp": "hot"}

    result = registry.execute(call("get_weather", city="Paris"))

    assert not result.ok
    assert result.value is None
    assert isinstance(result.error, ToolResponseError)
    assert [e.loc for e in result.error.errors] == ["temp_c"]
    assert json.loads(result.content) == {
        "error": "Tool 'get_weather' returned an invalid response: temp_c: Field required",
        "details": [{"loc": "temp_c", "msg": "Field required"}],
    }


def test_tool_exception_is_returned_as_error(registry: ToolRegistry) -> None:
    @registry.register
    def get_weather(city: str) -> Weather:
        raise RuntimeError(f"provider down for {city}")

    result = registry.execute(call("get_weather", city="Paris"))

    assert isinstance(result.error, ToolExecutionError)
    assert str(result.error) == "RuntimeError: provider down for Paris"
    assert json.loads(result.content) == {
        "error": "RuntimeError: provider down for Paris",
        "details": [{"loc": "$", "msg": "RuntimeError: provider down for Paris"}],
    }


def test_successful_call_returns_validated_response(registry: ToolRegistry) -> None:
    @registry.register
    def get_weather(city: str, unit: str = "c") -> Weather:
        return Weather(temp_c=21.5 if unit == "c" else 0)

    result = registry.execute(call("get_weather", city="Paris"))

    assert result.ok
    assert result.error is None
    assert result.value == Weather(temp_c=21.5)
    assert json.loads(result.content) == {"temp_c": 21.5}
    assert result.name == "get_weather"


def test_response_model_accepts_dict_and_coerces(registry: ToolRegistry) -> None:
    @registry.register(response_model=Weather)
    def get_weather(city: str) -> Any:
        return {"temp_c": "18"}

    result = registry.execute(call("get_weather", city="Oslo"))
    assert result.value == Weather(temp_c=18.0)


def test_decorator_with_options_returns_original_function(registry: ToolRegistry) -> None:
    @registry.register(name="weather", description="Explicit description.")
    def get_weather(city: str) -> str:
        """Ignored docstring."""
        return city

    assert get_weather("Rome") == "Rome"
    tool = registry.get("weather")
    assert tool is not None
    assert tool.description == "Explicit description."
    assert tool.func is get_weather
    assert "weather" in registry
    assert "get_weather" not in registry
    assert registry.get("missing") is None


def test_direct_call_form_returns_original_function(registry: ToolRegistry) -> None:
    def ping() -> str:
        return "pong"

    assert registry.register(ping, description="Ping.") is ping
    assert len(registry) == 1
    assert registry.specs()[0].description == "Ping."


def test_docstring_description_joins_first_paragraph_lines(registry: ToolRegistry) -> None:
    @registry.register
    def lookup(key: str) -> str:
        """Look up a key
        in the store.

        Details nobody reads.
        """
        return key

    @registry.register
    def nodoc(key: str) -> str:
        return key

    assert [s.description for s in registry.specs()] == ["Look up a key in the store.", ""]


def test_explicit_args_model_is_used(registry: ToolRegistry) -> None:
    class SearchArgs(BaseModel):
        query: str = Field(min_length=1, description="Search terms")
        limit: int = 5

    @registry.register(args_model=SearchArgs)
    def search(query, limit):
        return [query] * limit

    schema = registry.specs()[0].input_schema
    assert schema["properties"]["query"]["description"] == "Search terms"

    ok = registry.execute(call("search", query="x", limit="2"))
    assert ok.ok
    assert ok.value == ["x", "x"]
    assert json.loads(ok.content) == ["x", "x"]

    bad = registry.execute(call("search", query=""))
    assert isinstance(bad.error, ToolArgumentError)
    assert bad.error.errors[0].loc == "query"


def test_return_annotation_sets_response_model(registry: ToolRegistry) -> None:
    @registry.register
    def get_weather(city: str) -> Weather:
        return Weather(temp_c=1)

    @registry.register
    def plain(city: str) -> dict[str, str]:
        return {"city": city}

    weather, other = (registry.get(n) for n in registry.names)
    assert weather is not None
    assert weather.response_model is Weather
    assert other is not None
    assert other.response_model is None


def test_nested_model_arguments_are_passed_as_instances(registry: ToolRegistry) -> None:
    received: list[Any] = []

    @registry.register
    def distance(origin: Point, target: Point, scale: float = 1.0) -> float:
        received.extend([origin, target])
        return (abs(origin.x - target.x) + abs(origin.y - target.y)) * scale

    result = registry.execute(call("distance", origin={"x": 0, "y": 0}, target={"x": 3, "y": 4}))

    assert result.ok
    assert all(isinstance(p, Point) for p in received)
    assert result.value == 7.0
    assert result.content == "7.0"
    schema = registry.specs()[0].input_schema
    assert schema["required"] == ["origin", "target"]
    assert "Point" in schema["$defs"]


def test_unannotated_parameters_accept_anything(registry: ToolRegistry) -> None:
    @registry.register
    def echo(value, times=1):
        return [value] * times

    schema = registry.specs()[0].input_schema
    assert schema["required"] == ["value"]
    assert registry.execute(call("echo", value={"a": 1}, times=2)).value == [{"a": 1}] * 2


def test_without_response_model_models_and_lists_are_serialised(registry: ToolRegistry) -> None:
    @registry.register
    def forecast(city: str) -> list[Weather]:
        return [Weather(temp_c=1), Weather(temp_c=2)]

    result = registry.execute(call("forecast", city="Paris"))
    assert result.ok
    assert json.loads(result.content) == [{"temp_c": 1.0}, {"temp_c": 2.0}]
    assert result.value == [Weather(temp_c=1), Weather(temp_c=2)]


def test_without_response_model_non_json_value_is_an_error(registry: ToolRegistry) -> None:
    @registry.register
    def make() -> object:
        return Opaque()

    result = registry.execute(call("make"))

    assert isinstance(result.error, ToolResponseError)
    assert "not JSON serialisable: Opaque" in str(result.error)
    assert result.value is None


def test_add_prebuilt_tool(registry: ToolRegistry) -> None:
    class Args(BaseModel):
        n: int

    tool = Tool(name="double", description="Double n.", func=lambda n: n * 2, args_model=Args)
    registry.add(tool)

    assert registry.get("double") is tool
    assert tool.spec() == ToolSpec(
        name="double", description="Double n.", input_schema=Args.model_json_schema()
    )
    assert registry.execute(call("double", n=4)).value == 8
    with pytest.raises(ToolRegistrationError, match="'double'"):
        registry.add(tool)


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("def f(*args: int) -> None: ...", r"\*args"),
        ("def f(**kwargs: int) -> None: ...", r"\*\*kwargs"),
        ("def f(a: int, /) -> None: ...", "positional-only"),
    ],
)
def test_unsupported_signatures_are_rejected(
    registry: ToolRegistry, source: str, kind: str
) -> None:
    namespace: dict[str, Any] = {}
    exec(source, namespace)
    with pytest.raises(ToolRegistrationError, match=f"'f' cannot use {kind}"):
        registry.register(namespace["f"])
    assert len(registry) == 0


def test_unresolvable_annotation_is_rejected(registry: ToolRegistry) -> None:
    def f(a: Missing) -> None:  # noqa: F821
        return None

    with pytest.raises(ToolRegistrationError, match="Cannot resolve type hints for tool 'f'"):
        registry.register(f)


def test_unschemable_annotation_is_rejected(registry: ToolRegistry) -> None:
    def f(a: Opaque) -> None:
        return None

    with pytest.raises(ToolRegistrationError, match="Cannot build an argument model for tool 'f'"):
        registry.register(f)


def test_callable_without_name_requires_explicit_name(registry: ToolRegistry) -> None:
    class Greeter:
        def __call__(self) -> str:
            return "hi"

    with pytest.raises(ToolRegistrationError, match="name is required"):
        registry.register(Greeter(), args_model=BaseModel, response_model=Weather)
