"""Output contracts: the shape a model's answer must take.

An ``OutputContract`` wraps a Pydantic model and provides its JSON Schema, the
prompt instructions that advertise it, and validation of extracted payloads.
Strict mode rejects unknown fields without changing the caller's model.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from .errors import SchemaValidationError
from .parsing import errors_from_pydantic

T = TypeVar("T", bound=BaseModel)

# Pydantic releases with a per-call ``extra`` override forbid unknown fields at
# every nesting level; older ones fall back to a top-level forbidding subclass.
_RUNTIME_EXTRA = "extra" in inspect.signature(BaseModel.model_validate).parameters


def _close_objects(node: Any) -> Any:
    """Copy a JSON Schema, adding ``additionalProperties: false`` to every object."""
    if isinstance(node, list):
        return [_close_objects(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {key: _close_objects(value) for key, value in node.items()}
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
    return out


class OutputContract(Generic[T]):
    """The required shape of a model's answer, backed by a Pydantic model."""

    def __init__(self, model: type[T], *, strict: bool = False, name: str | None = None) -> None:
        self.model = model
        self.strict = strict
        self.name = name if name is not None else model.__name__
        self._forbidding_model: type[T] | None = None

    def json_schema(self) -> dict[str, Any]:
        """Return the model's JSON Schema, closed to unknown fields in strict mode."""
        schema = self.model.model_json_schema()
        if self.strict:
            return cast(dict[str, Any], _close_objects(schema))
        return schema

    def instructions(self) -> str:
        """Return prompt text telling the model to answer with conforming JSON only."""
        lines = [
            "Return only JSON: respond with a single JSON object that conforms to the "
            f"JSON Schema for `{self.name}` below.",
            "Do not include any prose, explanations or Markdown code fences.",
        ]
        if self.strict:
            lines.append("Do not include any properties that are not defined in the schema.")
        schema = json.dumps(self.json_schema(), indent=2, sort_keys=True)
        return "\n".join([*lines, "", "JSON Schema:", schema])

    def system_prompt(self, base: str | None = None) -> str:
        """Return ``base`` (if given), a blank line, then the schema instructions."""
        if base:
            return f"{base}\n\n{self.instructions()}"
        return self.instructions()

    def validate(self, payload: Any, *, raw: str = "") -> T:
        """Validate ``payload`` and return an instance of the contract model.

        Raises ``SchemaValidationError`` with itemised errors on failure.
        """
        try:
            if not self.strict:
                return self.model.model_validate(payload)
            if _RUNTIME_EXTRA:
                return self.model.model_validate(payload, extra="forbid")
            return self._forbidding().model_validate(payload)
        except ValidationError as exc:
            errors = errors_from_pydantic(exc)
            plural = "" if len(errors) == 1 else "s"
            summary = "; ".join(str(error) for error in errors)
            raise SchemaValidationError(
                f"{len(errors)} validation error{plural} for {self.name}: {summary}",
                raw=raw,
                errors=errors,
                payload=payload,
            ) from exc

    def _forbidding(self) -> type[T]:
        if self._forbidding_model is None:
            config = ConfigDict(**self.model.model_config)
            config["extra"] = "forbid"
            namespace = {"model_config": config, "__module__": self.model.__module__}
            self._forbidding_model = cast(
                "type[T]", type(self.model.__name__, (self.model,), namespace)
            )
        return self._forbidding_model
