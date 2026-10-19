"""Turn raw model text into JSON payloads and validated objects.

Extraction is tolerant (bare JSON, Markdown fences, JSON embedded in prose);
validation is strict and always delegated to the ``OutputContract``.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import ParseError
from .types import ErrorDetail

if TYPE_CHECKING:
    from .schema import OutputContract

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_OPENER = re.compile(r"[{\[]")
_DECODER = json.JSONDecoder()


def errors_from_pydantic(exc: ValidationError) -> list[ErrorDetail]:
    """Convert a Pydantic ``ValidationError`` into itemised ``ErrorDetail`` objects."""
    return [
        ErrorDetail(
            loc=".".join(str(part) for part in err["loc"]) or "$",
            msg=err["msg"],
            type=err["type"],
            input=err.get("input"),
        )
        for err in exc.errors(include_url=False)
    ]


def extract_json(text: str) -> Any:
    """Extract the JSON payload from model output.

    Tries, in order: the whole text, each Markdown code fence (```` ```json ````
    or bare), then the first ``{``/``[`` position that decodes to an object or
    array. Raises ``ParseError`` describing the problem when nothing decodes.
    """
    stripped = text.strip()
    if not stripped:
        raise ParseError("Model output is empty", raw=text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for match in _FENCE.finditer(text):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    first_error: json.JSONDecodeError | None = None
    for match in _OPENER.finditer(text):
        try:
            payload, _ = _DECODER.raw_decode(text, match.start())
        except json.JSONDecodeError as exc:
            first_error = first_error or exc
            continue
        return payload
    if first_error is None:
        raise ParseError("No JSON object found in model output", raw=text)
    raise ParseError(f"Invalid JSON: {first_error}", raw=text)


def parse_output(text: str, contract: OutputContract[T]) -> T:
    """Extract JSON from ``text`` and validate it against ``contract``."""
    return contract.validate(extract_json(text), raw=text)
