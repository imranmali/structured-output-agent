"""Turn raw model text into JSON payloads and validated objects."""

from __future__ import annotations

from pydantic import ValidationError

from .types import ErrorDetail


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
