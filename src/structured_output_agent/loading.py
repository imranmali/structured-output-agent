"""Import Pydantic models and other objects from ``module:attribute`` specs."""

from __future__ import annotations

import importlib
from typing import Any

from pydantic import BaseModel


def load_object(spec: str) -> Any:
    """Import ``package.module:attr.path`` and return the attribute."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"Expected 'module:ClassName', got {spec!r}")
    obj: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def load_model(spec: str) -> type[BaseModel]:
    """Import ``package.module:ClassName`` and check it is a Pydantic model."""
    obj = load_object(spec)
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise TypeError(f"{spec} is not a Pydantic model class")
    return obj
