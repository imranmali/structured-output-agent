"""Request and response bodies of the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Hard ceiling on names; prompt and batch limits are configurable and are
# enforced by the endpoints so they can change without a code change.
_NAME_MAX = 200


class RunRequest(BaseModel):
    """``POST /v1/runs`` body."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=_NAME_MAX, description="Contract name")
    prompt: str = Field(min_length=1, description="User prompt")
    system_prompt: str | None = Field(default=None, description="Overrides the default")
    strict: bool | None = Field(default=None, description="Forbid unknown fields")


class BatchRequest(BaseModel):
    """``POST /v1/batches`` body."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=_NAME_MAX)
    prompts: list[str] = Field(min_length=1)
    system_prompt: str | None = None
    strict: bool | None = None


class ErrorItem(BaseModel):
    loc: str
    msg: str
    type: str


class FailureOut(BaseModel):
    attempt: int
    kind: str
    errors: list[ErrorItem]


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class RunResponse(BaseModel):
    run_id: str
    model: str
    output: dict[str, Any]
    attempts: int
    failures: list[FailureOut]
    tool_calls: int
    usage: UsageOut
    duration_ms: float


class BatchItemOk(RunResponse):
    status: Literal["ok"] = "ok"


class BatchItemError(BaseModel):
    status: Literal["error"] = "error"
    status_code: int
    error: dict[str, Any]


class BatchResponse(BaseModel):
    model: str
    results: list[BatchItemOk | BatchItemError]


class ModelInfo(BaseModel):
    name: str
    schema_: dict[str, Any] = Field(alias="schema", serialization_alias="schema")

    model_config = ConfigDict(populate_by_name=True)


class ModelList(BaseModel):
    models: list[ModelInfo]
