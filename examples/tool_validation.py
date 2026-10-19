"""Validate tool arguments and tool responses in both directions.

The scripted model first calls the tool with a bad argument; the call is
rejected before the tool runs and the error goes back to the model. The tool
itself returns a dict that is validated against its response model.
"""

from __future__ import annotations

from pydantic import BaseModel

from structured_output_agent import LLMResponse, ScriptedLLM, ToolCall
from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.tools import ToolRegistry
from structured_output_agent.validation_log import ValidationLogger


class StockQuote(BaseModel):
    symbol: str
    price: float
    currency: str


class Recommendation(BaseModel):
    symbol: str
    action: str
    reason: str


tools = ToolRegistry()


@tools.register(response_model=StockQuote)
def get_quote(symbol: str) -> dict[str, object]:
    """Return the latest quote for a ticker symbol."""
    return {"symbol": symbol.upper(), "price": 187.2, "currency": "USD"}


SCRIPTED = [
    LLMResponse(text="", tool_calls=[ToolCall(id="1", name="get_quote", arguments={"ticker": "acme"})]),
    LLMResponse(text="", tool_calls=[ToolCall(id="2", name="get_quote", arguments={"symbol": "acme"})]),
    '{"symbol": "ACME", "action": "hold", "reason": "Price is stable at 187.20 USD."}',
]


def main() -> Recommendation:
    logger = ValidationLogger()
    agent = StructuredOutputAgent(ScriptedLLM(SCRIPTED), tools=tools, logger=logger)

    result = agent.run("Should I buy ACME?", Recommendation)

    for event in logger.events:
        print(f"[{event.kind.value}] {event.contract}: {[str(e) for e in event.errors]}")
    print(f"{result.tool_calls} tool call(s); answer: {result.output.model_dump_json()}")
    return result.output


if __name__ == "__main__":
    main()
