"""Extract a typed invoice from free text.

Runs offline with a scripted model by default. Pass ``--live`` to call Claude
instead (requires ``pip install 'structured-output-agent[anthropic]'`` and an
``ANTHROPIC_API_KEY``).
"""

from __future__ import annotations

import sys
from datetime import date

from pydantic import BaseModel, Field

from structured_output_agent import AnthropicClient, LLMClient, ScriptedLLM
from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.validation_log import ValidationLogger


class LineItem(BaseModel):
    description: str
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)


class Invoice(BaseModel):
    invoice_number: str
    issued_on: date
    customer: str
    items: list[LineItem]
    total: float


TEXT = """Invoice INV-1042 issued 2026-10-01 to Acme Corp:
3 x Widget at $4.50, 1 x Gadget at $12.00. Total due $25.50."""

SCRIPTED = [
    # A typical first answer: right idea, wrong types.
    '{"invoice_number": "INV-1042", "issued_on": "Oct 1st", "customer": "Acme Corp",'
    ' "items": [{"description": "Widget", "quantity": "three", "unit_price": 4.5}],'
    ' "total": 25.5}',
    # The repaired answer after the agent sends the validation errors back.
    '{"invoice_number": "INV-1042", "issued_on": "2026-10-01", "customer": "Acme Corp",'
    ' "items": [{"description": "Widget", "quantity": 3, "unit_price": 4.5},'
    ' {"description": "Gadget", "quantity": 1, "unit_price": 12.0}], "total": 25.5}',
]


def main(live: bool = False) -> Invoice:
    llm: LLMClient = AnthropicClient() if live else ScriptedLLM(SCRIPTED)
    logger = ValidationLogger()
    agent = StructuredOutputAgent(llm, logger=logger, system_prompt="You extract invoices.")

    result = agent.run(f"Extract the invoice:\n\n{TEXT}", Invoice)

    for failure in result.failures:
        print(f"attempt {failure.number} failed ({failure.kind.value}):")
        for error in failure.errors:
            print(f"  - {error}")
    print(f"valid after {result.attempts} attempt(s):")
    print(result.output.model_dump_json(indent=2))
    return result.output


if __name__ == "__main__":
    main(live="--live" in sys.argv)
