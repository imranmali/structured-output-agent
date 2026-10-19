import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_invoice_example(capsys: pytest.CaptureFixture[str]) -> None:
    invoice = load("extract_invoice").main()

    assert invoice.invoice_number == "INV-1042"
    assert len(invoice.items) == 2
    out = capsys.readouterr().out
    assert "attempt 1 failed (validation_error)" in out
    assert "valid after 2 attempt(s)" in out


def test_tool_validation_example(capsys: pytest.CaptureFixture[str]) -> None:
    rec = load("tool_validation").main()

    assert rec.symbol == "ACME"
    out = capsys.readouterr().out
    assert "[tool_argument_error] get_quote" in out
    assert "2 tool call(s)" in out
