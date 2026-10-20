import argparse
import io
import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from structured_output_agent import ScriptedLLM
from structured_output_agent.cli import build_parser, cmd_run, load_model, main


class Pet(BaseModel):
    name: str
    legs: int


def test_demo_prints_report(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"

    assert main(["demo", "--log-file", str(log_file)]) == 0

    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "parse_error" in out
    assert "validation_error" in out
    assert "tool_argument_error" in out
    assert "OK after 3 attempt(s), 2 tool call(s)" in out
    assert "OK after 1 attempt(s)" in out
    assert '"first_attempt_success_rate": 0.5' in out
    assert len(log_file.read_text().splitlines()) == 3


def test_schema_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema", "tests.test_cli:Pet", "--strict"]) == 0

    schema = json.loads(capsys.readouterr().out)
    assert schema["required"] == ["name", "legs"]
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("no_colon", "module:ClassName"),
        ("json:dumps", "not a Pydantic model"),
        ("tests.test_cli:Missing", "Missing"),
        ("not_a_module_xyz:Thing", "not_a_module_xyz"),
    ],
)
def test_schema_command_errors(spec: str, message: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema", spec]) == 1
    assert message in capsys.readouterr().err


def test_load_model_supports_nested_attributes() -> None:
    assert load_model("structured_output_agent.cli:WeatherReport").__name__ == "WeatherReport"


def test_run_command_with_injected_llm() -> None:
    args = build_parser().parse_args(["run", "tests.test_cli:Pet", "a dog", "--max-attempts", "2"])
    llm = ScriptedLLM(["not json", '{"name": "Rex", "legs": 4}'])
    out = io.StringIO()

    assert cmd_run(args, out, llm=llm) == 0
    assert json.loads(out.getvalue()) == {"name": "Rex", "legs": 4}


def test_run_command_reports_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    import structured_output_agent.cli as cli

    monkeypatch.setattr(cli, "AnthropicClient", lambda model: ScriptedLLM(["bad"]))
    err = io.StringIO()
    monkeypatch.setattr(cli.sys, "stderr", err)

    assert main(["run", "tests.test_cli:Pet", "x", "--max-attempts", "1"]) == 1
    assert "No valid Pet" in err.getvalue()


def test_parser_requires_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    assert isinstance(build_parser(), argparse.ArgumentParser)


def test_serve_runs_the_server_with_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    import structured_output_agent.server.runner as runner

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "run", lambda **kwargs: calls.append(kwargs))

    assert main(["serve", "--host", "127.0.0.1", "--port", "9001"]) == 0
    assert main(["serve"]) == 0
    assert calls == [{"host": "127.0.0.1", "port": 9001}, {"host": "0.0.0.0", "port": 8000}]


@pytest.mark.parametrize("port", ["0", "70000", "http"])
def test_serve_rejects_invalid_ports(port: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--port", port])


def test_serve_without_the_server_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "structured_output_agent.server.runner", None)

    assert main(["serve"]) == 1
    assert "pip install 'structured-output-agent[server]'" in capsys.readouterr().err


def test_serve_reports_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setenv("SOA_MAX_IN_FLIGHT", "zero")

    assert main(["serve"]) == 1
    assert "SOA_MAX_IN_FLIGHT" in capsys.readouterr().err
