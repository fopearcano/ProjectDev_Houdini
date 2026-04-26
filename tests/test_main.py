"""Tests for :mod:`app.main`."""

from __future__ import annotations

import pytest

from app.config import Config
from app.llm.lmstudio_client import LMStudioUnreachableError
from app.main import build_parser, run


def test_build_parser_requires_command() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_joins_command_words() -> None:
    parser = build_parser()
    args = parser.parse_args(["create", "a", "sphere"])
    assert args.command == ["create", "a", "sphere"]


def test_build_parser_supports_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["--show-config", "--dry-run", "hi"])
    assert args.show_config is True
    assert args.dry_run is True


def test_run_dry_run_skips_llm_and_prints_config(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("plan_user_request should not be called in dry-run")

    monkeypatch.setattr("app.main.plan_user_request", boom)

    rc = run("create a sphere", Config(), dry_run=True)
    captured = capsys.readouterr()

    assert rc == 0
    assert "Received command: create a sphere" in captured.out
    assert "ProjectDev configuration:" in captured.out


def test_run_sends_prompt_and_prints_response(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_args: dict[str, object] = {}

    def fake_plan(prompt: str, *, config: Config) -> str:
        captured_args["prompt"] = prompt
        captured_args["config"] = config
        return "I would create a procedural rock."

    monkeypatch.setattr("app.main.plan_user_request", fake_plan)

    config = Config(lmstudio_base_url="http://x/v1", lmstudio_model="m")
    rc = run("make a procedural rock", config)
    out = capsys.readouterr().out

    assert rc == 0
    assert captured_args["prompt"] == "make a procedural rock"
    assert captured_args["config"] is config
    assert "Received command: make a procedural rock" in out
    assert "Assistant:" in out
    assert "I would create a procedural rock." in out


def test_run_reports_llm_errors_on_stderr(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_plan(prompt: str, *, config: Config) -> str:
        raise LMStudioUnreachableError("server down")

    monkeypatch.setattr("app.main.plan_user_request", fake_plan)

    rc = run("hi", Config(lmstudio_model="m"))
    captured = capsys.readouterr()

    assert rc == 1
    assert "Received command: hi" in captured.out
    assert "Assistant:" not in captured.out
    assert "LLM error: server down" in captured.err
