"""Tests for :mod:`app.main`."""

from __future__ import annotations

import pytest

from app.config import Config
from app.main import build_parser, run


def test_build_parser_requires_command() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_joins_command_words() -> None:
    parser = build_parser()
    args = parser.parse_args(["create", "a", "sphere"])
    assert args.command == ["create", "a", "sphere"]


def test_run_prints_command_and_config(capsys: pytest.CaptureFixture[str]) -> None:
    config = Config()
    rc = run("create a sphere in Houdini", config)
    captured = capsys.readouterr()

    assert rc == 0
    assert "Received command: create a sphere in Houdini" in captured.out
    assert "ProjectDev configuration:" in captured.out
