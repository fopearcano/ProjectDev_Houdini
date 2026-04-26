"""Tests for :mod:`app.houdini.bridge` (no Houdini required)."""

from __future__ import annotations

import json
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.config import Config
from app.houdini.bridge import (
    ExecutionResult,
    HoudiniBridge,
    HoudiniProcessError,
    HythonNotConfiguredError,
    HythonNotFoundError,
)
from app.houdini.executor import RESULT_BEGIN, RESULT_END
from app.houdini.schemas import ProjectPlan


@dataclass
class _FakeCompleted:
    args: list[str]
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@pytest.fixture()
def fake_hython(tmp_path: Path) -> Path:
    """Create a placeholder hython binary at a real, executable path."""

    path = tmp_path / "hython"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return path


def _trivial_plan() -> ProjectPlan:
    return ProjectPlan.model_validate(
        {
            "user_goal": "x",
            "actions": [
                {
                    "action_type": "create_node",
                    "context_path": "/obj",
                    "node_type": "geo",
                    "node_name": "rock1",
                }
            ],
        }
    )


def _result_payload(actions: list[dict[str, Any]], fatal: Any = None) -> str:
    return (
        f"\n{RESULT_BEGIN}\n"
        + json.dumps({"results": actions, "fatal": fatal})
        + f"\n{RESULT_END}\n"
    )


# --- Configuration ---------------------------------------------------------


def test_from_config_raises_when_path_missing() -> None:
    cfg = Config()
    with pytest.raises(HythonNotConfiguredError):
        HoudiniBridge.from_config(cfg)


def test_from_config_uses_configured_path(fake_hython: Path) -> None:
    cfg = Config(houdini_hython_path=str(fake_hython))
    bridge = HoudiniBridge.from_config(cfg, timeout=5.0)
    assert bridge.hython_path == str(fake_hython)
    assert bridge.timeout == 5.0


def test_execute_raises_when_hython_path_missing(tmp_path: Path) -> None:
    bridge = HoudiniBridge(hython_path=str(tmp_path / "missing"))
    with pytest.raises(HythonNotFoundError):
        bridge.execute(_trivial_plan())


# --- Successful execution --------------------------------------------------


def test_execute_invokes_runner_with_script_path(fake_hython: Path) -> None:
    seen: dict[str, Any] = {}

    def runner(args: list[str], **kwargs: Any) -> _FakeCompleted:
        seen["args"] = args
        seen["timeout"] = kwargs.get("timeout")
        seen["script_existed"] = Path(args[1]).is_file()
        seen["script_source"] = Path(args[1]).read_text(encoding="utf-8")
        return _FakeCompleted(
            args=args,
            stdout="some Houdini chatter\n"
            + _result_payload(
                [
                    {
                        "index": 0,
                        "action_type": "create_node",
                        "success": True,
                        "data": {"path": "/obj/rock1"},
                    }
                ]
            ),
        )

    bridge = HoudiniBridge(hython_path=str(fake_hython), timeout=7.5, runner=runner)
    result = bridge.execute(_trivial_plan())

    assert seen["args"][0] == str(fake_hython)
    assert seen["script_existed"] is True
    assert "import hou" in seen["script_source"]
    assert seen["timeout"] == 7.5
    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.returncode == 0
    assert result.error is None
    assert len(result.actions) == 1
    assert result.actions[0].action_type == "create_node"
    assert result.actions[0].success is True
    assert result.actions[0].data == {"path": "/obj/rock1"}


def test_execute_cleans_up_temp_script(fake_hython: Path) -> None:
    captured: dict[str, Any] = {}

    def runner(args: list[str], **_kwargs: Any) -> _FakeCompleted:
        captured["script_path"] = args[1]
        return _FakeCompleted(
            args=args,
            stdout=_result_payload([]),
        )

    bridge = HoudiniBridge(hython_path=str(fake_hython), runner=runner)
    bridge.execute(_trivial_plan())

    assert not Path(captured["script_path"]).exists()


def test_execute_reports_action_level_failure(fake_hython: Path) -> None:
    def runner(args: list[str], **_kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(
            args=args,
            stdout=_result_payload(
                [
                    {
                        "index": 0,
                        "action_type": "create_node",
                        "success": False,
                        "error": "LookupError: Node not found: '/obj/missing'",
                    }
                ]
            ),
        )

    bridge = HoudiniBridge(hython_path=str(fake_hython), runner=runner)
    result = bridge.execute(_trivial_plan())

    assert result.success is False
    assert result.returncode == 0
    assert result.error is None
    assert result.actions[0].success is False
    assert "Node not found" in (result.actions[0].error or "")


def test_execute_reports_fatal_payload(fake_hython: Path) -> None:
    def runner(args: list[str], **_kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(
            args=args,
            returncode=2,
            stdout=_result_payload([], fatal="hou import failed"),
        )

    bridge = HoudiniBridge(hython_path=str(fake_hython), runner=runner)
    result = bridge.execute(_trivial_plan())

    assert result.success is False
    assert result.error == "hou import failed"
    assert result.returncode == 2


def test_execute_handles_stdout_without_sentinels(fake_hython: Path) -> None:
    def runner(args: list[str], **_kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(
            args=args, returncode=1, stdout="random crash output", stderr="boom"
        )

    bridge = HoudiniBridge(hython_path=str(fake_hython), runner=runner)
    result = bridge.execute(_trivial_plan())

    assert result.success is False
    assert result.actions == []
    assert "exited with code 1" in (result.error or "")
    assert result.stderr == "boom"


def test_execute_handles_malformed_json_payload(fake_hython: Path) -> None:
    def runner(args: list[str], **_kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(
            args=args,
            stdout=f"{RESULT_BEGIN}\nnot json\n{RESULT_END}\n",
        )

    bridge = HoudiniBridge(hython_path=str(fake_hython), runner=runner)
    result = bridge.execute(_trivial_plan())

    assert result.success is False
    assert result.error is not None


# --- Process-level failures -----------------------------------------------


def test_execute_converts_timeout_to_process_error(fake_hython: Path) -> None:
    def runner(args: list[str], **kwargs: Any) -> _FakeCompleted:
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout", 0))

    bridge = HoudiniBridge(hython_path=str(fake_hython), timeout=3.0, runner=runner)
    with pytest.raises(HoudiniProcessError) as info:
        bridge.execute(_trivial_plan())
    assert "3.0" in str(info.value)


def test_execute_converts_file_not_found_to_hython_not_found(
    fake_hython: Path,
) -> None:
    def runner(*_args: Any, **_kwargs: Any) -> _FakeCompleted:
        raise FileNotFoundError("no such binary")

    bridge = HoudiniBridge(hython_path=str(fake_hython), runner=runner)
    with pytest.raises(HythonNotFoundError):
        bridge.execute(_trivial_plan())


def test_execute_converts_oserror_to_process_error(fake_hython: Path) -> None:
    def runner(*_args: Any, **_kwargs: Any) -> _FakeCompleted:
        raise OSError("permission denied")

    bridge = HoudiniBridge(hython_path=str(fake_hython), runner=runner)
    with pytest.raises(HoudiniProcessError):
        bridge.execute(_trivial_plan())


def test_execute_resolves_hython_via_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(_name: str) -> str:
        return "/usr/local/bin/hython"

    monkeypatch.setattr("app.houdini.bridge.shutil.which", fake_which)

    def runner(args: list[str], **_kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(args=args, stdout=_result_payload([]))

    bridge = HoudiniBridge(hython_path="hython", runner=runner)
    result = bridge.execute(_trivial_plan())
    assert result.success is True
    assert result.actions == []
