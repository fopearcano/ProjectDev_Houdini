"""Tests for :mod:`app.main`."""

from __future__ import annotations

from typing import Any

import pytest

from app.config import Config
from app.houdini.bridge import (
    ActionOutcome,
    ExecutionResult,
    HythonNotConfiguredError,
)
from app.houdini.schemas import InspectSceneAction, ProjectPlan
from app.llm.lmstudio_client import LMStudioUnreachableError
from app.main import build_parser, main, run, run_inspect


def test_main_requires_command_unless_inspect_given(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main([])
    err = capsys.readouterr().err
    assert "command is required" in err


def test_build_parser_joins_command_words() -> None:
    parser = build_parser()
    args = parser.parse_args(["create", "a", "sphere"])
    assert args.command == ["create", "a", "sphere"]


def test_build_parser_supports_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["--show-config", "--dry-run", "hi"])
    assert args.show_config is True
    assert args.dry_run is True


def test_build_parser_supports_inspect_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["--inspect", "--depth", "2", "--context-path", "/obj"])
    assert args.inspect is True
    assert args.depth == 2
    assert args.context_path == "/obj"
    assert args.command == []


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


def test_run_prints_project_plan_when_skill_matches(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = ProjectPlan.model_validate(
        {
            "user_goal": "make a rock",
            "actions": [
                {
                    "action_type": "create_node",
                    "context_path": "/obj",
                    "node_type": "geo",
                    "node_name": "procedural_rock_geo",
                }
            ],
        }
    )

    def fake_plan(prompt: str, *, config: Config) -> ProjectPlan:
        return plan

    monkeypatch.setattr("app.main.plan_user_request", fake_plan)

    rc = run("make a rock", Config())
    out = capsys.readouterr().out

    assert rc == 0
    assert "Plan (1 actions):" in out
    assert '"action_type": "create_node"' in out
    assert "procedural_rock_geo" in out


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


# --- --inspect -------------------------------------------------------------


class _FakeBridge:
    def __init__(self, result: ExecutionResult) -> None:
        self._result = result
        self.calls: list[ProjectPlan] = []

    def execute(self, plan: ProjectPlan) -> ExecutionResult:
        self.calls.append(plan)
        return self._result


def _scene_payload() -> dict[str, Any]:
    return {
        "hip_file": "/tmp/scene.hip",
        "context_path": "/obj",
        "max_depth": 1,
        "root": {
            "path": "/obj",
            "name": "obj",
            "type": "obj_context",
            "children_count": 1,
            "parameters": {},
            "inputs": [],
            "outputs": [],
            "children": [
                {
                    "path": "/obj/geo1",
                    "name": "geo1",
                    "type": "geo",
                    "children_count": 0,
                    "parameters": {"tx": 1.0},
                    "inputs": [],
                    "outputs": [],
                }
            ],
        },
    }


def test_run_inspect_invokes_bridge_and_prints_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bridge = _FakeBridge(
        ExecutionResult(
            success=True,
            returncode=0,
            stdout="",
            stderr="",
            actions=[
                ActionOutcome(
                    index=0,
                    action_type="inspect_scene",
                    success=True,
                    data=_scene_payload(),
                )
            ],
        )
    )

    rc = run_inspect(Config(), context_path="/obj", depth=1, bridge=bridge)
    out = capsys.readouterr().out

    assert rc == 0
    assert len(bridge.calls) == 1
    plan = bridge.calls[0]
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert isinstance(action, InspectSceneAction)
    assert action.context_path == "/obj"
    assert action.max_depth == 1
    assert "Hip file: /tmp/scene.hip" in out
    assert "/obj/geo1  [geo]  children=0" in out


def test_run_inspect_passes_custom_context_and_depth(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bridge = _FakeBridge(
        ExecutionResult(
            success=True,
            returncode=0,
            stdout="",
            stderr="",
            actions=[
                ActionOutcome(
                    index=0,
                    action_type="inspect_scene",
                    success=True,
                    data=_scene_payload(),
                )
            ],
        )
    )

    rc = run_inspect(Config(), context_path="/mat", depth=2, bridge=bridge)
    assert rc == 0
    action = bridge.calls[0].actions[0]
    assert isinstance(action, InspectSceneAction)
    assert action.context_path == "/mat"
    assert action.max_depth == 2
    capsys.readouterr()  # drain


def test_run_inspect_rejects_invalid_depth(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = run_inspect(Config(), depth=99)
    err = capsys.readouterr().err
    assert rc == 2
    assert "Invalid inspect arguments" in err


def test_run_inspect_reports_bridge_error_on_stderr(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_config: Config) -> Any:
        raise HythonNotConfiguredError("HOUDINI_HYTHON_PATH not set")

    monkeypatch.setattr("app.main.HoudiniBridge.from_config", classmethod(lambda cls, c: boom(c)))

    rc = run_inspect(Config())
    err = capsys.readouterr().err
    assert rc == 1
    assert "HOUDINI_HYTHON_PATH not set" in err


def test_run_inspect_reports_action_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bridge = _FakeBridge(
        ExecutionResult(
            success=False,
            returncode=0,
            stdout="",
            stderr="",
            actions=[
                ActionOutcome(
                    index=0,
                    action_type="inspect_scene",
                    success=False,
                    error="LookupError: context_path not found: '/obj'",
                )
            ],
        )
    )

    rc = run_inspect(Config(), bridge=bridge)
    err = capsys.readouterr().err
    assert rc == 1
    assert "Inspection failed" in err
    assert "context_path not found" in err


def test_main_routes_inspect_flag_through_run_inspect(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_inspect(
        config: Config,
        *,
        context_path: str = "/obj",
        depth: int = 1,
        bridge: Any = None,
    ) -> int:
        captured["context_path"] = context_path
        captured["depth"] = depth
        return 0

    monkeypatch.setattr("app.main.run_inspect", fake_run_inspect)

    rc = main(["--inspect", "--depth", "2", "--context-path", "/obj/geo1"])
    assert rc == 0
    assert captured == {"context_path": "/obj/geo1", "depth": 2}
    capsys.readouterr()
