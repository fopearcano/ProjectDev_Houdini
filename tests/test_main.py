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
from app.houdini.schemas import (
    InspectSceneAction,
    ProjectPlan,
    SaveFileAction,
)
from app.llm.lmstudio_client import LMStudioUnreachableError
from app.llm.planner import PlannerError
from app.main import (
    build_parser,
    main,
    run_dry_run,
    run_execute,
    run_inspect,
)


# --- helpers ---------------------------------------------------------------


class _FakeBridge:
    def __init__(self, result: ExecutionResult) -> None:
        self._result = result
        self.calls: list[ProjectPlan] = []

    def execute(self, plan: ProjectPlan) -> ExecutionResult:
        self.calls.append(plan)
        return self._result


def _trivial_plan() -> ProjectPlan:
    return ProjectPlan.model_validate(
        {
            "user_goal": "make a sphere",
            "actions": [
                {
                    "action_type": "create_node",
                    "context_path": "/obj",
                    "node_type": "geo",
                    "node_name": "sphere_geo",
                }
            ],
            "notes": ["uses a sphere SOP"],
        }
    )


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
                    "path": "/obj/sphere_geo",
                    "name": "sphere_geo",
                    "type": "geo",
                    "children_count": 0,
                    "parameters": {},
                    "inputs": [],
                    "outputs": [],
                }
            ],
        },
    }


def _execution_result(
    plan: ProjectPlan, *, fail_index: int | None = None
) -> ExecutionResult:
    """Build a synthetic ExecutionResult for an augmented plan."""

    actions: list[ActionOutcome] = []
    for index, action in enumerate(plan.actions):
        if fail_index is not None and index == fail_index:
            actions.append(
                ActionOutcome(
                    index=index,
                    action_type=action.action_type,
                    success=False,
                    error=f"LookupError: simulated failure at {index}",
                )
            )
            continue
        if action.action_type == "inspect_scene":
            data: dict[str, Any] = _scene_payload()
        elif action.action_type == "save_file":
            data = {"file_path": action.file_path}  # type: ignore[union-attr]
        else:
            data = {"path": "/obj/sphere_geo"}
        actions.append(
            ActionOutcome(
                index=index,
                action_type=action.action_type,
                success=True,
                data=data,
            )
        )
    success = all(o.success for o in actions)
    return ExecutionResult(
        success=success,
        returncode=0,
        stdout="",
        stderr="",
        actions=actions,
    )


# --- parser ----------------------------------------------------------------


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


def test_build_parser_exposes_execution_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--execute", "--save", "/tmp/x.hip", "make", "a", "sphere"]
    )
    assert args.execute is True
    assert args.dry_run is False
    assert args.save == "/tmp/x.hip"


def test_dry_run_and_execute_are_mutually_exclusive() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dry-run", "--execute", "hi"])


def test_build_parser_supports_inspect_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["--inspect", "--depth", "2", "--context-path", "/obj"])
    assert args.inspect is True
    assert args.depth == 2
    assert args.context_path == "/obj"
    assert args.command == []


# --- run_dry_run -----------------------------------------------------------


def test_dry_run_does_not_invoke_bridge_and_prints_plan(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _trivial_plan()

    def fake_plan(prompt: str, *, config: Config) -> ProjectPlan:
        assert prompt == "make a sphere"
        return plan

    monkeypatch.setattr("app.main.plan_user_request", fake_plan)

    def boom(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("HoudiniBridge should not be touched in dry run")

    monkeypatch.setattr("app.main.HoudiniBridge.from_config", classmethod(boom))

    rc = run_dry_run("make a sphere", Config())
    out = capsys.readouterr().out

    assert rc == 0
    assert "Goal: make a sphere" in out
    assert "Plan (1 actions):" in out
    assert "create_node /obj/sphere_geo (geo)" in out
    assert "uses a sphere SOP" in out
    assert "(dry run -- nothing was executed)" in out


def test_dry_run_appends_save_file_action_visibly(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.main.plan_user_request", lambda p, **_: _trivial_plan())

    rc = run_dry_run("make a sphere", Config(), save_path="/tmp/out.hip")
    out = capsys.readouterr().out

    assert rc == 0
    assert "Plan (2 actions):" in out
    assert "save_file /tmp/out.hip" in out


def test_dry_run_reports_planner_error(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a: Any, **_kw: Any) -> Any:
        raise PlannerError("LLM produced garbage")

    monkeypatch.setattr("app.main.plan_user_request", boom)

    rc = run_dry_run("hi", Config())
    err = capsys.readouterr().err
    assert rc == 1
    assert "Planner error: LLM produced garbage" in err


def test_dry_run_reports_lmstudio_error(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a: Any, **_kw: Any) -> Any:
        raise LMStudioUnreachableError("server down")

    monkeypatch.setattr("app.main.plan_user_request", boom)

    rc = run_dry_run("hi", Config())
    err = capsys.readouterr().err
    assert rc == 1
    assert "LLM error: server down" in err


# --- run_execute -----------------------------------------------------------


def test_execute_appends_inspect_and_prints_full_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.main.plan_user_request", lambda p, **_: _trivial_plan())

    captured: dict[str, ProjectPlan] = {}

    class StubBridge:
        @classmethod
        def from_config(cls, _config: Config) -> "StubBridge":
            return cls()

        def execute(self, plan: ProjectPlan) -> ExecutionResult:
            captured["plan"] = plan
            return _execution_result(plan)

    monkeypatch.setattr("app.main.HoudiniBridge", StubBridge)

    rc = run_execute("make a sphere", Config())
    out = capsys.readouterr().out

    assert rc == 0
    plan = captured["plan"]
    assert isinstance(plan.actions[-1], InspectSceneAction)
    assert plan.actions[-1].context_path == "/obj"
    assert plan.actions[-1].max_depth == 1

    assert "Goal: make a sphere" in out
    assert "Plan (2 actions):" in out
    assert "Execution:" in out
    assert "1. create_node /obj/sphere_geo" in out
    assert "2. inspect_scene /obj" in out
    assert "Scene after execution:" in out
    assert "Hip file: /tmp/scene.hip" in out


def test_execute_with_save_path_inserts_save_action_before_inspect(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.main.plan_user_request", lambda p, **_: _trivial_plan())

    captured: dict[str, ProjectPlan] = {}

    class _Capturing:
        def execute(self, plan: ProjectPlan) -> ExecutionResult:
            captured["plan"] = plan
            return _execution_result(plan)

    rc = run_execute(
        "make a sphere",
        Config(),
        save_path="/tmp/out.hip",
        bridge=_Capturing(),
    )
    capsys.readouterr()

    assert rc == 0
    plan = captured["plan"]
    assert len(plan.actions) == 3
    assert isinstance(plan.actions[-2], SaveFileAction)
    assert plan.actions[-2].file_path == "/tmp/out.hip"
    assert isinstance(plan.actions[-1], InspectSceneAction)


def test_execute_reports_per_action_failures(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.main.plan_user_request", lambda p, **_: _trivial_plan())

    class _FailingBridge:
        def execute(self, plan: ProjectPlan) -> ExecutionResult:
            # First action fails; inspect_scene at index 1 still succeeds.
            return _execution_result(plan, fail_index=0)

    rc = run_execute("make a sphere", Config(), bridge=_FailingBridge())
    out = capsys.readouterr().out

    assert rc == 1
    assert "1. create_node /obj/sphere_geo (geo)" in out
    assert "error: LookupError: simulated failure at 0" in out
    assert "Errors:" in out
    assert "1. create_node /obj/sphere_geo (geo): LookupError" in out
    # Scene summary still shown because inspect succeeded.
    assert "Scene after execution:" in out


def test_execute_reports_bridge_errors(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.main.plan_user_request", lambda p, **_: _trivial_plan())

    def from_config(cls: Any, _config: Config) -> Any:
        raise HythonNotConfiguredError("HOUDINI_HYTHON_PATH not set")

    monkeypatch.setattr(
        "app.main.HoudiniBridge.from_config", classmethod(from_config)
    )

    rc = run_execute("hi", Config())
    err = capsys.readouterr().err

    assert rc == 1
    assert "Houdini error: HOUDINI_HYTHON_PATH not set" in err


def test_execute_handles_missing_inspect_outcome_gracefully(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.main.plan_user_request", lambda p, **_: _trivial_plan())

    class _NoInspectBridge:
        def execute(self, plan: ProjectPlan) -> ExecutionResult:
            outcomes = [
                ActionOutcome(
                    index=i,
                    action_type=plan.actions[i].action_type,
                    success=True,
                    data={"path": "/x"},
                )
                for i in range(len(plan.actions) - 1)  # drop inspect
            ]
            return ExecutionResult(
                success=False,
                returncode=0,
                stdout="",
                stderr="",
                actions=outcomes,
            )

    rc = run_execute("hi", Config(), bridge=_NoInspectBridge())
    out = capsys.readouterr().out

    assert rc == 1
    assert "Scene after execution: <inspection unavailable>" in out


# --- run_inspect (standalone) ---------------------------------------------


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
    assert isinstance(plan.actions[0], InspectSceneAction)
    assert plan.actions[0].context_path == "/obj"
    assert plan.actions[0].max_depth == 1
    assert "Hip file: /tmp/scene.hip" in out


def test_run_inspect_rejects_invalid_depth(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = run_inspect(Config(), depth=99)
    err = capsys.readouterr().err
    assert rc == 2
    assert "Invalid inspect arguments" in err


# --- main() routing --------------------------------------------------------


def test_main_default_is_dry_run(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_dry(command: str, config: Config, *, save_path: str | None = None) -> int:
        seen["mode"] = "dry"
        seen["command"] = command
        seen["save_path"] = save_path
        return 0

    def fake_exec(*_a: Any, **_kw: Any) -> int:
        seen["mode"] = "exec"
        return 0

    monkeypatch.setattr("app.main.run_dry_run", fake_dry)
    monkeypatch.setattr("app.main.run_execute", fake_exec)

    rc = main(["make", "a", "sphere"])
    capsys.readouterr()

    assert rc == 0
    assert seen["mode"] == "dry"
    assert seen["command"] == "make a sphere"
    assert seen["save_path"] is None


def test_main_execute_flag_routes_to_run_execute(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_exec(
        command: str,
        config: Config,
        *,
        save_path: str | None = None,
        inspect_context: str = "/obj",
        inspect_depth: int = 1,
        bridge: Any = None,
    ) -> int:
        seen["command"] = command
        seen["save_path"] = save_path
        seen["inspect_context"] = inspect_context
        seen["inspect_depth"] = inspect_depth
        return 0

    def boom(*_a: Any, **_kw: Any) -> int:
        raise AssertionError("dry run should not be called")

    monkeypatch.setattr("app.main.run_execute", fake_exec)
    monkeypatch.setattr("app.main.run_dry_run", boom)

    rc = main(
        [
            "--execute",
            "--save",
            "/tmp/out.hip",
            "--depth",
            "2",
            "--context-path",
            "/obj/geo1",
            "make",
            "a",
            "sphere",
        ]
    )
    capsys.readouterr()

    assert rc == 0
    assert seen["command"] == "make a sphere"
    assert seen["save_path"] == "/tmp/out.hip"
    assert seen["inspect_context"] == "/obj/geo1"
    assert seen["inspect_depth"] == 2


def test_main_inspect_flag_skips_other_modes(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_inspect(
        config: Config,
        *,
        context_path: str = "/obj",
        depth: int = 1,
        bridge: Any = None,
    ) -> int:
        seen["context_path"] = context_path
        seen["depth"] = depth
        return 0

    def boom(*_a: Any, **_kw: Any) -> int:
        raise AssertionError("dry/execute should not be called")

    monkeypatch.setattr("app.main.run_inspect", fake_inspect)
    monkeypatch.setattr("app.main.run_dry_run", boom)
    monkeypatch.setattr("app.main.run_execute", boom)

    rc = main(["--inspect", "--depth", "2", "--context-path", "/obj/geo1"])
    capsys.readouterr()

    assert rc == 0
    assert seen == {"context_path": "/obj/geo1", "depth": 2}


def test_main_show_config_prints_before_running(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.main.run_dry_run", lambda *a, **kw: 0)

    rc = main(["--show-config", "hi"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "ProjectDev configuration" in out
