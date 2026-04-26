"""Tests for :mod:`app.houdini.executor`.

These tests exercise script generation and the in-script handlers
without requiring Houdini, by injecting a small ``hou`` stub into
``sys.modules`` before executing the generated script.
"""

from __future__ import annotations

import json
import re
import sys
import types
from typing import Any

import pytest

from app.houdini.executor import (
    RESULT_BEGIN,
    RESULT_END,
    SUPPORTED_ACTION_TYPES,
    UnsupportedActionError,
    build_script,
)
from app.houdini.schemas import ProjectPlan


# --- Fake hou stub ---------------------------------------------------------


class _FakeParm:
    def __init__(self, name: str, value: Any = 0.0) -> None:
        self._name = name
        self._value: Any = value

    def name(self) -> str:
        return self._name

    def eval(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        self._value = value


class _FakeType:
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class _FakeNode:
    def __init__(self, path: str, name: str, type_name: str = "root") -> None:
        self._path = path
        self._name = name
        self._type_name = type_name
        self._children: list["_FakeNode"] = []
        self._params: dict[str, _FakeParm] = {}
        self._input_slots: list["_FakeNode | None"] = []
        self._consumers: list["_FakeNode"] = []
        self.laid_out_calls = 0

    # hou-compatible surface
    def path(self) -> str:
        return self._path

    def name(self) -> str:
        return self._name

    def type(self) -> _FakeType:
        return _FakeType(self._type_name)

    def parm(self, name: str) -> _FakeParm | None:
        return self._params.get(name)

    def parmTuple(self, name: str) -> _FakeParm | None:
        return None

    def parms(self) -> list[_FakeParm]:
        return list(self._params.values())

    def children(self) -> list["_FakeNode"]:
        return list(self._children)

    def createNode(self, node_type: str, node_name: str) -> "_FakeNode":
        path = (
            "/" + node_name
            if self._path == "/"
            else self._path.rstrip("/") + "/" + node_name
        )
        child = _FakeNode(path=path, name=node_name, type_name=node_type)
        # Pretend the node exposes a few common parameters.
        child._params["tx"] = _FakeParm("tx", 0.0)
        child._params["visible"] = _FakeParm("visible", True)
        child._params["radx"] = _FakeParm("radx", 1.0)
        self._children.append(child)
        return child

    def setInput(self, idx: int, src: "_FakeNode", output_idx: int = 0) -> None:
        while len(self._input_slots) <= idx:
            self._input_slots.append(None)
        self._input_slots[idx] = src
        if self not in src._consumers:
            src._consumers.append(self)

    def inputs(self) -> tuple["_FakeNode | None", ...]:
        return tuple(self._input_slots)

    def outputs(self) -> tuple["_FakeNode", ...]:
        return tuple(self._consumers)

    def layoutChildren(self) -> None:
        self.laid_out_calls += 1


class _FakeHipFile:
    def __init__(self) -> None:
        self.saved: list[str] = []
        self._path: str = ""

    def save(self, path: str) -> None:
        self.saved.append(path)
        self._path = path

    def path(self) -> str:
        return self._path


class _FakeHou:
    def __init__(self) -> None:
        self.hipFile = _FakeHipFile()
        self.root = _FakeNode("/", "/", "root")
        # Pre-populate /obj like a fresh Houdini scene.
        self.root._children.append(_FakeNode("/obj", "obj", "obj_context"))

    def node(self, path: str) -> _FakeNode | None:
        if path == "/":
            return self.root
        parts = [p for p in path.split("/") if p]
        cursor = self.root
        for part in parts:
            match = next((c for c in cursor._children if c._name == part), None)
            if match is None:
                return None
            cursor = match
        return cursor


_FAKE_HOU = _FakeHou()


def _run_generated_script(plan: ProjectPlan) -> dict[str, Any]:
    """Compile and execute the generated script with a stub ``hou`` module."""

    global _FAKE_HOU
    _FAKE_HOU = _FakeHou()

    fake_module = types.ModuleType("hou")
    fake_module.node = _FAKE_HOU.node  # type: ignore[attr-defined]
    fake_module.hipFile = _FAKE_HOU.hipFile  # type: ignore[attr-defined]

    source = build_script(plan)
    sys.modules["hou"] = fake_module
    try:
        captured: list[str] = []

        def fake_print(*args: Any, **_kwargs: Any) -> None:
            captured.append(" ".join(str(a) for a in args))

        namespace: dict[str, Any] = {
            "__name__": "gen_runner",
            "print": fake_print,
        }
        exec(compile(source, "<gen>", "exec"), namespace)
        namespace["_run"]()
        text = "\n".join(captured)
    finally:
        sys.modules.pop("hou", None)

    begin = text.index(RESULT_BEGIN)
    end = text.index(RESULT_END, begin)
    blob = text[begin + len(RESULT_BEGIN) : end].strip()
    return json.loads(blob)


# --- Script generation -----------------------------------------------------


def test_supported_action_types_match_spec() -> None:
    assert SUPPORTED_ACTION_TYPES == {
        "create_node",
        "set_parameter",
        "connect_nodes",
        "layout_children",
        "save_file",
        "inspect_scene",
    }


def test_build_script_produces_valid_python() -> None:
    plan = ProjectPlan.model_validate(
        {
            "user_goal": "make a sphere",
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
    src = build_script(plan)
    compile(src, "<gen>", "exec")  # will raise on syntax error


def test_build_script_embeds_plan_and_imports_hou() -> None:
    plan = ProjectPlan.model_validate(
        {
            "user_goal": "make a sphere",
            "actions": [
                {
                    "action_type": "create_node",
                    "context_path": "/obj",
                    "node_type": "geo",
                    "node_name": "rock_special",
                }
            ],
        }
    )
    src = build_script(plan)

    assert "import hou" in src
    assert RESULT_BEGIN in src
    assert RESULT_END in src

    match = re.search(r"_PLAN = json\.loads\((.+?)\)\n", src)
    assert match is not None
    embedded = json.loads(eval(match.group(1)))  # repr() literal -> str -> json
    assert embedded["user_goal"] == "make a sphere"
    assert embedded["actions"][0]["node_name"] == "rock_special"


def test_build_script_is_deterministic_for_same_plan() -> None:
    plan = ProjectPlan.model_validate(
        {
            "user_goal": "x",
            "actions": [
                {
                    "action_type": "save_file",
                    "file_path": "/tmp/scene.hip",
                }
            ],
        }
    )
    assert build_script(plan) == build_script(plan)


def test_build_script_does_not_inline_user_text_as_code() -> None:
    plan = ProjectPlan.model_validate(
        {
            "user_goal": "tricky goal",
            "actions": [
                {
                    "action_type": "create_node",
                    "context_path": "/obj",
                    "node_type": "geo",
                    "node_name": "rock1",
                    "parameters": {"label": 'a"); import os; os.system("evil"); ("'},
                }
            ],
        }
    )
    src = build_script(plan)

    # The dangerous text only appears inside the embedded JSON literal,
    # never as raw Python tokens.
    assert "os.system" not in src.replace("os.system", "", 1) or True  # readability
    # Stronger check: the only occurrence of the payload is inside _PLAN literal.
    occurrences = [m.start() for m in re.finditer(r"os\.system", src)]
    plan_literal_start = src.index("_PLAN = json.loads(")
    plan_literal_end = src.index("\n", plan_literal_start)
    for pos in occurrences:
        assert plan_literal_start <= pos <= plan_literal_end


def test_build_script_rejects_unsupported_actions() -> None:
    plan = ProjectPlan.model_validate(
        {
            "user_goal": "delete it",
            "actions": [
                {"action_type": "delete_node", "target_path": "/obj/geo1"},
            ],
        }
    )
    with pytest.raises(UnsupportedActionError) as info:
        build_script(plan)
    assert "delete_node" in str(info.value)


# --- Generated-script behavior (with stub hou) -----------------------------


def test_generated_script_executes_create_node() -> None:
    plan = ProjectPlan.model_validate(
        {
            "user_goal": "x",
            "actions": [
                {
                    "action_type": "create_node",
                    "context_path": "/obj",
                    "node_type": "geo",
                    "node_name": "rock1",
                    "parameters": {"tx": 1.5, "visible": True},
                }
            ],
        }
    )
    payload = _run_generated_script(plan)

    assert payload["fatal"] is None
    assert payload["results"][0]["success"] is True
    assert payload["results"][0]["data"]["path"] == "/obj/rock1"


def test_generated_script_reports_missing_node() -> None:
    plan = ProjectPlan.model_validate(
        {
            "user_goal": "x",
            "actions": [
                {
                    "action_type": "set_parameter",
                    "target_path": "/obj/does_not_exist",
                    "parameters": {"tx": 1.0},
                }
            ],
        }
    )
    payload = _run_generated_script(plan)

    assert payload["fatal"] is None
    result = payload["results"][0]
    assert result["success"] is False
    assert "Node not found" in result["error"]
    assert "/obj/does_not_exist" in result["error"]


def test_generated_script_runs_full_pipeline() -> None:
    plan = ProjectPlan.model_validate(
        {
            "user_goal": "build something",
            "actions": [
                {
                    "action_type": "create_node",
                    "context_path": "/obj",
                    "node_type": "geo",
                    "node_name": "rock1",
                },
                {
                    "action_type": "create_node",
                    "context_path": "/obj",
                    "node_type": "geo",
                    "node_name": "rock2",
                },
                {
                    "action_type": "connect_nodes",
                    "from_path": "/obj/rock1",
                    "to_path": "/obj/rock2",
                },
                {
                    "action_type": "layout_children",
                    "context_path": "/obj",
                },
                {
                    "action_type": "save_file",
                    "file_path": "/tmp/scene.hip",
                },
                {
                    "action_type": "inspect_scene",
                    "context_path": "/obj",
                    "max_depth": 1,
                },
            ],
        }
    )
    payload = _run_generated_script(plan)

    assert payload["fatal"] is None
    types_ = [r["action_type"] for r in payload["results"]]
    assert types_ == [
        "create_node",
        "create_node",
        "connect_nodes",
        "layout_children",
        "save_file",
        "inspect_scene",
    ]
    assert all(r["success"] for r in payload["results"])
    assert _FAKE_HOU.hipFile.saved == ["/tmp/scene.hip"]
    inspect = payload["results"][-1]["data"]
    assert inspect["context_path"] == "/obj"
    assert inspect["max_depth"] == 1
    assert inspect["hip_file"] == "/tmp/scene.hip"
    root = inspect["root"]
    assert root["path"] == "/obj"
    assert root["children_count"] == 2
    assert {child["name"] for child in root["children"]} == {"rock1", "rock2"}
    # The connect_nodes step should be reflected in the inspector's input/output graph.
    rock2 = next(c for c in root["children"] if c["name"] == "rock2")
    assert "/obj/rock1" in rock2["inputs"]
