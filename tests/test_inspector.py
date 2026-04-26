"""Tests for :mod:`app.houdini.inspector`."""

from __future__ import annotations

from typing import Any

import pytest

from app.houdini.inspector import (
    MAX_PARAMETERS_PER_NODE,
    collect_inputs,
    collect_node_parameters,
    collect_outputs,
    describe_node,
    describe_scene,
    format_scene_summary,
)


# --- Stub Houdini surface ---------------------------------------------------


class _Parm:
    def __init__(self, name: str, value: Any) -> None:
        self._name = name
        self._value = value

    def name(self) -> str:
        return self._name

    def eval(self) -> Any:
        return self._value


class _BadParm:
    def name(self) -> str:
        return "broken"

    def eval(self) -> Any:
        raise RuntimeError("evaluation failed")


class _Type:
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class _Node:
    def __init__(
        self,
        path: str,
        type_name: str = "geo",
        params: list[_Parm] | None = None,
    ) -> None:
        self._path = path
        self._name = path.rsplit("/", 1)[-1] or "/"
        self._type_name = type_name
        self._children: list["_Node"] = []
        self._inputs: list["_Node | None"] = []
        self._outputs: list["_Node"] = []
        self._params: list[_Parm] = list(params or [])

    def path(self) -> str:
        return self._path

    def name(self) -> str:
        return self._name

    def type(self) -> _Type:
        return _Type(self._type_name)

    def children(self) -> list["_Node"]:
        return list(self._children)

    def parms(self) -> list[_Parm]:
        return list(self._params)

    def inputs(self) -> tuple["_Node | None", ...]:
        return tuple(self._inputs)

    def outputs(self) -> tuple["_Node", ...]:
        return tuple(self._outputs)


class _Hou:
    def __init__(self, root: _Node, hip_path: str = "") -> None:
        self._index: dict[str, _Node] = {}
        self._register(root)
        self._hip_path = hip_path

    def _register(self, node: _Node) -> None:
        self._index[node.path()] = node
        for child in node.children():
            self._register(child)

    def node(self, path: str) -> _Node | None:
        return self._index.get(path)

    @property
    def hipFile(self):
        return self

    def path(self) -> str:
        return self._hip_path


# --- collect_node_parameters -----------------------------------------------


def test_collect_node_parameters_returns_simple_values() -> None:
    node = _Node(
        "/obj/geo1",
        params=[
            _Parm("tx", 1.5),
            _Parm("visible", True),
            _Parm("label", "rock"),
            _Parm("nothing", None),
        ],
    )
    params = collect_node_parameters(node)
    assert params == {"tx": 1.5, "visible": True, "label": "rock", "nothing": None}


def test_collect_node_parameters_keeps_short_lists() -> None:
    node = _Node("/obj/geo1", params=[_Parm("color", (0.1, 0.2, 0.3))])
    assert collect_node_parameters(node) == {"color": [0.1, 0.2, 0.3]}


def test_collect_node_parameters_skips_unjsonable_values() -> None:
    class _Opaque:
        pass

    node = _Node("/obj/geo1", params=[_Parm("ok", 1.0), _Parm("bad", _Opaque())])
    assert collect_node_parameters(node) == {"ok": 1.0}


def test_collect_node_parameters_tolerates_eval_errors() -> None:
    node = _Node("/obj/geo1", params=[_Parm("ok", 1.0), _BadParm()])
    assert collect_node_parameters(node) == {"ok": 1.0}


def test_collect_node_parameters_caps_at_max() -> None:
    parms = [_Parm(f"p{i}", float(i)) for i in range(MAX_PARAMETERS_PER_NODE + 5)]
    node = _Node("/obj/geo1", params=parms)
    result = collect_node_parameters(node)
    assert len(result) == MAX_PARAMETERS_PER_NODE


# --- collect_inputs / collect_outputs ---------------------------------------


def test_collect_inputs_reports_paths_and_empty_slots() -> None:
    src = _Node("/obj/box1")
    dst = _Node("/obj/transform1")
    dst._inputs = [src, None, src]

    assert collect_inputs(dst) == ["/obj/box1", None, "/obj/box1"]


def test_collect_outputs_reports_paths() -> None:
    src = _Node("/obj/box1")
    dst = _Node("/obj/transform1")
    src._outputs = [dst]
    assert collect_outputs(src) == ["/obj/transform1"]


def test_collect_inputs_handles_errors() -> None:
    class _BrokenNode:
        def inputs(self) -> Any:
            raise RuntimeError("oops")

    assert collect_inputs(_BrokenNode()) == []


# --- describe_node ----------------------------------------------------------


def test_describe_node_shape() -> None:
    parent = _Node("/obj/geo1")
    a = _Node("/obj/geo1/sphere1", type_name="sphere")
    b = _Node("/obj/geo1/transform1", type_name="xform")
    parent._children = [a, b]
    a._outputs = [b]
    b._inputs = [a]

    info = describe_node(parent, max_depth=1)

    assert info["path"] == "/obj/geo1"
    assert info["type"] == "geo"
    assert info["children_count"] == 2
    assert info["parameters"] == {}
    assert info["inputs"] == []
    assert info["outputs"] == []
    children_by_name = {c["name"]: c for c in info["children"]}
    assert children_by_name["sphere1"]["outputs"] == ["/obj/geo1/transform1"]
    assert children_by_name["transform1"]["inputs"] == ["/obj/geo1/sphere1"]
    # depth 1 stops at the immediate children: no grandchildren key.
    assert "children" not in children_by_name["sphere1"]


def test_describe_node_depth_two_includes_grandchildren() -> None:
    root = _Node("/obj")
    geo = _Node("/obj/geo1")
    inner = _Node("/obj/geo1/sphere1", type_name="sphere")
    root._children = [geo]
    geo._children = [inner]

    info = describe_node(root, max_depth=2)

    assert info["children"][0]["children"][0]["path"] == "/obj/geo1/sphere1"


# --- describe_scene ---------------------------------------------------------


def test_describe_scene_includes_hip_path() -> None:
    obj = _Node("/obj")
    geo = _Node("/obj/geo1")
    obj._children = [geo]
    hou = _Hou(obj, hip_path="/tmp/scene.hip")

    payload = describe_scene(hou, context_path="/obj", max_depth=1)

    assert payload["hip_file"] == "/tmp/scene.hip"
    assert payload["context_path"] == "/obj"
    assert payload["max_depth"] == 1
    assert payload["root"]["path"] == "/obj"
    assert payload["root"]["children_count"] == 1


def test_describe_scene_reports_unsaved_as_none() -> None:
    hou = _Hou(_Node("/obj"))
    payload = describe_scene(hou)
    assert payload["hip_file"] is None


def test_describe_scene_raises_when_context_missing() -> None:
    hou = _Hou(_Node("/obj"))
    with pytest.raises(LookupError):
        describe_scene(hou, context_path="/missing")


# --- format_scene_summary ---------------------------------------------------


def test_format_scene_summary_renders_readable_output() -> None:
    payload = {
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
                    "inputs": [None, "/obj/box1"],
                    "outputs": ["/obj/transform1"],
                }
            ],
        },
    }

    text = format_scene_summary(payload)

    assert "Hip file: /tmp/scene.hip" in text
    assert "Context: /obj (depth=1)" in text
    assert "/obj/geo1  [geo]  children=0" in text
    assert "params: tx=1.0" in text
    assert "inputs: <empty>, /obj/box1" in text
    assert "outputs: /obj/transform1" in text


def test_format_scene_summary_reports_unsaved() -> None:
    payload = {
        "hip_file": None,
        "context_path": "/obj",
        "max_depth": 1,
        "root": {
            "path": "/obj",
            "name": "obj",
            "type": "obj_context",
            "children_count": 0,
            "parameters": {},
            "inputs": [],
            "outputs": [],
            "children": [],
        },
    }
    assert "Hip file: <unsaved>" in format_scene_summary(payload)
