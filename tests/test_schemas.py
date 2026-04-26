"""Tests for :mod:`app.houdini.schemas`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.houdini.schemas import (
    ConnectNodesAction,
    CreateNodeAction,
    DeleteNodeAction,
    InspectSceneAction,
    LayoutChildrenAction,
    ProjectPlan,
    SaveFileAction,
    SetParameterAction,
)


# --- create_node ------------------------------------------------------------


def test_valid_create_node_action() -> None:
    action = CreateNodeAction.model_validate(
        {
            "action_type": "create_node",
            "context_path": "/obj",
            "node_type": "geo",
            "node_name": "rock1",
            "parameters": {"tx": 1.0, "visible": True, "label": "rock"},
        }
    )

    assert action.action_type == "create_node"
    assert action.context_path == "/obj"
    assert action.node_type == "geo"
    assert action.node_name == "rock1"
    assert action.parameters == {"tx": 1.0, "visible": True, "label": "rock"}


def test_create_node_defaults_parameters_to_empty() -> None:
    action = CreateNodeAction.model_validate(
        {
            "action_type": "create_node",
            "context_path": "/obj",
            "node_type": "geo",
            "node_name": "rock1",
        }
    )
    assert action.parameters == {}


def test_create_node_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CreateNodeAction.model_validate(
            {
                "action_type": "create_node",
                "context_path": "/obj",
                "node_type": "geo",
                "node_name": "rock1",
                "code": "import os; os.system('rm -rf /')",
            }
        )


def test_create_node_rejects_path_without_leading_slash() -> None:
    with pytest.raises(ValidationError) as info:
        CreateNodeAction.model_validate(
            {
                "action_type": "create_node",
                "context_path": "obj",
                "node_type": "geo",
                "node_name": "rock1",
            }
        )
    assert "must start with '/'" in str(info.value)


@pytest.mark.parametrize(
    "bad_path",
    [
        "/obj/../etc",
        "/obj//geo",
        "/obj/geo;rm",
        "/obj/geo|cat",
        "/obj/geo with space",
    ],
)
def test_create_node_rejects_dangerous_paths(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        CreateNodeAction.model_validate(
            {
                "action_type": "create_node",
                "context_path": bad_path,
                "node_type": "geo",
                "node_name": "rock1",
            }
        )


@pytest.mark.parametrize(
    "bad_name",
    [
        "1starts_with_digit",
        "has space",
        "has/slash",
        "rm -rf",
        "name;ls",
        "$evil",
        "..",
        "",
    ],
)
def test_create_node_rejects_invalid_names(bad_name: str) -> None:
    with pytest.raises(ValidationError):
        CreateNodeAction.model_validate(
            {
                "action_type": "create_node",
                "context_path": "/obj",
                "node_type": "geo",
                "node_name": bad_name,
            }
        )


def test_create_node_rejects_non_json_parameter_values() -> None:
    with pytest.raises(ValidationError) as info:
        CreateNodeAction.model_validate(
            {
                "action_type": "create_node",
                "context_path": "/obj",
                "node_type": "geo",
                "node_name": "rock1",
                "parameters": {"callback": lambda: None},
            }
        )
    assert "JSON-compatible" in str(info.value)


def test_create_node_accepts_nested_json_parameters() -> None:
    action = CreateNodeAction.model_validate(
        {
            "action_type": "create_node",
            "context_path": "/obj",
            "node_type": "geo",
            "node_name": "rock1",
            "parameters": {
                "color": [0.1, 0.2, 0.3],
                "tags": {"kind": "hero", "lod": [0, 1, 2]},
            },
        }
    )
    assert action.parameters["tags"]["lod"] == [0, 1, 2]


def test_action_is_frozen() -> None:
    action = CreateNodeAction.model_validate(
        {
            "action_type": "create_node",
            "context_path": "/obj",
            "node_type": "geo",
            "node_name": "rock1",
        }
    )
    with pytest.raises(ValidationError):
        action.node_name = "other"  # type: ignore[misc]


# --- set_parameter ----------------------------------------------------------


def test_set_parameter_requires_parameters() -> None:
    with pytest.raises(ValidationError):
        SetParameterAction.model_validate(
            {
                "action_type": "set_parameter",
                "target_path": "/obj/geo1",
                "parameters": {},
            }
        )


def test_set_parameter_validates_path() -> None:
    with pytest.raises(ValidationError):
        SetParameterAction.model_validate(
            {
                "action_type": "set_parameter",
                "target_path": "obj/geo1",
                "parameters": {"tx": 1.0},
            }
        )


# --- connect_nodes ----------------------------------------------------------


def test_connect_nodes_valid() -> None:
    action = ConnectNodesAction.model_validate(
        {
            "action_type": "connect_nodes",
            "from_path": "/obj/geo1/box1",
            "to_path": "/obj/geo1/transform1",
            "from_output": 0,
            "to_input": 0,
        }
    )
    assert action.from_path == "/obj/geo1/box1"


def test_connect_nodes_rejects_self_connection() -> None:
    with pytest.raises(ValidationError):
        ConnectNodesAction.model_validate(
            {
                "action_type": "connect_nodes",
                "from_path": "/obj/geo1/box1",
                "to_path": "/obj/geo1/box1",
            }
        )


def test_connect_nodes_rejects_negative_index() -> None:
    with pytest.raises(ValidationError):
        ConnectNodesAction.model_validate(
            {
                "action_type": "connect_nodes",
                "from_path": "/obj/a",
                "to_path": "/obj/b",
                "from_output": -1,
            }
        )


# --- delete / layout / inspect ---------------------------------------------


def test_delete_node_valid() -> None:
    action = DeleteNodeAction.model_validate(
        {"action_type": "delete_node", "target_path": "/obj/geo1"}
    )
    assert action.target_path == "/obj/geo1"


def test_layout_children_valid() -> None:
    action = LayoutChildrenAction.model_validate(
        {"action_type": "layout_children", "context_path": "/obj/geo1"}
    )
    assert action.context_path == "/obj/geo1"


def test_inspect_scene_defaults() -> None:
    action = InspectSceneAction.model_validate({"action_type": "inspect_scene"})
    assert action.context_path == "/obj"
    assert action.max_depth == 1


def test_inspect_scene_max_depth_bounds() -> None:
    with pytest.raises(ValidationError):
        InspectSceneAction.model_validate(
            {"action_type": "inspect_scene", "max_depth": 0}
        )
    with pytest.raises(ValidationError):
        InspectSceneAction.model_validate(
            {"action_type": "inspect_scene", "max_depth": 99}
        )


# --- save_file --------------------------------------------------------------


def test_save_file_valid() -> None:
    action = SaveFileAction.model_validate(
        {"action_type": "save_file", "file_path": "/tmp/scene.hip"}
    )
    assert action.file_path == "/tmp/scene.hip"


@pytest.mark.parametrize(
    "bad_path",
    [
        "/tmp/../etc/passwd",
        "/tmp/scene.hip; rm -rf /",
        "/tmp/$(whoami).hip",
        "/tmp/scene\nfoo.hip",
    ],
)
def test_save_file_rejects_dangerous_paths(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        SaveFileAction.model_validate(
            {"action_type": "save_file", "file_path": bad_path}
        )


# --- ProjectPlan ------------------------------------------------------------


def test_complete_project_plan_parses() -> None:
    plan = ProjectPlan.model_validate(
        {
            "user_goal": "make a procedural rock",
            "actions": [
                {
                    "action_type": "create_node",
                    "context_path": "/obj",
                    "node_type": "geo",
                    "node_name": "rock1",
                },
                {
                    "action_type": "create_node",
                    "context_path": "/obj/rock1",
                    "node_type": "sphere",
                    "node_name": "sphere1",
                    "parameters": {"radx": 1.0, "rady": 1.0, "radz": 1.0},
                },
                {
                    "action_type": "set_parameter",
                    "target_path": "/obj/rock1/sphere1",
                    "parameters": {"type": 2},
                },
                {
                    "action_type": "connect_nodes",
                    "from_path": "/obj/rock1/sphere1",
                    "to_path": "/obj/rock1/output1",
                },
                {"action_type": "layout_children", "context_path": "/obj/rock1"},
                {"action_type": "inspect_scene", "context_path": "/obj"},
                {"action_type": "save_file", "file_path": "/tmp/rock.hip"},
            ],
            "notes": ["Use a mountain SOP for displacement next."],
        }
    )

    assert plan.user_goal == "make a procedural rock"
    assert len(plan.actions) == 7
    assert isinstance(plan.actions[0], CreateNodeAction)
    assert isinstance(plan.actions[2], SetParameterAction)
    assert isinstance(plan.actions[3], ConnectNodesAction)
    assert isinstance(plan.actions[4], LayoutChildrenAction)
    assert isinstance(plan.actions[5], InspectSceneAction)
    assert isinstance(plan.actions[6], SaveFileAction)
    assert plan.notes == ["Use a mountain SOP for displacement next."]


def test_project_plan_rejects_unknown_action_type() -> None:
    with pytest.raises(ValidationError):
        ProjectPlan.model_validate(
            {
                "user_goal": "do something",
                "actions": [{"action_type": "exec_python", "code": "rm -rf /"}],
            }
        )


def test_project_plan_requires_user_goal() -> None:
    with pytest.raises(ValidationError):
        ProjectPlan.model_validate({"actions": []})


def test_project_plan_round_trips_through_json() -> None:
    plan = ProjectPlan.model_validate(
        {
            "user_goal": "make a sphere",
            "actions": [
                {
                    "action_type": "create_node",
                    "context_path": "/obj",
                    "node_type": "geo",
                    "node_name": "sphere1",
                }
            ],
        }
    )
    payload = plan.model_dump_json()
    restored = ProjectPlan.model_validate_json(payload)
    assert restored == plan
