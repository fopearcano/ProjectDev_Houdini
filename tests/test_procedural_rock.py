"""Tests for :mod:`app.skills.procedural_rock`."""

from __future__ import annotations

import pytest

from app.houdini.schemas import (
    ConnectNodesAction,
    CreateNodeAction,
    LayoutChildrenAction,
    ProjectPlan,
)
from app.skills.procedural_rock import (
    DEFAULT_NOISE_AMPLITUDE,
    DEFAULT_ROUGHNESS,
    DEFAULT_SCALE,
    GEO_PATH,
    build_plan,
)


def test_build_plan_returns_validated_project_plan() -> None:
    plan = build_plan("make a procedural rock")

    assert isinstance(plan, ProjectPlan)
    # Round-trip through JSON to confirm the plan is fully valid.
    restored = ProjectPlan.model_validate_json(plan.model_dump_json())
    assert restored == plan


def test_build_plan_creates_expected_node_chain() -> None:
    plan = build_plan("make a rock")

    creates = [a for a in plan.actions if isinstance(a, CreateNodeAction)]
    by_name = {a.node_name: a for a in creates}

    assert by_name["procedural_rock_geo"].context_path == "/obj"
    assert by_name["procedural_rock_geo"].node_type == "geo"

    for name, expected_type in (
        ("sphere1", "sphere"),
        ("mountain1", "mountain"),
        ("color1", "color"),
        ("normal1", "normal"),
        ("OUT_ROCK", "null"),
    ):
        action = by_name[name]
        assert action.node_type == expected_type
        assert action.context_path == GEO_PATH


def test_build_plan_wires_chain_in_correct_order() -> None:
    plan = build_plan("make a rock")

    connections = [
        (a.from_path, a.to_path)
        for a in plan.actions
        if isinstance(a, ConnectNodesAction)
    ]
    assert connections == [
        (f"{GEO_PATH}/sphere1", f"{GEO_PATH}/mountain1"),
        (f"{GEO_PATH}/mountain1", f"{GEO_PATH}/color1"),
        (f"{GEO_PATH}/color1", f"{GEO_PATH}/normal1"),
        (f"{GEO_PATH}/normal1", f"{GEO_PATH}/OUT_ROCK"),
    ]


def test_build_plan_layouts_inside_geo() -> None:
    plan = build_plan("make a rock")
    layouts = [a for a in plan.actions if isinstance(a, LayoutChildrenAction)]
    assert len(layouts) == 1
    assert layouts[0].context_path == GEO_PATH


def test_build_plan_uses_default_parameters() -> None:
    plan = build_plan("make a rock")
    by_name = {
        a.node_name: a for a in plan.actions if isinstance(a, CreateNodeAction)
    }

    sphere_params = by_name["sphere1"].parameters
    mountain_params = by_name["mountain1"].parameters
    color_params = by_name["color1"].parameters

    assert sphere_params["radx"] == DEFAULT_SCALE
    assert sphere_params["rady"] == DEFAULT_SCALE
    assert sphere_params["radz"] == DEFAULT_SCALE
    assert mountain_params["height"] == DEFAULT_NOISE_AMPLITUDE
    assert 0.05 <= mountain_params["elementsize"] <= 1.0
    assert 0.0 <= color_params["colorr"] <= 1.0


def test_build_plan_custom_parameters_flow_through() -> None:
    plan = build_plan(
        "make a stone",
        scale=2.5,
        noise_amplitude=0.8,
        roughness=1.0,
        color=(0.1, 0.2, 0.3),
    )
    by_name = {
        a.node_name: a for a in plan.actions if isinstance(a, CreateNodeAction)
    }

    assert by_name["sphere1"].parameters["radx"] == 2.5
    assert by_name["mountain1"].parameters["height"] == 0.8
    # roughness=1.0 should drive elementsize to its small clamp.
    assert by_name["mountain1"].parameters["elementsize"] == pytest.approx(0.2)
    assert by_name["color1"].parameters["colorr"] == 0.1
    assert by_name["color1"].parameters["colorg"] == 0.2
    assert by_name["color1"].parameters["colorb"] == 0.3


def test_build_plan_roughness_zero_gives_smooth() -> None:
    plan = build_plan("a rock", roughness=0.0)
    by_name = {
        a.node_name: a for a in plan.actions if isinstance(a, CreateNodeAction)
    }
    assert by_name["mountain1"].parameters["elementsize"] == pytest.approx(1.0)


def test_build_plan_records_user_goal_and_notes() -> None:
    plan = build_plan("make a mossy boulder")
    assert plan.user_goal == "make a mossy boulder"
    assert plan.notes
    assert any("OUT_ROCK" in note for note in plan.notes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": 0.0},
        {"scale": -1.0},
        {"noise_amplitude": -0.1},
        {"roughness": -0.1},
        {"roughness": 1.1},
        {"color": (0.5, 0.5)},
        {"color": (0.5, 0.5, 1.5)},
    ],
)
def test_build_plan_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        build_plan("a rock", **kwargs)


def test_build_plan_falls_back_to_default_goal_when_blank() -> None:
    plan = build_plan("   ")
    assert plan.user_goal == "create a procedural rock"
