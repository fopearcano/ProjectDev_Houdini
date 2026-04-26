"""Procedural rock skill.

Builds a small, clean SOP network under ``/obj/procedural_rock_geo``:

    sphere1 -> mountain1 -> color1 -> normal1 -> OUT_ROCK

The skill produces a validated :class:`ProjectPlan`; it does not run
anything in Houdini. Default node types are standard Houdini SOPs that
should be available in a stock install (``sphere``, ``mountain``,
``color``, ``normal``, ``null``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from app.houdini.schemas import ProjectPlan

GEO_PATH: Final[str] = "/obj/procedural_rock_geo"

DEFAULT_SCALE: Final[float] = 1.0
DEFAULT_NOISE_AMPLITUDE: Final[float] = 0.2
DEFAULT_ROUGHNESS: Final[float] = 0.5
DEFAULT_COLOR: Final[tuple[float, float, float]] = (0.45, 0.40, 0.35)


@dataclass(frozen=True)
class RockParameters:
    """User-tunable inputs for the procedural rock skill."""

    scale: float = DEFAULT_SCALE
    noise_amplitude: float = DEFAULT_NOISE_AMPLITUDE
    roughness: float = DEFAULT_ROUGHNESS
    color: tuple[float, float, float] = DEFAULT_COLOR

    def validated(self) -> "RockParameters":
        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        if self.noise_amplitude < 0:
            raise ValueError("noise_amplitude must be >= 0")
        if not 0.0 <= self.roughness <= 1.0:
            raise ValueError("roughness must be in [0, 1]")
        if len(self.color) != 3 or not all(0.0 <= c <= 1.0 for c in self.color):
            raise ValueError("color must be three floats in [0, 1]")
        return self


def _roughness_to_element_size(roughness: float) -> float:
    """Map ``roughness`` in [0, 1] to a Houdini ``mountain`` element size.

    Smaller element sizes give finer, rougher noise features.
    """

    return max(0.05, 1.0 - 0.8 * roughness)


def build_plan(
    user_prompt: str,
    *,
    scale: float = DEFAULT_SCALE,
    noise_amplitude: float = DEFAULT_NOISE_AMPLITUDE,
    roughness: float = DEFAULT_ROUGHNESS,
    color: tuple[float, float, float] = DEFAULT_COLOR,
) -> ProjectPlan:
    """Return a validated :class:`ProjectPlan` for a procedural rock."""

    params = RockParameters(
        scale=scale,
        noise_amplitude=noise_amplitude,
        roughness=roughness,
        color=color,
    ).validated()

    sphere_path = f"{GEO_PATH}/sphere1"
    mountain_path = f"{GEO_PATH}/mountain1"
    color_path = f"{GEO_PATH}/color1"
    normal_path = f"{GEO_PATH}/normal1"
    out_path = f"{GEO_PATH}/OUT_ROCK"

    element_size = _roughness_to_element_size(params.roughness)
    cr, cg, cb = params.color

    actions: list[dict] = [
        {
            "action_type": "create_node",
            "context_path": "/obj",
            "node_type": "geo",
            "node_name": "procedural_rock_geo",
        },
        {
            "action_type": "create_node",
            "context_path": GEO_PATH,
            "node_type": "sphere",
            "node_name": "sphere1",
            "parameters": {
                "type": 2,  # polygon mesh, gives mountain SOP something to deform
                "radx": params.scale,
                "rady": params.scale,
                "radz": params.scale,
            },
        },
        {
            "action_type": "create_node",
            "context_path": GEO_PATH,
            "node_type": "mountain",
            "node_name": "mountain1",
            "parameters": {
                "height": params.noise_amplitude,
                "elementsize": element_size,
            },
        },
        {
            "action_type": "create_node",
            "context_path": GEO_PATH,
            "node_type": "color",
            "node_name": "color1",
            "parameters": {
                "colorr": cr,
                "colorg": cg,
                "colorb": cb,
            },
        },
        {
            "action_type": "create_node",
            "context_path": GEO_PATH,
            "node_type": "normal",
            "node_name": "normal1",
        },
        {
            "action_type": "create_node",
            "context_path": GEO_PATH,
            "node_type": "null",
            "node_name": "OUT_ROCK",
        },
        {
            "action_type": "connect_nodes",
            "from_path": sphere_path,
            "to_path": mountain_path,
        },
        {
            "action_type": "connect_nodes",
            "from_path": mountain_path,
            "to_path": color_path,
        },
        {
            "action_type": "connect_nodes",
            "from_path": color_path,
            "to_path": normal_path,
        },
        {
            "action_type": "connect_nodes",
            "from_path": normal_path,
            "to_path": out_path,
        },
        {
            "action_type": "layout_children",
            "context_path": GEO_PATH,
        },
    ]

    notes = [
        f"Procedural rock under {GEO_PATH} (scale={params.scale}, "
        f"amplitude={params.noise_amplitude}, roughness={params.roughness}).",
        f"OUT_ROCK is the network's clean output null at {out_path}.",
    ]

    return ProjectPlan.model_validate(
        {
            "user_goal": user_prompt.strip() or "create a procedural rock",
            "actions": actions,
            "notes": notes,
        }
    )


__all__ = [
    "DEFAULT_COLOR",
    "DEFAULT_NOISE_AMPLITUDE",
    "DEFAULT_ROUGHNESS",
    "DEFAULT_SCALE",
    "GEO_PATH",
    "RockParameters",
    "build_plan",
]
