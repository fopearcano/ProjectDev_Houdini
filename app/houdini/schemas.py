"""Pydantic schemas for the safe Houdini action layer.

The LLM never executes Python directly against Houdini. Instead it
emits a :class:`ProjectPlan` whose ``actions`` are validated by these
models before any runner touches a Houdini session. Validation here is
deliberately strict: paths must look like Houdini node paths, names
must not contain shell- or path-dangerous characters, and parameter
values must be JSON-compatible primitives or nested containers of
primitives.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


# --- Validation helpers -----------------------------------------------------

_NODE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NODE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]*$")
_DANGEROUS_NAME_TOKENS = ("..", "//")
_MAX_NESTED_DEPTH = 6
_PARAM_PRIMITIVES = (str, int, float, bool, type(None))


def _validate_node_path(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("path must be a string")
    if not value.startswith("/"):
        raise ValueError(f"path must start with '/': {value!r}")
    if any(token in value for token in _DANGEROUS_NAME_TOKENS):
        raise ValueError(f"path contains a dangerous token: {value!r}")
    if not _NODE_PATH_RE.match(value):
        raise ValueError(
            f"path contains invalid characters: {value!r} "
            "(allowed: letters, digits, '_', '-', '.', '/')"
        )
    return value


def _validate_node_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("node name must be a string")
    if not value:
        raise ValueError("node name must not be empty")
    if not _NODE_NAME_RE.match(value):
        raise ValueError(
            f"node name {value!r} is invalid: must match [A-Za-z_][A-Za-z0-9_]*"
        )
    return value


def _validate_node_type(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("node_type must be a non-empty string")
    if any(c in value for c in (" ", "\t", "\n", ";", "|", "&", "$", "`", "\"", "'")):
        raise ValueError(f"node_type contains an invalid character: {value!r}")
    return value


def _validate_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > _MAX_NESTED_DEPTH:
        raise ValueError("parameter value is nested too deeply")

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, list):
        return [_validate_json_value(v, depth=depth + 1) for v in value]
    if isinstance(value, tuple):
        return [_validate_json_value(v, depth=depth + 1) for v in value]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"parameter dict keys must be strings, got {type(k).__name__}"
                )
            cleaned[k] = _validate_json_value(v, depth=depth + 1)
        return cleaned
    raise ValueError(
        f"parameter values must be JSON-compatible primitives, "
        f"got {type(value).__name__}"
    )


def _validate_parameters(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("parameters must be a JSON object")
    return _validate_json_value(value)  # type: ignore[return-value]


# --- Reusable annotated types ----------------------------------------------

NodePath = Annotated[
    str,
    StringConstraints(min_length=1, max_length=512),
]
NodeName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
]
NodeType = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
]
FilePath = Annotated[
    str,
    StringConstraints(min_length=1, max_length=1024),
]


# --- Base action ------------------------------------------------------------


class _BaseAction(BaseModel):
    """Common configuration and helpers for every action."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Concrete actions -------------------------------------------------------


class CreateNodeAction(_BaseAction):
    """Create a new node under ``context_path``."""

    action_type: Literal["create_node"] = "create_node"
    context_path: NodePath
    node_type: NodeType
    node_name: NodeName
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("context_path")
    @classmethod
    def _check_context(cls, value: str) -> str:
        return _validate_node_path(value)

    @field_validator("node_type")
    @classmethod
    def _check_type(cls, value: str) -> str:
        return _validate_node_type(value)

    @field_validator("node_name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        return _validate_node_name(value)

    @field_validator("parameters", mode="before")
    @classmethod
    def _check_params(cls, value: Any) -> dict[str, Any]:
        return _validate_parameters(value)


class SetParameterAction(_BaseAction):
    """Set one or more parameters on an existing node."""

    action_type: Literal["set_parameter"] = "set_parameter"
    target_path: NodePath
    parameters: dict[str, Any]

    @field_validator("target_path")
    @classmethod
    def _check_target(cls, value: str) -> str:
        return _validate_node_path(value)

    @field_validator("parameters", mode="before")
    @classmethod
    def _check_params(cls, value: Any) -> dict[str, Any]:
        params = _validate_parameters(value)
        if not params:
            raise ValueError("set_parameter requires at least one parameter")
        return params


class ConnectNodesAction(_BaseAction):
    """Wire ``from_path`` output into ``to_path`` input."""

    action_type: Literal["connect_nodes"] = "connect_nodes"
    from_path: NodePath
    to_path: NodePath
    from_output: int = 0
    to_input: int = 0

    @field_validator("from_path", "to_path")
    @classmethod
    def _check_paths(cls, value: str) -> str:
        return _validate_node_path(value)

    @field_validator("from_output", "to_input")
    @classmethod
    def _check_indices(cls, value: int) -> int:
        if value < 0 or value > 64:
            raise ValueError("connector indices must be between 0 and 64")
        return value

    @model_validator(mode="after")
    def _check_distinct(self) -> "ConnectNodesAction":
        if self.from_path == self.to_path:
            raise ValueError("connect_nodes from_path and to_path must differ")
        return self


class DeleteNodeAction(_BaseAction):
    """Delete the node at ``target_path``."""

    action_type: Literal["delete_node"] = "delete_node"
    target_path: NodePath

    @field_validator("target_path")
    @classmethod
    def _check_target(cls, value: str) -> str:
        return _validate_node_path(value)


class LayoutChildrenAction(_BaseAction):
    """Apply Houdini's auto-layout to the children of ``context_path``."""

    action_type: Literal["layout_children"] = "layout_children"
    context_path: NodePath

    @field_validator("context_path")
    @classmethod
    def _check_context(cls, value: str) -> str:
        return _validate_node_path(value)


class SaveFileAction(_BaseAction):
    """Save the current scene to ``file_path``."""

    action_type: Literal["save_file"] = "save_file"
    file_path: FilePath

    @field_validator("file_path")
    @classmethod
    def _check_file_path(cls, value: str) -> str:
        if any(token in value for token in ("..", "\x00")):
            raise ValueError(f"file_path contains a dangerous token: {value!r}")
        if any(c in value for c in (";", "|", "&", "`", "$", "\n", "\r")):
            raise ValueError(f"file_path contains an invalid character: {value!r}")
        return value


class InspectSceneAction(_BaseAction):
    """Return a lightweight description of the scene under ``context_path``."""

    action_type: Literal["inspect_scene"] = "inspect_scene"
    context_path: NodePath = "/"
    max_depth: int = Field(default=2, ge=1, le=10)

    @field_validator("context_path")
    @classmethod
    def _check_context(cls, value: str) -> str:
        return _validate_node_path(value)


# --- Discriminated union ----------------------------------------------------

Action = Annotated[
    Union[
        CreateNodeAction,
        SetParameterAction,
        ConnectNodesAction,
        DeleteNodeAction,
        LayoutChildrenAction,
        SaveFileAction,
        InspectSceneAction,
    ],
    Field(discriminator="action_type"),
]


# --- Top-level plan ---------------------------------------------------------


class ProjectPlan(BaseModel):
    """A validated sequence of actions produced by the LLM planner."""

    model_config = ConfigDict(extra="forbid")

    user_goal: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    actions: list[Action] = Field(default_factory=list)
    notes: list[Annotated[str, StringConstraints(max_length=2000)]] = Field(
        default_factory=list
    )


__all__ = [
    "Action",
    "ConnectNodesAction",
    "CreateNodeAction",
    "DeleteNodeAction",
    "InspectSceneAction",
    "LayoutChildrenAction",
    "ProjectPlan",
    "SaveFileAction",
    "SetParameterAction",
]
