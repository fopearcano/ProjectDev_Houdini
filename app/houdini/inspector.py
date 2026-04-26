"""Scene inspection helpers.

These functions describe a Houdini scene as plain JSON-friendly
structures. They are intentionally self-contained (stdlib only) and
duck-typed so the executor can embed their source verbatim in the
generated hython script while the same functions remain importable
and testable from the host process via fakes.

Public surface:
    describe_scene(hou, *, context_path="/obj", max_depth=1) -> dict
    describe_node(node, *, max_depth=1, _depth=0)            -> dict
    format_scene_summary(payload)                            -> str
"""

from __future__ import annotations

from typing import Any, Mapping

MAX_PARAMETERS_PER_NODE = 12


def _is_simple(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None)))


def _to_jsonable(value: Any) -> Any:
    if _is_simple(value):
        return value
    if isinstance(value, (list, tuple)):
        if all(_is_simple(v) for v in value):
            return [v for v in value]
    return None


def collect_node_parameters(
    node: Any, *, max_params: int = MAX_PARAMETERS_PER_NODE
) -> dict[str, Any]:
    """Return a small dict of ``{name: value}`` for the node's parameters.

    Parameters whose value cannot be coerced to a JSON-friendly primitive
    or short list are skipped. Parameters whose ``eval()`` raises are
    also skipped. The result is capped at ``max_params``.
    """

    result: dict[str, Any] = {}
    try:
        parms = node.parms()
    except Exception:
        return result

    for parm in parms:
        if len(result) >= max_params:
            break
        try:
            name = parm.name()
        except Exception:
            continue
        try:
            raw = parm.eval()
        except Exception:
            continue
        value = _to_jsonable(raw)
        if value is None and not _is_simple(raw):
            continue
        result[name] = value
    return result


def collect_inputs(node: Any) -> list[str | None]:
    """Return the paths of each input slot (``None`` for empty slots)."""

    try:
        ins = node.inputs()
    except Exception:
        return []
    return [n.path() if n is not None else None for n in ins]


def collect_outputs(node: Any) -> list[str]:
    """Return the paths of every node consuming this node's outputs."""

    try:
        outs = node.outputs()
    except Exception:
        return []
    return [n.path() for n in outs if n is not None]


def describe_node(node: Any, *, max_depth: int = 1, _depth: int = 0) -> dict[str, Any]:
    """Return a JSON-friendly description of ``node`` and its descendants."""

    try:
        children = list(node.children())
    except Exception:
        children = []

    info: dict[str, Any] = {
        "path": node.path(),
        "name": node.name(),
        "type": node.type().name(),
        "children_count": len(children),
        "parameters": collect_node_parameters(node),
        "inputs": collect_inputs(node),
        "outputs": collect_outputs(node),
    }
    if _depth < max_depth:
        info["children"] = [
            describe_node(child, max_depth=max_depth, _depth=_depth + 1)
            for child in children
        ]
    return info


def describe_scene(
    hou: Any, *, context_path: str = "/obj", max_depth: int = 1
) -> dict[str, Any]:
    """Inspect ``context_path`` (default ``/obj``) and return scene info."""

    root = hou.node(context_path)
    if root is None:
        raise LookupError("context_path not found: " + repr(context_path))

    try:
        hip_path = hou.hipFile.path()
    except Exception:
        hip_path = None
    if not hip_path:
        hip_path = None

    return {
        "hip_file": hip_path,
        "context_path": context_path,
        "max_depth": max_depth,
        "root": describe_node(root, max_depth=max_depth),
    }


def _format_node(node: Mapping[str, Any], lines: list[str], *, indent: int) -> None:
    pad = "  " * indent
    lines.append(
        f"{pad}{node['path']}  [{node['type']}]  children={node['children_count']}"
    )
    params = node.get("parameters") or {}
    if params:
        rendered = ", ".join(f"{k}={v!r}" for k, v in params.items())
        lines.append(f"{pad}  params: {rendered}")
    inputs = node.get("inputs") or []
    if inputs:
        rendered = ", ".join(p if p is not None else "<empty>" for p in inputs)
        lines.append(f"{pad}  inputs: {rendered}")
    outputs = node.get("outputs") or []
    if outputs:
        lines.append(f"{pad}  outputs: {', '.join(outputs)}")
    for child in node.get("children", []) or []:
        _format_node(child, lines, indent=indent + 1)


def format_scene_summary(payload: Mapping[str, Any]) -> str:
    """Render a :func:`describe_scene` payload as readable text."""

    lines: list[str] = []
    lines.append(f"Hip file: {payload.get('hip_file') or '<unsaved>'}")
    lines.append(
        f"Context: {payload.get('context_path', '?')} "
        f"(depth={payload.get('max_depth', '?')})"
    )
    lines.append("")
    root = payload.get("root")
    if isinstance(root, Mapping):
        _format_node(root, lines, indent=0)
    else:
        lines.append("<no root data>")
    return "\n".join(lines)


__all__ = [
    "MAX_PARAMETERS_PER_NODE",
    "collect_inputs",
    "collect_node_parameters",
    "collect_outputs",
    "describe_node",
    "describe_scene",
    "format_scene_summary",
]
