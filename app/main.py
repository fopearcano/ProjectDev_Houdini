"""Command-line entry point for ProjectDev.

The CLI exposes three modes:

* default / ``--dry-run`` (safe): plan the user's request and print the
  resulting :class:`ProjectPlan` without touching Houdini.
* ``--execute``: plan, then run the actions through ``hython``, append a
  final ``inspect_scene``, and print a summary of what happened.
* ``--inspect``: run a standalone ``inspect_scene`` against a live
  Houdini and print the readable scene tree.

``--save PATH`` adds a ``save_file`` action to the plan (visible in
dry-run, executed under ``--execute``).
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Sequence

from app.config import Config
from app.houdini.bridge import (
    ActionOutcome,
    ExecutionResult,
    HoudiniBridge,
    HoudiniBridgeError,
)
from app.houdini.inspector import format_scene_summary
from app.houdini.schemas import (
    Action,
    ConnectNodesAction,
    CreateNodeAction,
    DeleteNodeAction,
    InspectSceneAction,
    LayoutChildrenAction,
    ProjectPlan,
    SaveFileAction,
    SetParameterAction,
)
from app.llm.lmstudio_client import LMStudioError
from app.llm.planner import PlannerError, plan_user_request

DEFAULT_INSPECT_DEPTH = 1
MAX_INSPECT_DEPTH = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="projectdev",
        description=(
            "Natural language assistant for controlling Houdini "
            "via safe structured actions."
        ),
    )
    parser.add_argument(
        "command",
        nargs="*",
        help="Natural language command to execute (quoted). "
        "Optional when --inspect is given.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the loaded configuration before planning.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and print, but do not execute (default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Plan and execute through Houdini, then inspect the result.",
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run a standalone inspect_scene against Houdini and exit.",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Append a save_file action with the given .hip path.",
    )
    parser.add_argument(
        "--context-path",
        default="/obj",
        help="Houdini context path to inspect (default: /obj).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_INSPECT_DEPTH,
        help=f"Inspection depth (default: {DEFAULT_INSPECT_DEPTH}, "
        f"max: {MAX_INSPECT_DEPTH}).",
    )
    return parser


# --- formatting helpers ----------------------------------------------------


def _action_label(action: Action) -> str:
    if isinstance(action, CreateNodeAction):
        suffix = action.context_path.rstrip("/")
        return f"create_node {suffix}/{action.node_name} ({action.node_type})"
    if isinstance(action, SetParameterAction):
        keys = ", ".join(sorted(action.parameters.keys()))
        return f"set_parameter {action.target_path} [{keys}]"
    if isinstance(action, ConnectNodesAction):
        return (
            f"connect_nodes {action.from_path} -> {action.to_path} "
            f"({action.from_output}->{action.to_input})"
        )
    if isinstance(action, DeleteNodeAction):
        return f"delete_node {action.target_path}"
    if isinstance(action, LayoutChildrenAction):
        return f"layout_children {action.context_path}"
    if isinstance(action, SaveFileAction):
        return f"save_file {action.file_path}"
    if isinstance(action, InspectSceneAction):
        return (
            f"inspect_scene {action.context_path} "
            f"(depth={action.max_depth})"
        )
    return action.action_type  # pragma: no cover - exhaustive above


def _format_plan_lines(plan: ProjectPlan) -> list[str]:
    lines = [f"Plan ({len(plan.actions)} actions):"]
    for index, action in enumerate(plan.actions, start=1):
        lines.append(f"  {index:2d}. {_action_label(action)}")
    if plan.notes:
        lines.append("")
        lines.append("Notes:")
        for note in plan.notes:
            lines.append(f"  - {note}")
    return lines


def _format_execution_lines(
    plan: ProjectPlan, result: ExecutionResult
) -> list[str]:
    lines = ["Execution:"]
    outcomes_by_index = {o.index: o for o in result.actions}
    errors: list[str] = []
    for index, action in enumerate(plan.actions):
        outcome = outcomes_by_index.get(index)
        label = _action_label(action)
        if outcome is None:
            lines.append(f"  [ ] {index + 1:2d}. {label}  (no result)")
            continue
        marker = "✓" if outcome.success else "✗"
        lines.append(f"  [{marker}] {index + 1:2d}. {label}")
        if not outcome.success and outcome.error:
            lines.append(f"        error: {outcome.error}")
            errors.append(f"{index + 1}. {label}: {outcome.error}")

    if result.error:
        errors.append(f"process: {result.error}")

    if errors:
        lines.append("")
        lines.append("Errors:")
        for entry in errors:
            lines.append(f"  - {entry}")
    return lines


# --- plan augmentation ----------------------------------------------------


def _augment_plan(
    plan: ProjectPlan,
    *,
    save_path: str | None,
    inspect_context: str,
    inspect_depth: int,
) -> ProjectPlan:
    """Return a copy of ``plan`` with optional save_file and a final inspect."""

    extra: list[Action] = []
    if save_path:
        extra.append(SaveFileAction(file_path=save_path))
    extra.append(
        InspectSceneAction(context_path=inspect_context, max_depth=inspect_depth)
    )
    return ProjectPlan(
        user_goal=plan.user_goal,
        actions=list(plan.actions) + extra,
        notes=list(plan.notes),
    )


def _last_inspect_outcome(
    plan: ProjectPlan, result: ExecutionResult
) -> ActionOutcome | None:
    for index, action in reversed(list(enumerate(plan.actions))):
        if isinstance(action, InspectSceneAction):
            for outcome in result.actions:
                if outcome.index == index:
                    return outcome
            return None
    return None


# --- entry points ----------------------------------------------------------


def _plan_for_command(command: str, config: Config) -> ProjectPlan | int:
    try:
        return plan_user_request(command, config=config)
    except LMStudioError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 1
    except PlannerError as exc:
        print(f"Planner error: {exc}", file=sys.stderr)
        return 1


def run_dry_run(
    command: str,
    config: Config,
    *,
    save_path: str | None = None,
) -> int:
    print(f"Goal: {command}")
    print()

    plan = _plan_for_command(command, config)
    if isinstance(plan, int):
        return plan

    if save_path:
        plan = ProjectPlan(
            user_goal=plan.user_goal,
            actions=list(plan.actions) + [SaveFileAction(file_path=save_path)],
            notes=list(plan.notes),
        )

    print(f"Resolved goal: {plan.user_goal}")
    print()
    for line in _format_plan_lines(plan):
        print(line)
    print()
    print("(dry run -- nothing was executed)")
    return 0


def run_execute(
    command: str,
    config: Config,
    *,
    save_path: str | None = None,
    inspect_context: str = "/obj",
    inspect_depth: int = DEFAULT_INSPECT_DEPTH,
    bridge: HoudiniBridge | None = None,
) -> int:
    print(f"Goal: {command}")
    print()

    plan = _plan_for_command(command, config)
    if isinstance(plan, int):
        return plan

    augmented = _augment_plan(
        plan,
        save_path=save_path,
        inspect_context=inspect_context,
        inspect_depth=inspect_depth,
    )

    print(f"Resolved goal: {augmented.user_goal}")
    print()
    for line in _format_plan_lines(augmented):
        print(line)
    print()

    try:
        if bridge is None:
            bridge = HoudiniBridge.from_config(config)
        result = bridge.execute(augmented)
    except HoudiniBridgeError as exc:
        print(f"Houdini error: {exc}", file=sys.stderr)
        return 1

    for line in _format_execution_lines(augmented, result):
        print(line)

    inspect_outcome = _last_inspect_outcome(augmented, result)
    print()
    if (
        inspect_outcome is not None
        and inspect_outcome.success
        and inspect_outcome.data is not None
    ):
        print("Scene after execution:")
        print(format_scene_summary(inspect_outcome.data))
    else:
        print("Scene after execution: <inspection unavailable>")
        if inspect_outcome is not None and inspect_outcome.error:
            print(f"  inspect error: {inspect_outcome.error}")

    return 0 if result.success else 1


def run_inspect(
    config: Config,
    *,
    context_path: str = "/obj",
    depth: int = DEFAULT_INSPECT_DEPTH,
    bridge: HoudiniBridge | None = None,
) -> int:
    """Run an inspect_scene action through Houdini and print the summary."""

    try:
        action = InspectSceneAction(context_path=context_path, max_depth=depth)
    except Exception as exc:
        print(f"Invalid inspect arguments: {exc}", file=sys.stderr)
        return 2

    plan = ProjectPlan(
        user_goal=f"inspect {context_path} (depth={depth})",
        actions=[action],
    )

    try:
        if bridge is None:
            bridge = HoudiniBridge.from_config(config)
        result = bridge.execute(plan)
    except HoudiniBridgeError as exc:
        print(f"Houdini error: {exc}", file=sys.stderr)
        return 1

    if not result.actions:
        print(
            f"Inspection failed: {result.error or 'no result returned'}",
            file=sys.stderr,
        )
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return 1

    outcome = result.actions[0]
    if not outcome.success or outcome.data is None:
        print(
            f"Inspection failed: {outcome.error or 'unknown error'}",
            file=sys.stderr,
        )
        return 1

    print(format_scene_summary(outcome.data))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = Config.from_env()
    if args.show_config:
        print(config.describe())
        print()

    if args.inspect:
        return run_inspect(
            config,
            context_path=args.context_path,
            depth=args.depth,
        )

    if not args.command:
        parser.error("a command is required unless --inspect is given")

    command = " ".join(args.command).strip()

    if args.execute:
        return run_execute(
            command,
            config,
            save_path=args.save,
            inspect_context=args.context_path,
            inspect_depth=args.depth,
        )

    # Default behavior is dry-run for safety.
    return run_dry_run(command, config, save_path=args.save)


__all__: Iterable[str] = (
    "build_parser",
    "main",
    "run_dry_run",
    "run_execute",
    "run_inspect",
)


if __name__ == "__main__":
    raise SystemExit(main())
