"""Command-line entry point for ProjectDev."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from app.config import Config
from app.houdini.bridge import HoudiniBridge, HoudiniBridgeError
from app.houdini.inspector import format_scene_summary
from app.houdini.schemas import InspectSceneAction, ProjectPlan
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
        help="Print the loaded configuration before sending the prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Echo the command and exit without contacting the LLM.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run an inspect_scene action through Houdini and print a "
        "readable scene summary.",
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


def run(
    command: str,
    config: Config,
    *,
    show_config: bool = False,
    dry_run: bool = False,
) -> int:
    """Handle a single command: send it to the planner and print the response."""

    print(f"Received command: {command}")
    if show_config or dry_run:
        print(config.describe())
    if dry_run:
        return 0

    try:
        plan = plan_user_request(command, config=config)
    except LMStudioError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 1
    except PlannerError as exc:
        print(f"Planner error: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"Plan ({len(plan.actions)} actions):")
    print(plan.model_dump_json(indent=2))
    return 0


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

    if args.inspect:
        return run_inspect(
            config,
            context_path=args.context_path,
            depth=args.depth,
        )

    if not args.command:
        parser.error("a command is required unless --inspect is given")

    command = " ".join(args.command).strip()
    return run(
        command,
        config,
        show_config=args.show_config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
