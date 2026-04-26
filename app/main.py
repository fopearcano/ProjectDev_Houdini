"""Command-line entry point for ProjectDev."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from app.config import Config
from app.houdini.schemas import ProjectPlan
from app.llm.lmstudio_client import LMStudioError
from app.llm.planner import plan_user_request


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
        nargs="+",
        help="Natural language command to execute (quoted).",
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
    return parser


def run(
    command: str,
    config: Config,
    *,
    show_config: bool = False,
    dry_run: bool = False,
) -> int:
    """Handle a single command: send it to the LLM and print the response."""

    print(f"Received command: {command}")
    if show_config or dry_run:
        print(config.describe())
    if dry_run:
        return 0

    try:
        response = plan_user_request(command, config=config)
    except LMStudioError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 1

    print()
    if isinstance(response, ProjectPlan):
        print(f"Plan ({len(response.actions)} actions):")
        print(response.model_dump_json(indent=2))
    else:
        print("Assistant:")
        print(response)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = " ".join(args.command).strip()

    config = Config.from_env()
    return run(
        command,
        config,
        show_config=args.show_config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
