"""Command-line entry point for ProjectDev."""

from __future__ import annotations

import argparse
from typing import Sequence

from app.config import Config


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
    return parser


def run(command: str, config: Config) -> int:
    """Handle a single command. For now this only echoes the input."""

    print(f"Received command: {command}")
    print(config.describe())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = " ".join(args.command).strip()

    config = Config.from_env()
    return run(command, config)


if __name__ == "__main__":
    raise SystemExit(main())
