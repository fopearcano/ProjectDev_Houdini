"""Environment-driven configuration for ProjectDev."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a declared dependency
    load_dotenv = None  # type: ignore[assignment]


DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"


@dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from environment variables."""

    lmstudio_base_url: str = DEFAULT_LMSTUDIO_BASE_URL
    lmstudio_model: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    houdini_hython_path: str = ""

    extra: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        dotenv_path: str | os.PathLike[str] | None = None,
    ) -> "Config":
        """Build a :class:`Config` from a mapping (defaults to ``os.environ``).

        If ``env`` is not provided and ``python-dotenv`` is installed, a local
        ``.env`` file (or the file at ``dotenv_path``) is loaded into
        ``os.environ`` before reading values.
        """

        if env is None:
            if load_dotenv is not None:
                if dotenv_path is not None:
                    load_dotenv(dotenv_path=Path(dotenv_path), override=False)
                else:
                    load_dotenv(override=False)
            env = os.environ

        return cls(
            lmstudio_base_url=env.get("LMSTUDIO_BASE_URL", DEFAULT_LMSTUDIO_BASE_URL)
            or DEFAULT_LMSTUDIO_BASE_URL,
            lmstudio_model=env.get("LMSTUDIO_MODEL", ""),
            openai_api_key=env.get("OPENAI_API_KEY", ""),
            anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
            houdini_hython_path=env.get("HOUDINI_HYTHON_PATH", ""),
        )

    def describe(self) -> str:
        """Return a human-readable, secret-safe summary of the configuration."""

        def mask(value: str) -> str:
            if not value:
                return "<unset>"
            if len(value) <= 4:
                return "***"
            return f"{value[:2]}***{value[-2:]}"

        return (
            "ProjectDev configuration:\n"
            f"  LMSTUDIO_BASE_URL   = {self.lmstudio_base_url}\n"
            f"  LMSTUDIO_MODEL      = {self.lmstudio_model or '<unset>'}\n"
            f"  OPENAI_API_KEY      = {mask(self.openai_api_key)}\n"
            f"  ANTHROPIC_API_KEY   = {mask(self.anthropic_api_key)}\n"
            f"  HOUDINI_HYTHON_PATH = {self.houdini_hython_path or '<unset>'}"
        )
