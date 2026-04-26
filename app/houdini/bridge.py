"""Run a validated ProjectPlan through Houdini's ``hython`` interpreter.

The bridge writes the script produced by :mod:`app.houdini.executor` to a
temporary file, invokes ``hython`` via :mod:`subprocess`, and parses the
sentinel-bracketed JSON payload back into a structured
:class:`ExecutionResult`. The actual ``subprocess.run`` call is
injectable so tests can stand in without a Houdini install.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from app.config import Config
from app.houdini.executor import (
    RESULT_BEGIN,
    RESULT_END,
    UnsupportedActionError,
    build_script,
)
from app.houdini.schemas import ProjectPlan

DEFAULT_TIMEOUT = 120.0
SubprocessRunner = Callable[..., "subprocess.CompletedProcess[str]"]


class HoudiniBridgeError(RuntimeError):
    """Base class for bridge-level failures (process or environment)."""


class HythonNotConfiguredError(HoudiniBridgeError):
    """``HOUDINI_HYTHON_PATH`` is empty or unset."""


class HythonNotFoundError(HoudiniBridgeError):
    """The configured ``hython`` path does not exist or is not executable."""


class HoudiniProcessError(HoudiniBridgeError):
    """The ``hython`` process exited abnormally or produced no result."""


@dataclass(frozen=True)
class ActionOutcome:
    """Result of executing a single action inside hython."""

    index: int
    action_type: str
    success: bool
    data: Mapping[str, Any] | None = None
    error: str | None = None
    traceback: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    """Aggregate result returned by :meth:`HoudiniBridge.execute`."""

    success: bool
    returncode: int
    stdout: str
    stderr: str
    actions: list[ActionOutcome] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class HoudiniBridge:
    """Execute a :class:`ProjectPlan` by shelling out to ``hython``."""

    hython_path: str
    timeout: float = DEFAULT_TIMEOUT
    runner: SubprocessRunner = subprocess.run

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        runner: SubprocessRunner | None = None,
    ) -> "HoudiniBridge":
        if not config.houdini_hython_path:
            raise HythonNotConfiguredError(
                "HOUDINI_HYTHON_PATH is not set. Set it in your .env "
                "to the absolute path of the hython executable."
            )
        return cls(
            hython_path=config.houdini_hython_path,
            timeout=timeout,
            runner=runner if runner is not None else subprocess.run,
        )

    def execute(self, plan: ProjectPlan) -> ExecutionResult:
        """Build the runner script, invoke hython, and parse the result."""

        script_source = build_script(plan)
        self._verify_hython()

        script_path = _write_temp_script(script_source)
        try:
            completed = self._invoke(script_path)
        finally:
            _safe_unlink(script_path)

        return _parse_completed(completed)

    def _verify_hython(self) -> None:
        path = Path(self.hython_path)
        if path.is_file() and os.access(path, os.X_OK):
            return
        if shutil.which(self.hython_path):
            return
        raise HythonNotFoundError(
            f"hython not found or not executable at {self.hython_path!r}"
        )

    def _invoke(self, script_path: Path) -> "subprocess.CompletedProcess[str]":
        try:
            return self.runner(
                [self.hython_path, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HoudiniProcessError(
                f"hython did not finish within {self.timeout}s"
            ) from exc
        except FileNotFoundError as exc:
            raise HythonNotFoundError(
                f"Failed to launch hython at {self.hython_path!r}: {exc}"
            ) from exc
        except OSError as exc:
            raise HoudiniProcessError(
                f"Failed to launch hython: {exc}"
            ) from exc


def _write_temp_script(source: str) -> Path:
    fd, name = tempfile.mkstemp(prefix="projectdev_", suffix=".py", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(source)
    except Exception:
        _safe_unlink(Path(name))
        raise
    return Path(name)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:  # pragma: no cover - best-effort cleanup
        pass


def _parse_completed(
    completed: "subprocess.CompletedProcess[str]",
) -> ExecutionResult:
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    returncode = completed.returncode

    payload = _extract_payload(stdout)

    if payload is None:
        error = (
            "no result payload found in hython stdout"
            if returncode == 0
            else f"hython exited with code {returncode}"
        )
        return ExecutionResult(
            success=False,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            actions=[],
            error=error,
        )

    fatal = payload.get("fatal")
    raw_results = payload.get("results") or []
    actions = [_to_outcome(item) for item in raw_results]

    success = (
        returncode == 0
        and fatal is None
        and all(outcome.success for outcome in actions)
    )

    error: str | None = None
    if fatal is not None:
        error = str(fatal)
    elif returncode != 0:
        error = f"hython exited with code {returncode}"

    return ExecutionResult(
        success=success,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        actions=actions,
        error=error,
    )


def _extract_payload(stdout: str) -> dict[str, Any] | None:
    begin = stdout.rfind(RESULT_BEGIN)
    if begin == -1:
        return None
    end = stdout.find(RESULT_END, begin)
    if end == -1:
        return None
    blob = stdout[begin + len(RESULT_BEGIN) : end].strip()
    try:
        data = json.loads(blob)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _to_outcome(item: Mapping[str, Any]) -> ActionOutcome:
    return ActionOutcome(
        index=int(item.get("index", -1)),
        action_type=str(item.get("action_type", "")),
        success=bool(item.get("success", False)),
        data=item.get("data") if isinstance(item.get("data"), dict) else None,
        error=item.get("error"),
        traceback=item.get("traceback"),
    )


__all__ = [
    "ActionOutcome",
    "ExecutionResult",
    "HoudiniBridge",
    "HoudiniBridgeError",
    "HoudiniProcessError",
    "HythonNotConfiguredError",
    "HythonNotFoundError",
    "UnsupportedActionError",
]
