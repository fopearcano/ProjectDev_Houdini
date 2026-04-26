"""Planning wrapper around an LLM client.

The planner first looks for a built-in skill that recognises the prompt
(currently only the procedural rock skill). If no skill matches, it
sends the request to the configured LM Studio model with the
``prompts/system_houdini_planner.md`` system prompt and parses the
response as a :class:`ProjectPlan`. If the first response cannot be
parsed, a single repair turn is attempted before giving up.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import ValidationError

from app.config import Config
from app.houdini.schemas import ProjectPlan
from app.llm.lmstudio_client import ChatMessage, LMStudioClient
from app.skills import procedural_rock

PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts"
    / "system_houdini_planner.md"
)

PlannerResult = ProjectPlan

_ROCK_PROMPT_RE = re.compile(r"\b(rocks?|stones?|boulders?)\b", re.IGNORECASE)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class PlannerError(RuntimeError):
    """The LLM could not be coaxed into producing a valid ProjectPlan."""


class _ChatLLM(Protocol):
    def chat(self, messages: Sequence[ChatMessage]) -> str: ...


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """Read and cache the planner system prompt from disk."""

    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_messages(
    user_prompt: str,
    *,
    system_prompt: str | None = None,
) -> list[ChatMessage]:
    user_prompt = user_prompt.strip()
    if not user_prompt:
        raise ValueError("user_prompt must not be empty")
    return [
        {"role": "system", "content": system_prompt or load_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from ``text``.

    Tries the raw text, then a fenced ```json ... ``` block, then the
    substring between the first ``{`` and last ``}``. Raises ``ValueError``
    with the most recent decode error if none parse.
    """

    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    fence = _FENCED_JSON_RE.search(stripped)
    if fence:
        candidates.append(fence.group(1))

    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError as exc:
            last_error = exc
            continue
        if isinstance(data, dict):
            return data
        last_error = ValueError(f"expected JSON object, got {type(data).__name__}")

    raise ValueError(
        f"Could not extract a JSON object from model output: {last_error}"
    )


def _parse_plan(text: str) -> ProjectPlan:
    data = _extract_json_object(text)
    return ProjectPlan.model_validate(data)


def _build_repair_messages(
    user_prompt: str,
    *,
    system_prompt: str,
    bad_response: str,
    error: str,
) -> list[ChatMessage]:
    repair_instructions = (
        "Your previous reply could not be parsed as a valid ProjectPlan.\n"
        f"Error: {error}\n\n"
        "Reply now with ONLY a single JSON object that matches the "
        "ProjectPlan schema. Do not include prose, Markdown, or code fences."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": bad_response},
        {"role": "user", "content": repair_instructions},
    ]


def plan_with_llm(
    user_prompt: str,
    client: _ChatLLM,
    *,
    system_prompt: str | None = None,
) -> ProjectPlan:
    """Send ``user_prompt`` through the LLM and return a validated plan.

    Performs at most one repair attempt if the first response is not a
    valid :class:`ProjectPlan`. Raises :class:`PlannerError` if both
    attempts fail.
    """

    if not user_prompt.strip():
        raise ValueError("user_prompt must not be empty")

    system_prompt_text = system_prompt or load_system_prompt()
    first_response = client.chat(build_messages(user_prompt, system_prompt=system_prompt_text))

    try:
        return _parse_plan(first_response)
    except (ValueError, ValidationError) as first_error:
        repair_messages = _build_repair_messages(
            user_prompt,
            system_prompt=system_prompt_text,
            bad_response=first_response,
            error=str(first_error),
        )
        repaired_response = client.chat(repair_messages)
        try:
            return _parse_plan(repaired_response)
        except (ValueError, ValidationError) as second_error:
            raise PlannerError(
                "LLM did not produce a valid ProjectPlan after one repair "
                f"attempt. First error: {first_error}. "
                f"Second error: {second_error}."
            ) from second_error


def _match_skill(user_prompt: str) -> ProjectPlan | None:
    if _ROCK_PROMPT_RE.search(user_prompt):
        return procedural_rock.build_plan(user_prompt)
    return None


def plan_user_request(
    user_prompt: str,
    *,
    client: _ChatLLM | None = None,
    config: Config | None = None,
    system_prompt: str | None = None,
) -> ProjectPlan:
    """Plan a response to ``user_prompt``.

    If a built-in skill recognises the prompt, returns its
    :class:`ProjectPlan` directly. Otherwise sends the prompt through
    the LLM (with at most one repair turn) and returns the parsed plan.
    Raises :class:`PlannerError` if the model cannot produce a valid
    plan.
    """

    skill_plan = _match_skill(user_prompt)
    if skill_plan is not None:
        return skill_plan

    if client is None:
        cfg = config if config is not None else Config.from_env()
        client = LMStudioClient.from_config(cfg)

    return plan_with_llm(user_prompt, client, system_prompt=system_prompt)


__all__ = [
    "PROMPT_PATH",
    "PlannerError",
    "PlannerResult",
    "build_messages",
    "load_system_prompt",
    "plan_user_request",
    "plan_with_llm",
]
