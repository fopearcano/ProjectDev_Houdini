"""Lightweight planning wrapper around an LLM client.

The planner first looks for a built-in skill that recognises the
prompt (currently only the procedural rock skill). If no skill matches,
it falls back to sending the prompt to the configured LM Studio model.
Skill matches return a structured :class:`ProjectPlan`; LLM fallbacks
return the raw assistant text.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence, Union

from app.config import Config
from app.houdini.schemas import ProjectPlan
from app.llm.lmstudio_client import ChatMessage, LMStudioClient
from app.skills import procedural_rock

DEFAULT_SYSTEM_PROMPT = (
    "You are ProjectDev, an assistant that helps a 3D artist control "
    "SideFX Houdini through safe, structured actions. For now, respond "
    "in plain English describing what you would do. Do not invent file "
    "paths or run code."
)

PlannerResult = Union[ProjectPlan, str]

_ROCK_PROMPT_RE = re.compile(r"\b(rocks?|stones?|boulders?)\b", re.IGNORECASE)


class _ChatLLM(Protocol):
    def chat(self, messages: Sequence[ChatMessage]) -> str: ...


def build_messages(
    user_prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[ChatMessage]:
    user_prompt = user_prompt.strip()
    if not user_prompt:
        raise ValueError("user_prompt must not be empty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _match_skill(user_prompt: str) -> ProjectPlan | None:
    if _ROCK_PROMPT_RE.search(user_prompt):
        return procedural_rock.build_plan(user_prompt)
    return None


def plan_user_request(
    user_prompt: str,
    *,
    client: _ChatLLM | None = None,
    config: Config | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> PlannerResult:
    """Plan a response to ``user_prompt``.

    If a built-in skill recognises the prompt, returns a validated
    :class:`ProjectPlan`. Otherwise sends the prompt through the LLM
    and returns the raw assistant text.
    """

    skill_plan = _match_skill(user_prompt)
    if skill_plan is not None:
        return skill_plan

    if client is None:
        cfg = config if config is not None else Config.from_env()
        client = LMStudioClient.from_config(cfg)

    messages = build_messages(user_prompt, system_prompt=system_prompt)
    return client.chat(messages)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "PlannerResult",
    "build_messages",
    "plan_user_request",
]
