"""Lightweight planning wrapper around an LLM client.

For now the planner just sends the user's prompt to the configured LM
Studio model with a short system prompt. Later it will be responsible
for turning natural language into structured Houdini actions.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from app.config import Config
from app.llm.lmstudio_client import ChatMessage, LMStudioClient

DEFAULT_SYSTEM_PROMPT = (
    "You are ProjectDev, an assistant that helps a 3D artist control "
    "SideFX Houdini through safe, structured actions. For now, respond "
    "in plain English describing what you would do. Do not invent file "
    "paths or run code."
)


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


def plan_user_request(
    user_prompt: str,
    *,
    client: _ChatLLM | None = None,
    config: Config | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Send ``user_prompt`` through the LLM and return the raw response.

    A ``client`` can be injected for tests; otherwise an
    :class:`LMStudioClient` is built from the supplied (or freshly
    loaded) :class:`Config`.
    """

    if client is None:
        cfg = config if config is not None else Config.from_env()
        client = LMStudioClient.from_config(cfg)

    messages = build_messages(user_prompt, system_prompt=system_prompt)
    return client.chat(messages)
