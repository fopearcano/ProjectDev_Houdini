"""Tests for :mod:`app.llm.planner`."""

from __future__ import annotations

from typing import Sequence

import pytest

from app.config import Config
from app.houdini.schemas import ProjectPlan
from app.llm.lmstudio_client import ChatMessage
from app.llm.planner import DEFAULT_SYSTEM_PROMPT, build_messages, plan_user_request


class FakeClient:
    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.calls: list[Sequence[ChatMessage]] = []

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        return self.response


def test_build_messages_includes_system_and_user_roles() -> None:
    messages = build_messages("make a sphere")

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert messages[1] == {"role": "user", "content": "make a sphere"}


def test_build_messages_strips_whitespace() -> None:
    messages = build_messages("   make a sphere\n")
    assert messages[1]["content"] == "make a sphere"


def test_build_messages_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError):
        build_messages("   ")


def test_plan_user_request_uses_injected_client() -> None:
    fake = FakeClient(response="I would create a sphere.")

    result = plan_user_request("make a sphere", client=fake)

    assert result == "I would create a sphere."
    assert len(fake.calls) == 1
    sent = fake.calls[0]
    assert sent[0]["role"] == "system"
    assert sent[1] == {"role": "user", "content": "make a sphere"}


def test_plan_user_request_supports_custom_system_prompt() -> None:
    fake = FakeClient()
    plan_user_request("hello", client=fake, system_prompt="be terse")

    assert fake.calls[0][0]["content"] == "be terse"


def test_plan_user_request_builds_client_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class StubClient:
        def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
            captured["base_url"] = base_url
            captured["model"] = model

        @classmethod
        def from_config(cls, config: Config, *, timeout: float = 60.0) -> "StubClient":
            return cls(base_url=config.lmstudio_base_url, model=config.lmstudio_model)

        def chat(self, messages: Sequence[ChatMessage]) -> str:
            captured["messages"] = list(messages)
            return "stubbed"

    monkeypatch.setattr("app.llm.planner.LMStudioClient", StubClient)

    config = Config(lmstudio_base_url="http://x/v1", lmstudio_model="m")
    result = plan_user_request("make a sphere", config=config)

    assert result == "stubbed"
    assert captured["base_url"] == "http://x/v1"
    assert captured["model"] == "m"
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[1] == {"role": "user", "content": "make a sphere"}


# --- Skill routing ---------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "make a rock",
        "Make a procedural rock",
        "create a stone for me",
        "build a boulder",
        "I want some rocks",
        "carve some stones",
        "design a cluster of boulders",
    ],
)
def test_rock_prompts_route_to_procedural_rock_skill(prompt: str) -> None:
    fake = FakeClient(response="should not be called")

    result = plan_user_request(prompt, client=fake)

    assert isinstance(result, ProjectPlan)
    assert fake.calls == []
    # The plan must include the procedural rock geo container.
    geo_actions = [
        a for a in result.actions
        if getattr(a, "node_name", None) == "procedural_rock_geo"
    ]
    assert len(geo_actions) == 1
    assert result.user_goal == prompt


@pytest.mark.parametrize(
    "prompt",
    [
        "rocky road",        # 'rock' inside a longer word, not whole-word
        "make a sphere",
        "build a procedural mountain",
        "stoneware ceramics",
    ],
)
def test_non_rock_prompts_fall_through_to_llm(prompt: str) -> None:
    fake = FakeClient(response="LLM response")

    result = plan_user_request(prompt, client=fake)

    assert result == "LLM response"
    assert len(fake.calls) == 1


def test_rock_route_skips_llm_client_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skill matches must not even attempt to build an LMStudioClient."""

    def boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("LMStudioClient should not be constructed")

    monkeypatch.setattr("app.llm.planner.LMStudioClient", boom)

    result = plan_user_request("make a procedural rock")
    assert isinstance(result, ProjectPlan)
