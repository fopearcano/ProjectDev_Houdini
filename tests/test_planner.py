"""Tests for :mod:`app.llm.planner`."""

from __future__ import annotations

import json
from typing import Sequence

import pytest

from app.config import Config
from app.houdini.schemas import ProjectPlan
from app.llm.lmstudio_client import ChatMessage
from app.llm.planner import (
    PROMPT_PATH,
    PlannerError,
    build_messages,
    load_system_prompt,
    plan_user_request,
    plan_with_llm,
)


@pytest.fixture(autouse=True)
def _reset_prompt_cache() -> None:
    load_system_prompt.cache_clear()


VALID_PLAN_JSON = json.dumps(
    {
        "user_goal": "make a sphere",
        "actions": [
            {
                "action_type": "create_node",
                "context_path": "/obj",
                "node_type": "geo",
                "node_name": "sphere_geo",
            }
        ],
        "notes": [],
    }
)


def _alt_plan_json(goal: str = "alt") -> str:
    return json.dumps(
        {
            "user_goal": goal,
            "actions": [
                {"action_type": "inspect_scene", "context_path": "/obj"}
            ],
        }
    )


class FakeClient:
    def __init__(self, responses: list[str] | str) -> None:
        if isinstance(responses, str):
            responses = [responses]
        self._responses = list(responses)
        self.calls: list[Sequence[ChatMessage]] = []

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("FakeClient received an unexpected extra call")
        return self._responses.pop(0)


# --- system prompt ---------------------------------------------------------


def test_system_prompt_file_exists_and_has_key_directives() -> None:
    text = load_system_prompt()
    assert text  # non-empty
    # Hard rules the planner relies on.
    assert "JSON" in text
    assert "Reply with the JSON object only" in text
    for action in (
        "create_node",
        "set_parameter",
        "connect_nodes",
        "delete_node",
        "layout_children",
        "save_file",
        "inspect_scene",
    ):
        assert action in text


def test_load_system_prompt_caches() -> None:
    a = load_system_prompt()
    b = load_system_prompt()
    assert a is b  # same object thanks to lru_cache


def test_load_system_prompt_reads_from_disk(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    custom = tmp_path / "prompt.md"
    custom.write_text("custom prompt body", encoding="utf-8")

    monkeypatch.setattr("app.llm.planner.PROMPT_PATH", custom)
    load_system_prompt.cache_clear()

    assert load_system_prompt() == "custom prompt body"


def test_prompt_path_points_to_real_file() -> None:
    assert PROMPT_PATH.is_file()


# --- build_messages --------------------------------------------------------


def test_build_messages_uses_loaded_prompt_by_default() -> None:
    messages = build_messages("make a sphere")

    assert messages[0]["role"] == "system"
    assert "ProjectPlan" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "make a sphere"}


def test_build_messages_strips_whitespace() -> None:
    messages = build_messages("   make a sphere\n")
    assert messages[1]["content"] == "make a sphere"


def test_build_messages_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError):
        build_messages("   ")


# --- plan_with_llm ---------------------------------------------------------


def test_plan_with_llm_parses_clean_json() -> None:
    fake = FakeClient(VALID_PLAN_JSON)

    plan = plan_with_llm("make a sphere", fake)

    assert isinstance(plan, ProjectPlan)
    assert plan.user_goal == "make a sphere"
    assert len(fake.calls) == 1
    assert fake.calls[0][1] == {"role": "user", "content": "make a sphere"}


def test_plan_with_llm_strips_code_fences() -> None:
    fenced = f"```json\n{VALID_PLAN_JSON}\n```"
    fake = FakeClient(fenced)
    plan = plan_with_llm("make a sphere", fake)
    assert plan.user_goal == "make a sphere"


def test_plan_with_llm_strips_prose_around_json() -> None:
    wrapped = f"Sure! Here's the plan:\n{VALID_PLAN_JSON}\nLet me know if anything is off."
    fake = FakeClient(wrapped)
    plan = plan_with_llm("make a sphere", fake)
    assert plan.user_goal == "make a sphere"


def test_plan_with_llm_repairs_after_first_failure() -> None:
    fake = FakeClient(["this is not JSON at all", VALID_PLAN_JSON])

    plan = plan_with_llm("make a sphere", fake)

    assert plan.user_goal == "make a sphere"
    assert len(fake.calls) == 2

    # The repair turn should include the original prompt, the bad
    # response, and a clear instruction.
    repair_messages = fake.calls[1]
    roles = [m["role"] for m in repair_messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert repair_messages[1]["content"] == "make a sphere"
    assert repair_messages[2]["content"] == "this is not JSON at all"
    repair_text = repair_messages[3]["content"]
    assert "previous reply" in repair_text.lower()
    assert "ProjectPlan" in repair_text
    assert "JSON" in repair_text


def test_plan_with_llm_repairs_when_validation_fails() -> None:
    bad_plan = json.dumps(
        {
            "user_goal": "x",
            "actions": [{"action_type": "exec_python", "code": "rm -rf /"}],
        }
    )
    fake = FakeClient([bad_plan, VALID_PLAN_JSON])
    plan = plan_with_llm("make a sphere", fake)
    assert plan.user_goal == "make a sphere"
    assert len(fake.calls) == 2


def test_plan_with_llm_raises_planner_error_after_two_failures() -> None:
    fake = FakeClient(["junk", "still junk"])

    with pytest.raises(PlannerError) as info:
        plan_with_llm("make a sphere", fake)

    msg = str(info.value)
    assert "First error" in msg
    assert "Second error" in msg
    assert len(fake.calls) == 2


def test_plan_with_llm_raises_when_action_type_invalid_after_repair() -> None:
    bad = json.dumps(
        {
            "user_goal": "x",
            "actions": [{"action_type": "exec_python", "code": "import os"}],
        }
    )
    fake = FakeClient([bad, bad])
    with pytest.raises(PlannerError):
        plan_with_llm("make a sphere", fake)


def test_plan_with_llm_supports_custom_system_prompt() -> None:
    fake = FakeClient(VALID_PLAN_JSON)
    plan_with_llm("make a sphere", fake, system_prompt="be terse")
    assert fake.calls[0][0]["content"] == "be terse"


def test_plan_with_llm_rejects_empty_prompt() -> None:
    fake = FakeClient(VALID_PLAN_JSON)
    with pytest.raises(ValueError):
        plan_with_llm("   ", fake)


# --- plan_user_request -----------------------------------------------------


def test_plan_user_request_routes_llm_path() -> None:
    fake = FakeClient(VALID_PLAN_JSON)

    plan = plan_user_request("make a sphere", client=fake)

    assert isinstance(plan, ProjectPlan)
    assert plan.user_goal == "make a sphere"


def test_plan_user_request_supports_custom_system_prompt() -> None:
    fake = FakeClient(VALID_PLAN_JSON)
    plan_user_request("make a sphere", client=fake, system_prompt="be terse")
    assert fake.calls[0][0]["content"] == "be terse"


def test_plan_user_request_builds_client_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            return VALID_PLAN_JSON

    monkeypatch.setattr("app.llm.planner.LMStudioClient", StubClient)

    config = Config(lmstudio_base_url="http://x/v1", lmstudio_model="m")
    plan = plan_user_request("make a sphere", config=config)

    assert isinstance(plan, ProjectPlan)
    assert captured["base_url"] == "http://x/v1"
    assert captured["model"] == "m"


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
    fake = FakeClient(["should not be called"])

    result = plan_user_request(prompt, client=fake)

    assert isinstance(result, ProjectPlan)
    assert fake.calls == []
    geo_actions = [
        a for a in result.actions
        if getattr(a, "node_name", None) == "procedural_rock_geo"
    ]
    assert len(geo_actions) == 1
    assert result.user_goal == prompt


@pytest.mark.parametrize(
    "prompt",
    [
        "rocky road",
        "make a sphere",
        "build a procedural mountain",
        "stoneware ceramics",
    ],
)
def test_non_rock_prompts_fall_through_to_llm(prompt: str) -> None:
    fake = FakeClient(_alt_plan_json(prompt))

    plan = plan_user_request(prompt, client=fake)

    assert isinstance(plan, ProjectPlan)
    assert plan.user_goal == prompt
    assert len(fake.calls) == 1


def test_rock_route_skips_llm_client_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("LMStudioClient should not be constructed")

    monkeypatch.setattr("app.llm.planner.LMStudioClient", boom)

    result = plan_user_request("make a procedural rock")
    assert isinstance(result, ProjectPlan)
