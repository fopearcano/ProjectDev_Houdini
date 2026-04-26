"""Tests for :mod:`app.config`."""

from __future__ import annotations

from app.config import DEFAULT_LMSTUDIO_BASE_URL, Config


def test_from_env_uses_defaults_when_empty() -> None:
    config = Config.from_env(env={})

    assert config.lmstudio_base_url == DEFAULT_LMSTUDIO_BASE_URL
    assert config.lmstudio_model == ""
    assert config.openai_api_key == ""
    assert config.anthropic_api_key == ""
    assert config.houdini_hython_path == ""


def test_from_env_reads_all_known_keys() -> None:
    env = {
        "LMSTUDIO_BASE_URL": "http://example.com/v1",
        "LMSTUDIO_MODEL": "qwen2.5-coder",
        "OPENAI_API_KEY": "sk-openai-123456",
        "ANTHROPIC_API_KEY": "sk-ant-abcdef",
        "HOUDINI_HYTHON_PATH": "/opt/houdini/bin/hython",
    }

    config = Config.from_env(env=env)

    assert config.lmstudio_base_url == "http://example.com/v1"
    assert config.lmstudio_model == "qwen2.5-coder"
    assert config.openai_api_key == "sk-openai-123456"
    assert config.anthropic_api_key == "sk-ant-abcdef"
    assert config.houdini_hython_path == "/opt/houdini/bin/hython"


def test_from_env_falls_back_to_default_when_base_url_blank() -> None:
    config = Config.from_env(env={"LMSTUDIO_BASE_URL": ""})

    assert config.lmstudio_base_url == DEFAULT_LMSTUDIO_BASE_URL


def test_describe_masks_secrets() -> None:
    config = Config(
        openai_api_key="sk-openai-123456",
        anthropic_api_key="sk-ant-abcdef",
    )

    text = config.describe()

    assert "sk-openai-123456" not in text
    assert "sk-ant-abcdef" not in text
    assert "***" in text


def test_describe_marks_unset_fields() -> None:
    config = Config()

    text = config.describe()

    assert "LMSTUDIO_MODEL      = <unset>" in text
    assert "HOUDINI_HYTHON_PATH = <unset>" in text


def test_config_is_frozen() -> None:
    config = Config()
    try:
        config.lmstudio_model = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Config should be immutable")
