"""Tests for :mod:`app.llm.lmstudio_client`."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.config import Config
from app.llm.lmstudio_client import (
    LMStudioClient,
    LMStudioError,
    LMStudioMalformedResponseError,
    LMStudioMissingModelError,
    LMStudioUnreachableError,
)


def _ok_response(content: str = "Hello!", status_code: int = 200) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = {
        "id": "chatcmpl-1",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}}
        ],
    }
    response.text = ""
    return response


def _error_response(status_code: int, body: str) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.text = body
    response.json.side_effect = ValueError("not json")
    return response


def test_from_config_uses_config_values() -> None:
    cfg = Config(
        lmstudio_base_url="http://example.com/v1",
        lmstudio_model="qwen2.5-coder",
    )
    client = LMStudioClient.from_config(cfg, timeout=5.0)

    assert client.base_url == "http://example.com/v1"
    assert client.model == "qwen2.5-coder"
    assert client.timeout == 5.0


def test_chat_posts_expected_payload_and_returns_content() -> None:
    client = LMStudioClient(base_url="http://localhost:1234/v1", model="m")
    response = _ok_response("Sure thing.")

    with patch("app.llm.lmstudio_client.httpx.post", return_value=response) as post:
        result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "Sure thing."
    post.assert_called_once()
    url = post.call_args.args[0]
    payload: dict[str, Any] = post.call_args.kwargs["json"]
    assert url == "http://localhost:1234/v1/chat/completions"
    assert payload["model"] == "m"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert post.call_args.kwargs["timeout"] == 60.0


def test_chat_strips_trailing_slash_in_base_url() -> None:
    client = LMStudioClient(base_url="http://localhost:1234/v1/", model="m")
    response = _ok_response()

    with patch("app.llm.lmstudio_client.httpx.post", return_value=response) as post:
        client.chat([{"role": "user", "content": "hi"}])

    assert post.call_args.args[0] == "http://localhost:1234/v1/chat/completions"


def test_chat_includes_temperature_and_extra_payload() -> None:
    client = LMStudioClient(base_url="http://x/v1", model="m")
    response = _ok_response()

    with patch("app.llm.lmstudio_client.httpx.post", return_value=response) as post:
        client.chat(
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
            extra_payload={"max_tokens": 32},
        )

    payload = post.call_args.kwargs["json"]
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 32


def test_chat_requires_model() -> None:
    client = LMStudioClient(base_url="http://x/v1", model="")

    with pytest.raises(LMStudioMissingModelError):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_requires_messages() -> None:
    client = LMStudioClient(base_url="http://x/v1", model="m")

    with pytest.raises(ValueError):
        client.chat([])


def test_chat_validates_message_shape() -> None:
    client = LMStudioClient(base_url="http://x/v1", model="m")

    with pytest.raises(ValueError):
        client.chat([{"role": "user"}])  # missing content


def test_chat_raises_unreachable_on_connect_error() -> None:
    client = LMStudioClient(base_url="http://localhost:1234/v1", model="m")

    with patch(
        "app.llm.lmstudio_client.httpx.post",
        side_effect=httpx.ConnectError("refused"),
    ):
        with pytest.raises(LMStudioUnreachableError) as info:
            client.chat([{"role": "user", "content": "hi"}])

    assert "localhost:1234" in str(info.value)


def test_chat_raises_unreachable_on_timeout() -> None:
    client = LMStudioClient(base_url="http://x/v1", model="m", timeout=2.0)

    with patch(
        "app.llm.lmstudio_client.httpx.post",
        side_effect=httpx.ReadTimeout("slow"),
    ):
        with pytest.raises(LMStudioUnreachableError) as info:
            client.chat([{"role": "user", "content": "hi"}])

    assert "2.0s" in str(info.value)


def test_chat_raises_missing_model_on_404() -> None:
    client = LMStudioClient(base_url="http://x/v1", model="ghost")
    response = _error_response(404, "model not found")

    with patch("app.llm.lmstudio_client.httpx.post", return_value=response):
        with pytest.raises(LMStudioMissingModelError):
            client.chat([{"role": "user", "content": "hi"}])


def test_chat_detects_missing_model_in_error_body() -> None:
    client = LMStudioClient(base_url="http://x/v1", model="ghost")
    response = _error_response(400, '{"error": "Model ghost is not found"}')

    with patch("app.llm.lmstudio_client.httpx.post", return_value=response):
        with pytest.raises(LMStudioMissingModelError):
            client.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_generic_error_on_other_http_failures() -> None:
    client = LMStudioClient(base_url="http://x/v1", model="m")
    response = _error_response(500, "internal error")

    with patch("app.llm.lmstudio_client.httpx.post", return_value=response):
        with pytest.raises(LMStudioError) as info:
            client.chat([{"role": "user", "content": "hi"}])

    assert not isinstance(info.value, LMStudioMissingModelError)
    assert "500" in str(info.value)


def test_chat_raises_malformed_when_not_json() -> None:
    client = LMStudioClient(base_url="http://x/v1", model="m")
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.side_effect = ValueError("bad json")
    response.text = "not json"

    with patch("app.llm.lmstudio_client.httpx.post", return_value=response):
        with pytest.raises(LMStudioMalformedResponseError):
            client.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_malformed_when_choices_missing() -> None:
    client = LMStudioClient(base_url="http://x/v1", model="m")
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {"id": "x"}
    response.text = ""

    with patch("app.llm.lmstudio_client.httpx.post", return_value=response):
        with pytest.raises(LMStudioMalformedResponseError):
            client.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_malformed_when_content_not_string() -> None:
    client = LMStudioClient(base_url="http://x/v1", model="m")
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": 123}}]
    }
    response.text = ""

    with patch("app.llm.lmstudio_client.httpx.post", return_value=response):
        with pytest.raises(LMStudioMalformedResponseError):
            client.chat([{"role": "user", "content": "hi"}])
