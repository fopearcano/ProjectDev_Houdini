"""HTTP client for a local LM Studio server.

LM Studio exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint.
This module provides a small, dependency-light client around that endpoint
so the rest of ProjectDev can stay backend-agnostic. The same protocol can
later be implemented for OpenAI- or Anthropic-hosted models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import httpx

from app.config import Config

DEFAULT_TIMEOUT = 60.0
ChatMessage = Mapping[str, str]


class LMStudioError(RuntimeError):
    """Base error for any LM Studio client failure."""


class LMStudioUnreachableError(LMStudioError):
    """The LM Studio server could not be reached."""


class LMStudioMissingModelError(LMStudioError):
    """No model is configured or the configured model is unknown to the server."""


class LMStudioMalformedResponseError(LMStudioError):
    """The server returned a response that did not match the expected schema."""


@dataclass(frozen=True)
class LMStudioClient:
    """Thin synchronous client for LM Studio's chat completions endpoint."""

    base_url: str
    model: str
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_config(cls, config: Config, *, timeout: float = DEFAULT_TIMEOUT) -> "LMStudioClient":
        return cls(
            base_url=config.lmstudio_base_url,
            model=config.lmstudio_model,
            timeout=timeout,
        )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> str:
        """Send ``messages`` to the server and return the assistant text.

        ``messages`` must be an iterable of OpenAI-style chat messages, e.g.
        ``[{"role": "user", "content": "hi"}]``.
        """

        if not self.model:
            raise LMStudioMissingModelError(
                "No LMSTUDIO_MODEL configured. Set it in .env or pass model="
                "<id> when constructing LMStudioClient."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _normalize_messages(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if extra_payload:
            payload.update(extra_payload)

        url = self._endpoint("chat/completions")

        try:
            response = httpx.post(url, json=payload, timeout=self.timeout)
        except httpx.ConnectError as exc:
            raise LMStudioUnreachableError(
                f"Could not connect to LM Studio at {self.base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LMStudioUnreachableError(
                f"LM Studio at {self.base_url} did not respond within "
                f"{self.timeout}s: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise LMStudioUnreachableError(
                f"HTTP error talking to LM Studio at {self.base_url}: {exc}"
            ) from exc

        return _parse_chat_response(response, model=self.model)

    def _endpoint(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


def _normalize_messages(messages: Iterable[ChatMessage]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if not role or content is None:
            raise ValueError(
                f"Message at index {index} is missing 'role' or 'content': {message!r}"
            )
        normalized.append({"role": str(role), "content": str(content)})
    if not normalized:
        raise ValueError("messages must contain at least one entry")
    return normalized


def _parse_chat_response(response: httpx.Response, *, model: str) -> str:
    if response.status_code == 404:
        raise LMStudioMissingModelError(
            f"LM Studio returned 404 for model {model!r}. "
            "Make sure the model is loaded in LM Studio."
        )

    if response.status_code >= 400:
        body = _safe_body(response)
        if _looks_like_missing_model(body):
            raise LMStudioMissingModelError(
                f"LM Studio rejected model {model!r}: {body}"
            )
        raise LMStudioError(
            f"LM Studio returned HTTP {response.status_code}: {body}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise LMStudioMalformedResponseError(
            f"LM Studio response was not valid JSON: {exc}"
        ) from exc

    try:
        choices = data["choices"]
        first = choices[0]
        message = first["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LMStudioMalformedResponseError(
            f"Unexpected response shape from LM Studio: {data!r}"
        ) from exc

    if not isinstance(content, str):
        raise LMStudioMalformedResponseError(
            f"Expected string content, got {type(content).__name__}: {content!r}"
        )

    return content


def _safe_body(response: httpx.Response) -> str:
    try:
        return response.text
    except Exception:  # pragma: no cover - defensive
        return "<unreadable body>"


def _looks_like_missing_model(body: str) -> bool:
    lowered = body.lower()
    return "model" in lowered and (
        "not found" in lowered or "unknown" in lowered or "no such" in lowered
    )
