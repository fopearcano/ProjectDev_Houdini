"""LLM clients and prompt orchestration.

This package hosts wrappers for local (LM Studio) and, later, hosted
(OpenAI, Anthropic) language model backends, along with the prompt
templates and planning logic that translate user requests into
structured Houdini actions.
"""

from app.llm.lmstudio_client import (
    LMStudioClient,
    LMStudioError,
    LMStudioMalformedResponseError,
    LMStudioMissingModelError,
    LMStudioUnreachableError,
)

__all__ = [
    "LMStudioClient",
    "LMStudioError",
    "LMStudioMalformedResponseError",
    "LMStudioMissingModelError",
    "LMStudioUnreachableError",
]
