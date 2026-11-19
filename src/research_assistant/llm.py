"""Chat model factory for OpenRouter (OpenAI-compatible API)."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from research_assistant.config import Settings, get_settings

logger = logging.getLogger(__name__)

Role = Literal["main", "judge"]


class LLMConfigError(RuntimeError):
    """Raised when the LLM cannot be configured (e.g. missing API key)."""


def _build(model: str, settings: Settings) -> ChatOpenAI:
    key = settings.openrouter_api_key
    if key is None or not key.get_secret_value():
        raise LLMConfigError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
    return ChatOpenAI(
        model=model,
        base_url=settings.llm.base_url,
        api_key=key,
        timeout=settings.llm.timeout_s,
        max_retries=settings.llm.max_retries,
        temperature=settings.llm.temperature,
    )


def get_chat_model(
    role: Role, settings: Settings | None = None
) -> Runnable[LanguageModelInput, BaseMessage]:
    """Return the chat model for `role`, falling back to the configured fallback model."""
    settings = settings or get_settings()
    model_name = getattr(settings.models, role)
    primary = _build(model_name, settings)
    fallback_name = settings.models.fallback
    if fallback_name == model_name:
        return primary
    logger.debug("LLM role=%s model=%s fallback=%s", role, model_name, fallback_name)
    return primary.with_fallbacks([_build(fallback_name, settings)])
