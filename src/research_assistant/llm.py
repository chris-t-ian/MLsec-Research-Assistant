"""Chat model factory for OpenRouter (OpenAI-compatible API)."""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, TypeVar

import openai
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from research_assistant.config import Settings, get_settings
from research_assistant.models import TokenUsage

logger = logging.getLogger(__name__)

Role = Literal["main", "judge"]


class LLMConfigError(RuntimeError):
    """Raised when the LLM cannot be configured (e.g. missing API key)."""


class LLMCallError(RuntimeError):
    """An LLM call failed after retries. The message never contains secrets or prompts."""


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


# --------------------------------------------------------------------------- structured output

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    """Returns a Pydantic object for a (system, user) prompt, plus the tokens it used."""

    def invoke_structured(self, schema: type[T], system: str, user: str) -> tuple[T, TokenUsage]:
        """Raise :class:`LLMCallError` on failure."""
        ...


class _ParseFailure(Exception):
    """The model answered, but not with a valid instance of the schema."""


# Transient errors worth retrying ourselves (on top of the client's own retries).
_RETRYABLE = (
    _ParseFailure,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def usage_from_message(message: Any) -> TokenUsage:
    """Token usage reported on an AIMessage (zeros if the provider sent none)."""
    meta = getattr(message, "usage_metadata", None) or {}
    return TokenUsage(
        input_tokens=int(meta.get("input_tokens") or 0),
        output_tokens=int(meta.get("output_tokens") or 0),
        total_tokens=int(meta.get("total_tokens") or 0),
    )


class ChatStructuredLLM:
    """:class:`StructuredLLM` on a LangChain chat model (tool calling, retried with tenacity).

    Timeouts come from the chat model (``llm.timeout_s``). Failed attempts are retried with
    exponential backoff; the final failure is raised as :class:`LLMCallError`.
    """

    def __init__(
        self,
        model: Runnable[LanguageModelInput, BaseMessage],
        max_attempts: int = 3,
        backoff_s: float = 1.0,
    ) -> None:
        self._model = model
        self._max_attempts = max(1, max_attempts)
        self._backoff_s = backoff_s

    def invoke_structured(self, schema: type[T], system: str, user: str) -> tuple[T, TokenUsage]:
        runnable = self._model.with_structured_output(  # type: ignore[attr-defined]
            schema, method="function_calling", include_raw=True
        )
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        usage = TokenUsage()

        def _once() -> T:
            nonlocal usage
            out = runnable.invoke(messages)
            usage = usage + usage_from_message(out.get("raw"))
            parsed = out.get("parsed")
            if out.get("parsing_error") is not None or not isinstance(parsed, schema):
                raise _ParseFailure(schema.__name__)
            return parsed

        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=self._backoff_s, max=20),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=False,
        )
        try:
            result = retrying(_once)
        except RetryError as exc:
            cause = exc.last_attempt.exception()
            name = type(cause).__name__ if cause else "unknown"
            raise LLMCallError(f"LLM call for {schema.__name__} failed ({name})") from cause
        except Exception as exc:  # non-retryable (auth, bad request, ...)
            raise LLMCallError(
                f"LLM call for {schema.__name__} failed ({type(exc).__name__})"
            ) from exc
        return result, usage


def get_structured_llm(role: Role, settings: Settings | None = None) -> ChatStructuredLLM:
    settings = settings or get_settings()
    return ChatStructuredLLM(get_chat_model(role, settings), max_attempts=settings.llm.max_retries)
