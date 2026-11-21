"""Chat model factory for OpenRouter (OpenAI-compatible API)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, TypeVar

import openai
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception,
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
        stream_usage=True,
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


# --------------------------------------------------------------------------- tool calling


class ToolCallingLLM(Protocol):
    """Chat model bound to tools: returns an AIMessage (text and/or tool calls)."""

    def invoke_messages(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Raise :class:`LLMCallError` on failure."""
        ...


class ChatToolCallingLLM:
    """:class:`ToolCallingLLM` on a tool-bound LangChain chat model (tenacity retries)."""

    def __init__(
        self,
        model: Runnable[LanguageModelInput, BaseMessage],
        max_attempts: int = 3,
        backoff_s: float = 1.0,
        model_name: str | None = None,
    ) -> None:
        self._model = model
        self._max_attempts = max(1, max_attempts)
        self._backoff_s = backoff_s
        self.model_name = model_name

    def invoke_messages(self, messages: Sequence[BaseMessage]) -> AIMessage:
        def _once() -> AIMessage:
            out = self._model.invoke(list(messages))
            if not isinstance(out, AIMessage):
                raise _ParseFailure("AIMessage")
            return out

        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=self._backoff_s, max=20),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=False,
        )
        try:
            return retrying(_once)
        except RetryError as exc:
            cause = exc.last_attempt.exception()
            name = type(cause).__name__ if cause else "unknown"
            raise LLMCallError(f"Tool-calling LLM call failed ({name})") from cause
        except Exception as exc:  # non-retryable (auth, bad request, ...)
            raise LLMCallError(f"Tool-calling LLM call failed ({type(exc).__name__})") from exc


def get_tool_calling_llm(
    role: Role, tools: Sequence[Any], settings: Settings | None = None
) -> ChatToolCallingLLM:
    """The ``role`` model (with fallback model) bound to ``tools`` (BaseTools or JSON schemas)."""
    settings = settings or get_settings()
    model_name = getattr(settings.models, role)
    primary: Runnable[LanguageModelInput, BaseMessage] = _build(model_name, settings).bind_tools(
        list(tools)
    )
    fallback_name = settings.models.fallback
    if fallback_name != model_name:
        primary = primary.with_fallbacks([_build(fallback_name, settings).bind_tools(list(tools))])
    return ChatToolCallingLLM(primary, max_attempts=settings.llm.max_retries, model_name=model_name)


# --------------------------------------------------------------------------- streaming


class StreamResult(BaseModel):
    text: str
    usage: TokenUsage = TokenUsage()
    model: str | None = None


class StreamingLLM(Protocol):
    """Streams a text answer for a (system, user) prompt, calling ``on_token`` per chunk."""

    def stream_text(self, system: str, user: str, on_token: Callable[[str], None]) -> StreamResult:
        """Raise :class:`LLMCallError` on failure."""
        ...


class _EmptyResponse(Exception):
    """The model returned no text."""


class _StreamInterrupted(Exception):
    """The stream failed after tokens were already emitted (not retried)."""


def content_text(content: Any) -> str:
    """Text of a message (chunk) content: a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


class ChatStreamingLLM:
    """:class:`StreamingLLM` on a LangChain chat model.

    A call is retried (exponential backoff) only while no token has been emitted yet;
    once tokens reached ``on_token``, a failure is final, so the UI never sees text twice.
    """

    def __init__(
        self,
        model: Runnable[LanguageModelInput, BaseMessage],
        max_attempts: int = 3,
        backoff_s: float = 1.0,
        model_name: str | None = None,
    ) -> None:
        self._model = model
        self._max_attempts = max(1, max_attempts)
        self._backoff_s = backoff_s
        self._model_name = model_name

    def stream_text(self, system: str, user: str, on_token: Callable[[str], None]) -> StreamResult:
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        emitted = False

        def _once() -> StreamResult:
            nonlocal emitted
            parts: list[str] = []
            agg: Any = None
            model: str | None = None
            try:
                for chunk in self._model.stream(messages):
                    agg = chunk if agg is None else agg + chunk
                    # Read per chunk: adding chunks concatenates metadata strings.
                    meta = getattr(chunk, "response_metadata", None) or {}
                    model = model or meta.get("model_name") or meta.get("model")
                    text = content_text(getattr(chunk, "content", ""))
                    if text:
                        emitted = True
                        parts.append(text)
                        on_token(text)
            except Exception as exc:
                if emitted:
                    raise _StreamInterrupted(type(exc).__name__) from exc
                raise
            text = "".join(parts)
            if not text.strip():
                raise _EmptyResponse()
            return StreamResult(
                text=text, usage=usage_from_message(agg), model=model or self._model_name
            )

        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=self._backoff_s, max=20),
            retry=retry_if_exception(
                lambda e: not emitted and isinstance(e, (*_RETRYABLE, _EmptyResponse))
            ),
            reraise=False,
        )
        try:
            return retrying(_once)
        except RetryError as exc:
            cause = exc.last_attempt.exception()
            name = type(cause).__name__ if cause else "unknown"
            raise LLMCallError(f"Streaming LLM call failed ({name})") from cause
        except _StreamInterrupted as exc:
            raise LLMCallError(f"LLM stream interrupted ({exc})") from exc
        except Exception as exc:  # non-retryable (auth, bad request, ...)
            raise LLMCallError(f"Streaming LLM call failed ({type(exc).__name__})") from exc


def get_streaming_llm(role: Role, settings: Settings | None = None) -> ChatStreamingLLM:
    settings = settings or get_settings()
    return ChatStreamingLLM(
        get_chat_model(role, settings),
        max_attempts=settings.llm.max_retries,
        model_name=getattr(settings.models, role),
    )
