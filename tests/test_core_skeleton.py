"""Unit tests for config, llm factory, events and models (no network)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableWithFallbacks
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, ValidationError

from research_assistant.config import SETTINGS_YAML, Settings
from research_assistant.events import EventSink, NullSink, RecordingSink
from research_assistant.llm import (
    LLMConfigError,
    get_chat_model,
    get_tool_calling_llm,
    usage_from_message,
)
from research_assistant.models import Chunk


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return Settings(_env_file=None, openrouter_api_key=SecretStr("test-key"))  # type: ignore[call-arg]


def test_settings_load_yaml(settings: Settings) -> None:
    assert SETTINGS_YAML.exists()
    assert settings.models.main and settings.models.judge and settings.models.fallback
    assert settings.limits.max_question_chars == 2000
    assert settings.llm.base_url == "https://openrouter.ai/api/v1"


def test_api_key_not_in_repr(settings: Settings) -> None:
    assert "test-key" not in repr(settings)


@pytest.mark.parametrize("role", ["main", "judge"])
def test_get_chat_model_uses_openrouter_with_fallback(settings: Settings, role: str) -> None:
    model = get_chat_model(role, settings)  # type: ignore[arg-type]
    assert isinstance(model, RunnableWithFallbacks)
    primary, fallback = model.runnable, model.fallbacks[0]
    assert isinstance(primary, ChatOpenAI) and isinstance(fallback, ChatOpenAI)
    assert primary.model_name == getattr(settings.models, role)
    assert fallback.model_name == settings.models.fallback
    assert str(primary.openai_api_base) == settings.llm.base_url
    assert primary.request_timeout == settings.llm.timeout_s
    assert primary.max_retries == settings.llm.max_retries


def _agent_llm(settings: Settings, model: str) -> ChatOpenAI:
    llm = get_tool_calling_llm("main", [], settings, model=model)._model
    assert isinstance(llm, RunnableWithFallbacks)
    bound = llm.runnable.bound  # type: ignore[attr-defined]
    assert isinstance(bound, ChatOpenAI)
    return bound


def test_only_the_anthropic_agent_loop_uses_prompt_caching(settings: Settings) -> None:
    assert _agent_llm(settings, "anthropic/some-model").extra_body == {
        "cache_control": {"type": "ephemeral"}
    }
    assert not _agent_llm(settings, "openai/some-model").extra_body
    # Single-shot calls never reuse a prefix, so caching would only add write cost.
    single = get_chat_model("main", settings, model="anthropic/some-model")
    assert isinstance(single, RunnableWithFallbacks)
    assert not single.runnable.extra_body  # type: ignore[attr-defined]


def test_prompt_cache_can_be_switched_off(settings: Settings) -> None:
    off = settings.model_copy(
        update={"llm": settings.llm.model_copy(update={"prompt_cache": False})}
    )
    assert not _agent_llm(off, "anthropic/some-model").extra_body


def test_every_model_gets_the_output_token_cap(settings: Settings) -> None:
    single = get_chat_model("main", settings, model="openai/some-model")
    assert isinstance(single, RunnableWithFallbacks)
    for llm in (single.runnable, _agent_llm(settings, "anthropic/some-model")):
        assert llm.max_tokens == settings.llm.max_tokens  # type: ignore[attr-defined]


def test_usage_includes_cache_tokens() -> None:
    msg = AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 5,
            "total_tokens": 105,
            "input_token_details": {"cache_read": 80, "cache_creation": 15},
        },
    )
    usage = usage_from_message(msg)
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (80, 15)
    total = usage + usage
    assert (total.cache_read_tokens, total.cache_write_tokens) == (160, 30)


def test_get_chat_model_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(LLMConfigError):
        get_chat_model("main", s)


def test_sinks_satisfy_protocol() -> None:
    assert isinstance(NullSink(), EventSink)
    assert isinstance(RecordingSink(), EventSink)


def test_recording_sink_records_in_order() -> None:
    sink = RecordingSink()
    sink.step_started("s1", "retrieve", {"q": "x"})
    sink.tool_result("search", {"q": "x"}, [1, 2])
    sink.token("Hel")
    sink.token("lo")
    sink.step_finished("s1", "done", {"n": 2})
    assert sink.kinds() == ["step_started", "tool_result", "token", "token", "step_finished"]
    assert sink.text() == "Hello"
    assert sink.events[-1].data["metadata"] == {"n": 2}


def test_chunk_validates_span() -> None:
    base = dict(id="c1", arxiv_id="0000.00000", title="t", text="abc")
    Chunk(**base, char_start=0, char_end=3)
    with pytest.raises(ValidationError):
        Chunk(**base, char_start=5, char_end=3)
