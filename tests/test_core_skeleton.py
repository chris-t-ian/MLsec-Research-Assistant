"""Unit tests for config, llm factory, events and models (no network)."""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableWithFallbacks
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, ValidationError

from research_assistant.config import SETTINGS_YAML, Settings
from research_assistant.events import EventSink, NullSink, RecordingSink
from research_assistant.llm import LLMConfigError, get_chat_model
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
