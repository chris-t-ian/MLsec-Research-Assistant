"""Hardening: LLM fallback on provider errors, friendly tool errors when external APIs are
down, startup checks (config, secrets, index) and the per-session rate limit.

All HTTP is mocked: OpenRouter via an httpx.MockTransport injected into ChatOpenAI, arXiv and
Semantic Scholar via their clients' ``http`` parameter.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, ValidationError
from tenacity import wait_none

from research_assistant import config as config_mod
from research_assistant import llm as llm_mod
from research_assistant import startup
from research_assistant.agent.graph import AgentDeps, ResearchAgent, run_turn
from research_assistant.agent.state import SessionState
from research_assistant.config import (
    ArxivConfig,
    ConfigError,
    GenerationConfig,
    PathsConfig,
    SemanticScholarConfig,
    Settings,
    describe_validation_error,
    get_settings,
)
from research_assistant.generation.answer import AnswerDeps
from research_assistant.ingest.arxiv_fetch import ArxivClient, RateLimiter
from research_assistant.ingest.index import BM25_FILE, CHROMA_DIR, BM25Store
from research_assistant.llm import (
    LLMCallError,
    StreamResult,
    get_streaming_llm,
    get_tool_calling_llm,
)
from research_assistant.models import Chunk
from research_assistant.retrieval.strategies import RetrievalDeps
from research_assistant.security.rate_limit import SessionRateLimiter
from research_assistant.security.tool_policy import AuditLog, ToolPolicy
from research_assistant.tools import StaticKnowledgeBase, ToolDeps, arxiv_search
from research_assistant.tools.citations import citation_graph
from research_assistant.tools.semantic_scholar import SemanticScholarClient

# --------------------------------------------------------------------------- OpenRouter

MAIN = "anthropic/claude-test-main"
FALLBACK = "openai/test-fallback"


def _completion(model: str, text: str) -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _sse(model: str, text: str) -> bytes:
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    lines = [
        "data: "
        + json.dumps(
            {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": model} | c
        )
        for c in chunks
    ]
    return ("\n\n".join([*lines, "data: [DONE]"]) + "\n\n").encode()


class FakeOpenRouter:
    """Answers chat completions: ``status`` for the main model, success for the fallback."""

    def __init__(self, status: int) -> None:
        self.status = status
        self.models: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        self.models.append(model)
        if model != FALLBACK:
            return httpx.Response(
                self.status,
                json={"error": {"message": "provider error", "code": self.status}},
                headers={"retry-after": "0"},
            )
        if body.get("stream"):
            return httpx.Response(
                200,
                content=_sse(model, "fallback streamed"),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=_completion(model, "fallback answer"))


@pytest.fixture
def openrouter(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], tuple[Settings, FakeOpenRouter]]:
    def setup(status: int) -> tuple[Settings, FakeOpenRouter]:
        fake = FakeOpenRouter(status)
        client = httpx.Client(transport=httpx.MockTransport(fake))
        real = ChatOpenAI

        def chat(**kwargs: Any) -> ChatOpenAI:
            return real(**kwargs, http_client=client)

        monkeypatch.setattr(llm_mod, "ChatOpenAI", chat)
        base = get_settings()
        settings = base.model_copy(
            update={
                "openrouter_api_key": SecretStr("test-key"),
                "models": base.models.model_copy(update={"main": MAIN, "fallback": FALLBACK}),
                # No client-side retries: the first provider error goes to the fallback.
                "llm": base.llm.model_copy(update={"max_retries": 0}),
            }
        )
        return settings, fake

    return setup


@pytest.mark.parametrize("status", [429, 500])
def test_tool_calling_llm_falls_back_on_provider_error(
    openrouter: Callable[[int], tuple[Settings, FakeOpenRouter]], status: int
) -> None:
    settings, fake = openrouter(status)
    llm = get_tool_calling_llm("main", [], settings)

    msg = llm.invoke_messages([HumanMessage(content="hello")])

    assert isinstance(msg, AIMessage) and msg.content == "fallback answer"
    assert fake.models == [MAIN, FALLBACK]


@pytest.mark.parametrize("status", [429, 500])
def test_streaming_llm_falls_back_on_provider_error(
    openrouter: Callable[[int], tuple[Settings, FakeOpenRouter]], status: int
) -> None:
    settings, fake = openrouter(status)
    tokens: list[str] = []

    res = get_streaming_llm("main", settings).stream_text("system", "user", tokens.append)

    assert res.text == "fallback streamed" and "".join(tokens) == "fallback streamed"
    assert fake.models == [MAIN, FALLBACK]


def test_llm_error_without_fallback_is_typed_and_secret_free(
    openrouter: Callable[[int], tuple[Settings, FakeOpenRouter]],
) -> None:
    settings, fake = openrouter(500)
    only_main = settings.model_copy(
        update={"models": settings.models.model_copy(update={"fallback": MAIN})}
    )
    llm = get_tool_calling_llm("main", [], only_main)

    with pytest.raises(LLMCallError) as exc:
        llm.invoke_messages([HumanMessage(content="hello")])

    assert "test-key" not in str(exc.value)
    assert fake.models == [MAIN]


# --------------------------------------------------------------------------- arXiv / S2 down


def _arxiv_deps(handler: Callable[[httpx.Request], httpx.Response]) -> tuple[ToolDeps, list[Any]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = ArxivClient(
        ArxivConfig(max_retries=2),
        http=httpx.Client(transport=httpx.MockTransport(record)),
        limiter=RateLimiter(0),
        retry_wait=wait_none(),
    )
    return ToolDeps(settings=get_settings(), kb=StaticKnowledgeBase(set()), arxiv=client), seen


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


@pytest.mark.parametrize("handler", [_connect_error, _timeout, lambda r: httpx.Response(503)])
def test_arxiv_down_gives_friendly_tool_error_after_retries(
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    deps, seen = _arxiv_deps(handler)

    result = arxiv_search("membership inference", deps=deps)

    assert not result.ok and result.source == "arxiv"
    assert result.error == "arXiv is not reachable right now. Please try again in a minute."
    assert len(seen) == 3  # 1 try + max_retries=2


def _s2_deps(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[ToolDeps, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = SemanticScholarClient(
        SemanticScholarConfig(max_retries=2),
        http=httpx.Client(transport=httpx.MockTransport(record)),
        limiter=RateLimiter(0),
        retry_wait=wait_none(),
    )
    deps = ToolDeps(settings=get_settings(), kb=StaticKnowledgeBase(set()), semantic_scholar=client)
    return deps, seen


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (_connect_error, "Semantic Scholar is not reachable right now. Please try again later."),
        (_timeout, "Semantic Scholar is not reachable right now. Please try again later."),
        (
            lambda r: httpx.Response(503),
            "Semantic Scholar is not reachable right now. Please try again later.",
        ),
        (
            lambda r: httpx.Response(429),
            "Semantic Scholar is rate-limiting requests. Please try again in a minute.",
        ),
    ],
)
def test_semantic_scholar_down_gives_friendly_tool_error(
    handler: Callable[[httpx.Request], httpx.Response], message: str
) -> None:
    deps, seen = _s2_deps(handler)

    result = citation_graph("1610.05820", "citations", deps=deps)

    assert not result.ok and result.source == "semantic_scholar"
    assert result.error == message
    assert len(seen) == 3


class _ScriptedLLM:
    def __init__(self, messages: list[AIMessage]) -> None:
        self.messages = messages

    def invoke_messages(self, messages: Any) -> AIMessage:
        return self.messages.pop(0) if self.messages else AIMessage(content="done")


class _EmptyIndex:
    def dense_search(self, *args: Any, **kw: Any) -> list[Any]:
        return []

    def bm25_search(self, *args: Any, **kw: Any) -> list[Any]:
        return []

    def metadata(self) -> list[Any]:
        return []


class _FixedStreamingLLM:
    def __init__(self, text: str) -> None:
        self.text = text

    def stream_text(self, system: str, user: str, on_token: Callable[[str], None]) -> Any:
        on_token(self.text)
        return StreamResult(text=self.text, model="fake/model")


def test_agent_turn_survives_semantic_scholar_outage() -> None:
    """The friendly tool error reaches the turn result; the turn still produces an answer."""
    deps, _ = _s2_deps(_connect_error)
    call = {
        "name": "citation_graph",
        "args": {"arxiv_id": "1610.05820"},
        "id": "c1",
        "type": "tool_call",
    }
    retrieval = RetrievalDeps(_EmptyIndex())  # type: ignore[arg-type]
    gen = _FixedStreamingLLM("The citation lookup failed [T1].")
    agent = ResearchAgent(
        AgentDeps(
            get_settings(),
            llm=_ScriptedLLM([AIMessage(content="", tool_calls=[call])]),
            answer_deps=AnswerDeps(retrieval, gen, GenerationConfig(), "fake/model"),
            retrieval=retrieval,
            tool_deps=deps,
            policy=ToolPolicy(6, AuditLog(None)),
        )
    )
    res = run_turn("Who cites 1610.05820?", SessionState(verification="off"), agent=agent)

    [outcome] = [o for o in res.tool_results if o.name == "citation_graph"]
    assert outcome.allowed and not outcome.result.ok
    assert outcome.result.error and "not reachable" in outcome.result.error
    assert "Traceback" not in res.answer and res.answer


# --------------------------------------------------------------------------- startup checks


def _settings_with_index(index_dir: Path) -> Settings:
    base = get_settings()
    return base.model_copy(
        update={"paths": PathsConfig(**{**base.paths.model_dump(), "index_dir": index_dir})}
    )


def test_missing_index_points_to_ingest_script(tmp_path: Path) -> None:
    problem = startup.check_index(_settings_with_index(tmp_path / "index"))

    assert problem is not None and problem.check == "index"
    assert "No search index found" in problem.message
    assert "scripts/ingest.py" in problem.message and "uv run just ingest" in problem.message


def test_empty_index_is_reported(tmp_path: Path) -> None:
    (tmp_path / CHROMA_DIR).mkdir()
    BM25Store(tmp_path / BM25_FILE).save()

    problem = startup.check_index(_settings_with_index(tmp_path))

    assert problem is not None and "empty" in problem.message


def test_present_index_passes(tmp_path: Path) -> None:
    (tmp_path / CHROMA_DIR).mkdir()
    store = BM25Store(tmp_path / BM25_FILE)
    chunk = Chunk(
        id="1610.05820:0:0",
        arxiv_id="1610.05820",
        title="T",
        authors=["A"],
        year=2017,
        section="Intro",
        text="membership inference",
        char_start=0,
        char_end=20,
    )
    store.upsert([chunk], "membership_inference")
    store.save()

    assert startup.check_index(_settings_with_index(tmp_path)) is None


def test_ensure_ready_raises_readable_error(tmp_path: Path) -> None:
    settings = _settings_with_index(tmp_path / "missing").model_copy(
        update={"openrouter_api_key": None}
    )

    with pytest.raises(startup.StartupError) as exc:
        startup.ensure_ready("cli", settings)

    text = str(exc.value)
    assert text.startswith("The research assistant cannot start:")
    assert "[secrets] OPENROUTER_API_KEY not set" in text
    assert "[index]" in text and "scripts/ingest.py" in text


def test_app_secrets_are_checked_only_for_the_app() -> None:
    settings = get_settings().model_copy(
        update={
            "openrouter_api_key": SecretStr("k"),
            "chainlit_auth_secret": None,
            "demo_username": "demo",
            "demo_password": SecretStr(" "),
        }
    )
    [problem] = startup.check_secrets(settings, "app")
    assert "CHAINLIT_AUTH_SECRET" in problem.message and "DEMO_PASSWORD" in problem.message
    assert startup.check_secrets(settings, "cli") == []


def test_unknown_default_strategy_is_reported() -> None:
    base = get_settings()
    settings = base.model_copy(
        update={"agent": base.agent.model_copy(update={"default_strategy": "nope"})}
    )
    problem = startup.check_strategies(settings)
    assert problem is not None and "agent.default_strategy" in problem.message


def test_invalid_config_message_lists_fields_without_values() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(  # type: ignore[call-arg]
            limits={"max_question_chars": -1},
            llm={"timeout_s": "sk-or-secret-value"},
        )
    text = describe_validation_error(exc.value)

    assert "limits.max_question_chars" in text and "llm.timeout_s" in text
    assert "sk-or-secret-value" not in text


def test_get_settings_fails_fast_with_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> Settings:
        return Settings(agent={"max_tool_calls_per_turn": 99})  # type: ignore[call-arg]

    monkeypatch.setattr(config_mod, "Settings", broken)
    with pytest.raises(ConfigError) as exc:
        get_settings.__wrapped__()  # bypass the per-process cache

    assert "agent.max_tool_calls_per_turn" in str(exc.value)


def test_run_checks_reports_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> Settings:
        raise ConfigError("Invalid configuration in config/settings.yaml or .env:\n  x: bad")

    monkeypatch.setattr(startup, "get_settings", broken)
    [problem] = startup.run_checks("app")
    assert problem.check == "config" and "x: bad" in problem.message


def test_app_exits_at_startup_with_readable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = startup.StartupProblem(
        check="index", message=f"No search index found in data/index. {startup.INGEST_HINT}"
    )
    monkeypatch.setattr(startup, "run_checks", lambda target="app", settings=None: [problem])
    monkeypatch.delitem(sys.modules, "main", raising=False)

    with pytest.raises(SystemExit) as exc:
        importlib.import_module("main")

    assert "cannot start" in str(exc.value) and "scripts/ingest.py" in str(exc.value)
    sys.modules.pop("main", None)


# --------------------------------------------------------------------------- rate limit


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_rate_limit_allows_20_per_minute_per_session() -> None:
    clock = Clock()
    limiter = SessionRateLimiter(20, clock=clock)

    assert all(limiter.check("a").allowed for _ in range(20))
    denied = limiter.check("a")
    assert not denied.allowed and denied.retry_after_s == 60
    assert "wait 60 seconds" in denied.message
    assert limiter.check("b").allowed  # other sessions are independent

    clock.now += 30
    assert not limiter.check("a").allowed  # rejected messages do not extend the window
    clock.now += 30
    assert limiter.check("a").allowed  # the oldest messages left the window


def test_rate_limit_is_configured_from_settings() -> None:
    assert get_settings().limits.messages_per_minute == 20
    with pytest.raises(ValidationError):
        Settings(limits={"messages_per_minute": 0})  # type: ignore[call-arg]
