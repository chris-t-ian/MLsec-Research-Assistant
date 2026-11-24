"""Chainlit entry point: ``uv run chainlit run app/main.py -w``.

Thin layer: it calls the core (``run_turn``, ``confirm_pending_ingest``, exports) and renders
the results with :mod:`render`. No retrieval, agent, security or export logic lives here.

Startup fails fast: the module runs the core startup checks (configuration, secrets,
strategies, search index) when Chainlit loads it and exits with a readable message if any
fails. Every incoming message passes the per-session rate limit first.

Chat history is stored in SQLite (schema: ``research_assistant.persistence``); it needs a
login, so a password callback checks the demo user from .env. Each answer message stores its
``TurnResult`` as JSON in its metadata, from which the export actions rebuild the conversation.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.data import get_data_layer
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import Select, Slider
from chainlit.types import ThreadDict
from chainlit.utils import utc_now
from pydantic import ValidationError
from render import (
    RAG_STEP_NAME,
    ROOT_STEP_NAME,
    ChainlitSink,
    IngestTaskSink,
    answer_markdown,
    claim_element,
    ensure_files_dir,
    format_tool_result,
    help_element,
    md,
    rag_elements,
    rag_markdown,
    rag_summary,
    run_summary,
)

from research_assistant.agent import SessionState, TurnResult
from research_assistant.agent.graph import confirm_pending_ingest, default_agent, run_turn
from research_assistant.app_guide import default_app_guide, quick_reference
from research_assistant.auth import check_demo_credentials
from research_assistant.config import ConfigError, Settings, get_settings
from research_assistant.export import (
    Conversation,
    conversation_from_thread,
    export_citations_csv,
    export_json,
    export_messages_csv,
    export_pdf,
    turn_metadata,
)
from research_assistant.models import IngestRequest
from research_assistant.persistence import init_db, sqlalchemy_url
from research_assistant.rag_trace import build_rag_trace
from research_assistant.retrieval.strategies import (
    StrategyConfigError,
    get_strategy,
    strategy_names,
)
from research_assistant.security.rate_limit import default_rate_limiter
from research_assistant.startup import format_problems, run_checks

logger = logging.getLogger("app")

_startup_problems = run_checks("app")
if _startup_problems:
    _msg = format_problems(_startup_problems)
    logger.critical(_msg)
    raise SystemExit(_msg)

SESSION_KEY = "research_session"
RESEARCH_PROFILE = "Research Assistant"
GUIDE_PROFILE = "App Guide"
HELP_PATH = Path(__file__).resolve().parent / "help.md"
HELP_COMMAND = "help"
INGEST_TIMEOUT_S = 300
MAX_TOP_K = 20

GENERIC_ERROR = (
    "Sorry, something went wrong while handling your message. Please try again. "
    "If it keeps happening, mention error id `{error_id}`."
)
CONFIG_ERROR = (
    "The assistant is not configured correctly, so it cannot answer right now. "
    "Please check config/settings.yaml and config/strategies.yaml (error id `{error_id}`)."
)


def _error_id() -> str:
    return uuid.uuid4().hex[:8]


def _new_session(settings: Settings) -> SessionState:
    # The agent's memory is keyed by session_id: using the thread id lets a resumed thread
    # continue its history (as long as the server process is the same; memory is in RAM).
    return SessionState(
        session_id=cl.context.session.thread_id, strategy=settings.agent.default_strategy
    )


def _session() -> SessionState:
    session = cl.user_session.get(SESSION_KEY)
    if not isinstance(session, SessionState):
        session = _new_session(get_settings())
        cl.user_session.set(SESSION_KEY, session)
    return session


# --------------------------------------------------------------------------- persistence


@cl.data_layer
def chat_data_layer() -> SQLAlchemyDataLayer:
    path = init_db(get_settings().paths.resolve().chat_db)  # idempotent
    return SQLAlchemyDataLayer(conninfo=sqlalchemy_url(path))


@cl.password_auth_callback
async def password_auth(username: str, password: str) -> cl.User | None:
    try:
        settings = get_settings()
    except ConfigError:
        logger.exception("Login failed: configuration error")
        return None
    if check_demo_credentials(username, password, settings):
        return cl.User(identifier=username, metadata={"role": "demo", "provider": "credentials"})
    logger.info("Rejected login attempt")
    return None


# --------------------------------------------------------------------------- start


RESEARCH_STARTERS = [
    cl.Starter(
        label="Literature: LiRA vs. shadow models",
        message=(
            "How does LiRA improve on the shadow-model membership inference attack of "
            "Shokri et al., and why does it report TPR at low FPR?"
        ),
    ),
    cl.Starter(
        label="DP-SGD epsilon",
        message=(
            "What epsilon does DP-SGD reach with noise multiplier 1.1, sampling rate 0.01, "
            "10,000 steps and delta 1e-5? Compare RDP and PLD accounting."
        ),
    ),
    cl.Starter(
        label="Citation graph",
        message="Which papers cite Shokri et al.'s membership inference paper (1610.05820)?",
    ),
    cl.Starter(
        label="BibTeX export",
        message="Give me BibTeX for 1610.05820 and 2112.03570.",
    ),
]

GUIDE_STARTERS = [
    cl.Starter(label="What is HyDE?", message="What is HyDE and when should I use it?"),
    cl.Starter(
        label="Which strategy should I pick?",
        message="Which retrieval strategy should I use, and what do they cost?",
    ),
    cl.Starter(
        label="How are citations checked?", message="How do citations and verification work?"
    ),
    cl.Starter(label="Export", message="How do I export a conversation, and in which formats?"),
]


@cl.set_chat_profiles
async def chat_profiles(user: Any = None, language: Any = None) -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name=RESEARCH_PROFILE,
            markdown_description=(
                "Cited answers from ML privacy & security papers, with tools. "
                "Type **/help** for a quick reference."
            ),
            icon="/public/favicon.svg",
            default=True,
            starters=RESEARCH_STARTERS,
        ),
        cl.ChatProfile(
            name=GUIDE_PROFILE,
            markdown_description=(
                "Questions about using the app: strategies, citations, export, tools, security."
            ),
            icon="/public/guide.svg",
            starters=GUIDE_STARTERS,
        ),
    ]


def _is_guide() -> bool:
    return cl.user_session.get("chat_profile") == GUIDE_PROFILE


async def _set_research_commands() -> None:
    await cl.context.emitter.set_commands(
        [
            {
                "id": HELP_COMMAND,
                "description": "Quick reference: settings, strategies, tools, export",
                "icon": "circle-help",
                "button": False,
                "persistent": False,
            }
        ]
    )


def _chat_settings(settings: Settings, session: SessionState) -> cl.ChatSettings:
    names = strategy_names(settings)
    default = session.strategy or settings.agent.default_strategy
    top_k = session.top_k or get_strategy(default, settings).top_k
    models = settings.models.choices()
    return cl.ChatSettings(
        [
            Select(
                id="strategy",
                label="Retrieval strategy",
                values=names,
                initial_value=default if default in names else names[0],
                description="From config/strategies.yaml. Rerank strategies are slow on CPU.",
            ),
            Slider(
                id="top_k",
                label="Chunks per search (top_k)",
                initial=min(top_k, MAX_TOP_K),
                min=1,
                max=MAX_TOP_K,
                step=1,
            ),
            Select(
                id="model",
                label="Answer model",
                values=models,
                initial_value=session.model or models[0],
                description="From config/settings.yaml (models.main, fallback, selectable).",
            ),
        ]
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    try:
        settings = get_settings()
        if _is_guide():
            return  # the App Guide has no settings; its index is built on the first question
        session = _new_session(settings)
        cl.user_session.set(SESSION_KEY, session)
        await _chat_settings(settings, session).send()
        await _set_research_commands()
        await cl.make_async(default_agent)()  # build the agent now, not on the first question
    except (ConfigError, StrategyConfigError, ValidationError):
        error_id = _error_id()
        logger.exception("Chat start failed [error id %s]", error_id)
        await cl.Message(content=CONFIG_ERROR.format(error_id=error_id)).send()
    except Exception:
        error_id = _error_id()
        logger.exception("Chat start failed [error id %s]", error_id)
        await cl.Message(content=GENERIC_ERROR.format(error_id=error_id)).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    """A thread was reopened from the history sidebar: rebuild the session state."""
    try:
        settings = get_settings()
        if _is_guide():
            return
        session = _new_session(settings)
        saved = cl.user_session.get("chat_settings")  # restored by Chainlit from the thread
        if isinstance(saved, dict):
            _apply_settings(session, saved, settings)
        cl.user_session.set(SESSION_KEY, session)
        await _chat_settings(settings, session).send()
        await _set_research_commands()
        await cl.make_async(default_agent)()
    except Exception:
        error_id = _error_id()
        logger.exception("Chat resume failed [error id %s]", error_id)
        await cl.Message(content=GENERIC_ERROR.format(error_id=error_id)).send()


def _apply_settings(session: SessionState, values: dict[str, Any], settings: Settings) -> None:
    """Copy valid chat-setting values into the session (raises on invalid values)."""
    strategy = values.get("strategy")
    if isinstance(strategy, str) and strategy in strategy_names(settings):
        session.strategy = strategy
    if values.get("top_k") is not None:
        session.top_k = int(values["top_k"])
    model = values.get("model")
    if isinstance(model, str) and model in settings.models.choices():
        session.model = model


@cl.on_settings_update
async def on_settings_update(values: dict[str, Any]) -> None:
    session = _session()
    settings = get_settings()
    try:
        _apply_settings(session, values, settings)
    except (ValidationError, TypeError, ValueError):
        logger.warning("Rejected chat settings %r", values)
        await cl.Message(content="That setting is not valid; the previous value is kept.").send()


# --------------------------------------------------------------------------- messages


async def _render_turn(
    result: TurnResult, message: cl.Message, sink: ChainlitSink, session: SessionState
) -> None:
    message.content = answer_markdown(result)
    elements: list[Any] = []
    if not result.blocked:
        elements += [sink.reuse_source(c) for c in result.citations]
        elements += [claim_element(c, result.citations) for c in result.claim_checks]
        elements += sink.files
    message.elements = elements
    ensure_files_dir()
    strategy = session.strategy or get_settings().agent.default_strategy
    message.metadata = turn_metadata(result, strategy, session.top_k)
    message.actions = export_actions()
    if sink.streamed:  # the message already exists in the UI: replace the raw streamed text
        await message.update()
    else:
        await message.send()


async def _send_rag_view(result: TurnResult, answer: cl.Message) -> None:
    """Collapsible "How this answer was built" step under the answer (never breaks a turn)."""
    try:
        trace = await cl.make_async(build_rag_trace)(result)
        step = cl.Step(
            name=RAG_STEP_NAME,
            type="retrieval",
            parent_id=answer.id,
            show_input=False,
            default_open=False,
        )
        step.start = utc_now()
        step.output = f"_{rag_summary(trace)}_\n\n{rag_markdown(trace)}"
        step.end = utc_now()
        await step.send()
        # One at a time: Step.send() sends its elements concurrently, which scrambles the
        # chart order in the UI (candidates per search, then latency).
        for element in rag_elements(trace):
            await element.send(for_id=step.id)
    except Exception:
        logger.exception("Rendering the RAG view failed")


async def _handle_ingest(session: SessionState, req: IngestRequest) -> None:
    """Ask the user to confirm one pending ingest request, then run it with progress."""
    arxiv_id = req.arxiv_id  # from the core's request, never from the client's payload
    prompt = (
        f"{md(req.message, 500)}\n\n"
        f"Add **arXiv:{md(arxiv_id)}** to the knowledge base? This downloads the paper, scans "
        "it for hidden prompt injections and indexes it (can take a few minutes)."
    )
    response = await cl.AskActionMessage(
        content=prompt,
        actions=[
            cl.Action(name="ingest", payload={"decision": "ingest"}, label="Ingest", icon="check"),
            cl.Action(name="cancel", payload={"decision": "cancel"}, label="Cancel", icon="x"),
        ],
        timeout=INGEST_TIMEOUT_S,
    ).send()
    confirmed = bool(response) and (response or {}).get("payload", {}).get("decision") == "ingest"

    if not confirmed:
        await cl.make_async(confirm_pending_ingest)(session, arxiv_id, False)
        await cl.Message(content=f"Okay, arXiv:{md(arxiv_id)} was not added.").send()
        return

    tasks = await IngestTaskSink.create(arxiv_id)
    result = await cl.make_async(confirm_pending_ingest)(session, arxiv_id, True, tasks)
    await tasks.finish(result.ok)
    await cl.Message(content=format_tool_result("confirm_ingest", result)).send()


def _is_help_request(message: cl.Message) -> bool:
    text = (message.content or "").strip().lower()
    return message.command == HELP_COMMAND or text in ("/help", "help", "/?")


async def _guide_message(message: cl.Message) -> None:
    """App Guide profile: answer from app/help.md, citing its sections in the side panel."""
    root = cl.Step(name="App Guide", type="run", show_input=False)
    root.start = utc_now()
    await root.send()
    answer = cl.Message(content="", created_at=utc_now())
    sink = ChainlitSink(root, answer)
    try:
        guide = await cl.make_async(default_app_guide)(HELP_PATH)
        result = await cl.make_async(guide.answer)(message.content, sink)
    finally:
        root.end = utc_now()
        await root.update()
    answer.content = result.text
    answer.elements = [help_element(s) for s in result.cited]
    if sink.streamed:
        await answer.update()
    else:
        await answer.send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    error_id = _error_id()
    try:
        decision = default_rate_limiter().check(cl.context.session.id)
        if not decision.allowed:
            logger.info("Rate limit hit (session %s)", cl.context.session.id)
            await cl.Message(content=decision.message).send()
            return
        if _is_guide():
            await _guide_message(message)
            return
        if _is_help_request(message):
            await cl.Message(content=await cl.make_async(quick_reference)()).send()
            return
        session = _session()
        root = cl.Step(name=ROOT_STEP_NAME, type="run", show_input=False)
        root.start = utc_now()
        await root.send()
        # Streamed messages get no timestamp from Chainlit; without one the stored answer
        # sorts before its question in the history and exports.
        answer = cl.Message(content="", created_at=utc_now())
        sink = ChainlitSink(root, answer)
        try:
            result = await cl.make_async(run_turn)(message.content, session, sink)
        finally:
            root.name = ROOT_STEP_NAME  # it showed the running stage
            root.end = utc_now()
            await root.update()
        root.output = run_summary(result)
        await root.update()
        await _render_turn(result, answer, sink, session)
        if not result.blocked:
            await _send_rag_view(result, answer)
        for req in result.pending_ingests:
            await _handle_ingest(session, req)
    except Exception:
        logger.exception("Unhandled error in on_message [error id %s]", error_id)
        await cl.Message(content=GENERIC_ERROR.format(error_id=error_id)).send()


# --------------------------------------------------------------------------- export

EXPORT_FORMATS = ("json", "csv", "pdf")
EXPORT_ERROR = "Sorry, the export failed. Please try again (error id `{error_id}`)."


def export_actions() -> list[cl.Action]:
    return [
        cl.Action(
            name=f"export_{fmt}",
            payload={},
            label=f"Export {fmt.upper()}",
            icon="download",
            tooltip=f"Download the whole conversation as {fmt.upper()}",
        )
        for fmt in EXPORT_FORMATS
    ]


async def _load_conversation() -> Conversation | None:
    """The current thread as stored by the data layer (only the logged-in user's own)."""
    data_layer = get_data_layer()
    user = cl.context.session.user
    if data_layer is None or user is None:
        return None
    thread = await data_layer.get_thread(cl.context.session.thread_id)
    if not thread or thread.get("userIdentifier") != user.identifier:
        return None
    return conversation_from_thread(thread)


def _export_files(fmt: str, conv: Conversation) -> list[cl.File]:
    stem = f"conversation-{(conv.created_at or conv.exported_at):%Y%m%d-%H%M}"
    if fmt == "json":
        return [cl.File(name=f"{stem}.json", content=export_json(conv), mime="application/json")]
    if fmt == "csv":
        return [
            cl.File(
                name=f"{stem}-messages.csv", content=export_messages_csv(conv), mime="text/csv"
            ),
            cl.File(
                name=f"{stem}-citations.csv", content=export_citations_csv(conv), mime="text/csv"
            ),
        ]
    return [cl.File(name=f"{stem}.pdf", content=export_pdf(conv), mime="application/pdf")]


async def _export(action: cl.Action, fmt: str) -> None:
    error_id = _error_id()
    try:
        conv = await _load_conversation()
        if conv is None or not conv.messages:
            await cl.Message(
                content="Nothing to export yet: this conversation is not saved (logged in?)."
            ).send()
            return
        files = await cl.make_async(_export_files)(fmt, conv)
        target = action.forId
        if target is None:
            await cl.Message(content=f"{fmt.upper()} export:", elements=list(files)).send()
            return
        for f in files:  # attach the download(s) to the answer whose button was clicked
            await f.send(for_id=target, persist=False)
    except Exception:
        logger.exception("Export %s failed [error id %s]", fmt, error_id)
        await cl.Message(content=EXPORT_ERROR.format(error_id=error_id)).send()


@cl.action_callback("export_json")
async def on_export_json(action: cl.Action) -> None:
    await _export(action, "json")


@cl.action_callback("export_csv")
async def on_export_csv(action: cl.Action) -> None:
    await _export(action, "csv")


@cl.action_callback("export_pdf")
async def on_export_pdf(action: cl.Action) -> None:
    await _export(action, "pdf")
