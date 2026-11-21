"""Chainlit entry point: ``uv run chainlit run app/main.py -w``.

Thin layer: it calls the core (``run_turn``, ``confirm_pending_ingest``) in a worker thread
and renders the results with :mod:`render`. No retrieval, agent or security logic lives here.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select, Slider
from chainlit.utils import utc_now
from pydantic import ValidationError
from render import (
    ChainlitSink,
    IngestTaskSink,
    answer_markdown,
    format_tool_result,
    md,
    run_summary,
    source_element,
)

from research_assistant.agent import SessionState, TurnResult
from research_assistant.agent.graph import confirm_pending_ingest, default_agent, run_turn
from research_assistant.config import ConfigError, Settings, get_settings
from research_assistant.models import IngestRequest
from research_assistant.retrieval.strategies import (
    StrategyConfigError,
    get_strategy,
    strategy_names,
)

logger = logging.getLogger("app")

SESSION_KEY = "research_session"
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


def _session() -> SessionState:
    session = cl.user_session.get(SESSION_KEY)
    if not isinstance(session, SessionState):
        session = SessionState(strategy=get_settings().agent.default_strategy)
        cl.user_session.set(SESSION_KEY, session)
    return session


# --------------------------------------------------------------------------- start


@cl.set_starters
async def set_starters(user: Any = None, language: Any = None) -> list[cl.Starter]:
    return [
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
        session = SessionState(strategy=settings.agent.default_strategy)
        cl.user_session.set(SESSION_KEY, session)
        await _chat_settings(settings, session).send()
        await cl.make_async(default_agent)()  # build the agent now, not on the first question
    except (ConfigError, StrategyConfigError, ValidationError):
        error_id = _error_id()
        logger.exception("Chat start failed [error id %s]", error_id)
        await cl.Message(content=CONFIG_ERROR.format(error_id=error_id)).send()
    except Exception:
        error_id = _error_id()
        logger.exception("Chat start failed [error id %s]", error_id)
        await cl.Message(content=GENERIC_ERROR.format(error_id=error_id)).send()


@cl.on_settings_update
async def on_settings_update(values: dict[str, Any]) -> None:
    session = _session()
    settings = get_settings()
    try:
        strategy = values.get("strategy")
        if isinstance(strategy, str) and strategy in strategy_names(settings):
            session.strategy = strategy
        if values.get("top_k") is not None:
            session.top_k = int(values["top_k"])
        model = values.get("model")
        if isinstance(model, str) and model in settings.models.choices():
            session.model = model
    except (ValidationError, TypeError, ValueError):
        logger.warning("Rejected chat settings %r", values)
        await cl.Message(content="That setting is not valid; the previous value is kept.").send()


# --------------------------------------------------------------------------- messages


async def _render_turn(result: TurnResult, message: cl.Message, sink: ChainlitSink) -> None:
    message.content = answer_markdown(result)
    elements: list[Any] = []
    if not result.blocked:
        elements += [source_element(c) for c in result.citations]
        elements += sink.files
    message.elements = elements
    if sink.streamed:  # the message already exists in the UI: replace the raw streamed text
        await message.update()
    else:
        await message.send()


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


@cl.on_message
async def on_message(message: cl.Message) -> None:
    error_id = _error_id()
    try:
        session = _session()
        root = cl.Step(name="Research", type="run", show_input=False)
        root.start = utc_now()
        await root.send()
        answer = cl.Message(content="")
        sink = ChainlitSink(root, answer)
        try:
            result = await cl.make_async(run_turn)(message.content, session, sink)
        finally:
            root.end = utc_now()
            await root.update()
        root.output = run_summary(result)
        await root.update()
        await _render_turn(result, answer, sink)
        for req in result.pending_ingests:
            await _handle_ingest(session, req)
    except Exception:
        logger.exception("Unhandled error in on_message [error id %s]", error_id)
        await cl.Message(content=GENERIC_ERROR.format(error_id=error_id)).send()
