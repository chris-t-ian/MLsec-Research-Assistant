"""Run every strategy on every selected gold question through the agent, judge the answers
and cache one JSONL line per (strategy, question) so an interrupted run can be resumed.

Progress is reported only through an :class:`EventSink` (step ids ``eval:<strategy>:<qid>``).
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from research_assistant.abtest.gold import GoldQuestion, Split
from research_assistant.abtest.judge import Judge, LLMJudge
from research_assistant.abtest.records import (
    EvalRecord,
    RetrievedChunk,
    SearchRecord,
)
from research_assistant.agent.graph import (
    UNEXPECTED_ERROR_MESSAGE,
    AgentConfigError,
    AgentDeps,
    ResearchAgent,
    run_turn,
)
from research_assistant.agent.state import SessionState, TurnResult
from research_assistant.config import PROJECT_ROOT, Settings
from research_assistant.events import EventSink, NullSink, RecordingSink
from research_assistant.llm import ChatStructuredLLM, LLMConfigError, get_chat_model

logger = logging.getLogger(__name__)

RESULTS_FILE = "results.jsonl"
MANIFEST_FILE = "manifest.json"
BASELINE = "baseline_dense"

TurnFn = Callable[[str, SessionState, EventSink], TurnResult]


class EvalRunError(RuntimeError):
    """The evaluation cannot run (bad selection, missing config, ...). User-safe message."""


class RunManifest(BaseModel):
    """What a run evaluates; stored as manifest.json and reused on resume."""

    run_id: str
    created_at: str
    split: Split
    final: bool = False
    strategies: list[str]
    question_ids: list[str]
    main_model: str
    judge_model: str
    temperature: float
    k: int
    seed: int
    n_resamples: int
    git_commit: str
    git_dirty: bool
    gold_file: str
    gold_sha256: str
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- provenance


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def git_info(root: Path = PROJECT_ROOT) -> tuple[str, bool]:
    """(commit hash, working tree dirty). ``"unknown"`` outside a git checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown", False
    return commit or "unknown", bool(status.strip())


def append_test_log(log_file: Path, manifest: RunManifest, run_dir: Path, action: str) -> None:
    """One JSON line per test-split run (``action`` = start | resume)."""
    entry = {
        "timestamp": utc_now(),
        "action": action,
        "run_id": manifest.run_id,
        "run_dir": str(run_dir),
        "git_commit": manifest.git_commit,
        "git_dirty": manifest.git_dirty,
        "strategies": manifest.strategies,
        "n_questions": len(manifest.question_ids),
    }
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- storage


class ResultStore:
    """Append-only JSONL; the last line per (strategy, question) wins."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[tuple[str, str], EvalRecord]:
        records: dict[tuple[str, str], EvalRecord] = {}
        if not self.path.exists():
            return records
        with self.path.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    rec = EvalRecord.model_validate_json(line)
                except ValidationError:
                    # A line cut off by an interruption: ignore it, the pair is rerun.
                    logger.warning("Skipping unreadable line %d in %s", n, self.path)
                    continue
                records[rec.key] = rec
        return records

    def append(self, record: EvalRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
            fh.flush()


def save_manifest(run_dir: Path, manifest: RunManifest) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / MANIFEST_FILE).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_manifest(run_dir: Path) -> RunManifest:
    path = run_dir / MANIFEST_FILE
    try:
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise EvalRunError(f"{path} is missing or invalid; cannot resume this run") from exc


# --------------------------------------------------------------------------- one turn


def to_record(
    result: TurnResult,
    question: GoldQuestion,
    strategy: str,
    run_id: str,
    sink: RecordingSink,
) -> EvalRecord:
    searches = []
    for trace in result.retrievals:
        final = next((st for st in reversed(trace.stages) if st.stage == "select"), None)
        chunks = (
            [
                RetrievedChunk(chunk_id=c.chunk_id, arxiv_id=c.arxiv_id, rank=c.rank)
                for c in final.candidates
            ]
            if final
            else []
        )
        searches.append(
            SearchRecord(
                query=trace.semantic_query, chunks=chunks, latency_ms=trace.total_latency_ms
            )
        )
    return EvalRecord(
        run_id=run_id,
        strategy=strategy,
        question_id=question.id,
        question_type=question.type,
        split=question.split,
        question=question.question,
        answer=result.answer,
        raw_answer=result.raw_answer,
        cited_arxiv_ids=[c.arxiv_id for c in sorted(result.citations, key=lambda c: c.number)],
        searches=searches,
        tool_calls=[t.name for t in result.tool_results],
        blocked=result.blocked,
        model=result.model,
        token_usage=result.token_usage,
        latency_ms=result.latency_ms,
        event_counts=dict(Counter(sink.kinds())),
        created_at=utc_now(),
    )


def turn_failed(result: TurnResult) -> bool:
    """The agent hit an internal error (not a real answer); such pairs are retried."""
    return result.answer == UNEXPECTED_ERROR_MESSAGE


# --------------------------------------------------------------------------- run


class RunOutcome(BaseModel):
    records: list[EvalRecord]
    n_new: int = 0
    n_rejudged: int = 0
    failed: list[str] = Field(default_factory=list)  # "strategy/qid"


def run_eval(
    manifest: RunManifest,
    questions: Sequence[GoldQuestion],
    store: ResultStore,
    turn_fn: TurnFn,
    judge: Judge,
    progress: EventSink | None = None,
) -> RunOutcome:
    """Run all missing (strategy, question) pairs, questions outer / strategies inner so that
    partial runs stay paired. Cached pairs are skipped; cached records whose judging failed
    are judged again."""
    progress = progress or NullSink()
    by_id = {q.id: q for q in questions}
    missing_ids = [qid for qid in manifest.question_ids if qid not in by_id]
    if missing_ids:
        raise EvalRunError(f"questions not in the gold file: {', '.join(missing_ids)}")
    done = store.load()
    outcome = RunOutcome(records=[])
    total = len(manifest.question_ids) * len(manifest.strategies)
    i = 0
    for qid in manifest.question_ids:
        q = by_id[qid]
        for strategy in manifest.strategies:
            i += 1
            step = f"eval:{strategy}:{qid}"
            cached = done.get((strategy, qid))
            if cached is not None and cached.judge is not None and cached.judge.ok:
                outcome.records.append(cached)
                continue
            if cached is not None:
                progress.step_started(step, f"[{i}/{total}] judge {strategy} q{qid}", q.question)
                rec = cached.model_copy(update={"judge": judge.judge(q, cached.answer)})
                store.append(rec)
                outcome.records.append(rec)
                outcome.n_rejudged += 1
                progress.step_finished(step, {"judge_ok": rec.judge.ok if rec.judge else False})
                continue
            progress.step_started(step, f"[{i}/{total}] {strategy} q{qid}", q.question)
            session = SessionState(
                session_id=f"eval-{manifest.run_id}-{strategy}-{qid}",
                strategy=strategy,
                model=manifest.main_model,
            )
            sink = RecordingSink()
            result = turn_fn(q.question, session, sink)
            if turn_failed(result):
                outcome.failed.append(f"{strategy}/{qid}")
                progress.step_finished(step, {"failed": True})
                continue
            rec = to_record(result, q, strategy, manifest.run_id, sink)
            rec = rec.model_copy(update={"judge": judge.judge(q, rec.answer)})
            store.append(rec)
            outcome.records.append(rec)
            outcome.n_new += 1
            progress.step_finished(
                step,
                {
                    "latency_s": round(rec.latency_ms / 1000, 1),
                    "tokens": rec.token_usage.total_tokens,
                    "judge_ok": rec.judge.ok if rec.judge else False,
                },
            )
    return outcome


# --------------------------------------------------------------------------- real deps


def eval_settings(settings: Settings) -> Settings:
    """Settings for the agent under test: temperature 0 and no silent model fallback for the
    main model (``fallback = main``), so every answer comes from ``models.main``."""
    models = settings.models.model_copy(update={"fallback": settings.models.main})
    llm = settings.llm.model_copy(update={"temperature": 0.0})
    return settings.model_copy(update={"models": models, "llm": llm})


def build_turn_fn(settings: Settings) -> TurnFn:
    """The real agent (loads the index once), checked up front so a missing API key fails
    before the first question instead of producing error answers."""
    agent = ResearchAgent(AgentDeps(eval_settings(settings)))
    agent.deps.llm_for(settings.models.main)
    agent.deps.answer_deps_for(settings.models.main)

    def turn(question: str, session: SessionState, sink: EventSink) -> TurnResult:
        return run_turn(question, session, sink, agent=agent)

    return turn


def build_judge(settings: Settings) -> LLMJudge:
    """The judge model at temperature 0, without fallback to another model."""
    judge_model = settings.models.judge
    models = settings.models.model_copy(update={"fallback": judge_model})
    llm_cfg = settings.llm.model_copy(update={"temperature": 0.0})
    s = settings.model_copy(update={"models": models, "llm": llm_cfg})
    try:
        chat = get_chat_model("judge", s)
    except LLMConfigError as exc:
        raise AgentConfigError(str(exc)) from exc
    return LLMJudge(ChatStructuredLLM(chat, max_attempts=s.llm.max_retries), judge_model)


__all__ = [
    "BASELINE",
    "EvalRunError",
    "ResultStore",
    "RunManifest",
    "RunOutcome",
    "TurnFn",
    "append_test_log",
    "build_judge",
    "build_turn_fn",
    "eval_settings",
    "git_info",
    "load_manifest",
    "run_eval",
    "run_timestamp",
    "save_manifest",
    "to_record",
]
