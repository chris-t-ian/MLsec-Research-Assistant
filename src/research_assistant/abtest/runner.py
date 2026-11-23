"""Run every strategy x verification mode on every selected gold question through the agent,
judge the answers and cache one JSONL line per (strategy, mode, question) so an interrupted
run can be resumed.

Verification modes are claim-verification policies (off / flag / remove / regenerate). With
``flag``/``remove``/``regenerate`` the agent checks its own claims (and pays for it in tokens
and latency); with ``off`` the answer is checked after the turn, outside the measured cost,
so every arm gets a supported-claim rate.

Progress is reported only through an :class:`EventSink` (step ids
``eval:<arm>:<qid>``, arm = strategy or ``strategy+mode``).
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
    arm_name,
)
from research_assistant.abtest.retrieval_tuning import TuningManifest
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
from research_assistant.generation.answer import LLM_FAILED_MESSAGE, make_sources
from research_assistant.llm import ChatStructuredLLM, LLMConfigError, get_chat_model
from research_assistant.models import ClaimCheck, VerificationPolicy
from research_assistant.verification.verifier import VerificationError, verify_claims

logger = logging.getLogger(__name__)

RESULTS_FILE = "results.jsonl"
MANIFEST_FILE = "manifest.json"
BASELINE = "baseline_dense"

TurnFn = Callable[[str, SessionState, EventSink], TurnResult]


class ClaimMeasurement(BaseModel):
    checks: list[ClaimCheck] = Field(default_factory=list)
    supported_rate: float | None = None
    error: str | None = None


# Checks the claims of a stored answer (``raw_answer`` against ``sources``) after the turn.
MeasureFn = Callable[[EvalRecord], ClaimMeasurement]


class EvalRunError(RuntimeError):
    """The evaluation cannot run (bad selection, missing config, ...). User-safe message."""


class RunManifest(BaseModel):
    """What a run evaluates; stored as manifest.json and reused on resume."""

    run_id: str
    created_at: str
    split: Split
    final: bool = False
    strategies: list[str]
    # Claim-verification policies as an experiment dimension (older manifests: off only).
    verification_modes: list[VerificationPolicy] = Field(default_factory=lambda: ["off"])
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

    def arms(self) -> list[str]:
        """Experiment arms in run order: strategies outer, verification modes inner."""
        return [arm_name(s, m) for s in self.strategies for m in self.verification_modes]


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


def append_test_log(
    log_file: Path, manifest: RunManifest | TuningManifest, run_dir: Path, action: str
) -> None:
    """One JSON line per test-split run (``action`` = start | resume). ``kind`` tells a
    generation eval (run_eval.py) from a retrieval-only confirmation (tune_retrieval.py)."""
    entry = {
        "timestamp": utc_now(),
        "action": action,
        "kind": "retrieval" if isinstance(manifest, TuningManifest) else "generation",
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
    """Append-only JSONL; the last line per (strategy, mode, question) wins."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[tuple[str, str, str], EvalRecord]:
        records: dict[tuple[str, str, str], EvalRecord] = {}
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
    verification: VerificationPolicy = "off",
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
    info = result.verification
    return EvalRecord(
        run_id=run_id,
        strategy=strategy,
        verification=verification,
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
        sources={s.number: s.chunk_id for s in result.sources},
        claim_checks=result.claim_checks,
        supported_rate=result.supported_rate,
        claims_checked=info is not None and info.status != "skipped",
        claim_error=info.error if info is not None and info.status == "error" else None,
        created_at=utc_now(),
    )


# Answers that mean "no answer was produced" (internal error, LLM/API failure such as an
# OpenRouter 402/5xx). They say nothing about the strategy, so they are never stored as
# results; the pair is retried on resume.
FAILED_ANSWERS = frozenset({UNEXPECTED_ERROR_MESSAGE, LLM_FAILED_MESSAGE})
MAX_CONSECUTIVE_FAILURES = 3


def is_failed_answer(answer: str) -> bool:
    return answer.strip() in FAILED_ANSWERS


def turn_failed(result: TurnResult) -> bool:
    """No real answer was produced (internal or LLM error); such pairs are retried."""
    return is_failed_answer(result.answer)


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
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    measure: MeasureFn | None = None,
) -> RunOutcome:
    """Run all missing (strategy, mode, question) turns, questions outer / arms inner so that
    partial runs stay paired. Cached turns are skipped; cached records whose judging failed
    are judged again, cached failed answers (from older runs) are rerun.

    ``measure`` checks the claims of answers produced without verification (mode "off") for
    in-scope questions; cached "off" records without a claim check are checked on resume.

    Raises EvalRunError after ``max_consecutive_failures`` failed turns in a row (API down,
    out of credits); everything finished so far stays in the store for ``--resume``."""
    progress = progress or NullSink()
    by_id = {q.id: q for q in questions}
    missing_ids = [qid for qid in manifest.question_ids if qid not in by_id]
    if missing_ids:
        raise EvalRunError(f"questions not in the gold file: {', '.join(missing_ids)}")
    done = store.load()
    outcome = RunOutcome(records=[])
    arms = [(s, m) for s in manifest.strategies for m in manifest.verification_modes]
    total = len(manifest.question_ids) * len(arms)
    i = 0
    consecutive_failures = 0

    def needs_measure(rec: EvalRecord, q: GoldQuestion) -> bool:
        return (
            measure is not None
            and q.in_scope
            and rec.verification == "off"
            and not rec.claims_checked
            and bool(rec.sources)
        )

    def measured(rec: EvalRecord) -> EvalRecord:
        assert measure is not None
        m = measure(rec)
        return rec.model_copy(
            update={
                "claim_checks": m.checks,
                "supported_rate": m.supported_rate,
                "claims_checked": True,
                "claim_error": m.error,
            }
        )

    for qid in manifest.question_ids:
        q = by_id[qid]
        for strategy, mode in arms:
            i += 1
            arm = arm_name(strategy, mode)
            step = f"eval:{arm}:{qid}"
            cached = done.get((strategy, mode, qid))
            if cached is not None and is_failed_answer(cached.answer):
                cached = None
            if cached is not None:
                judged = cached.judge is not None and cached.judge.ok
                if judged and not needs_measure(cached, q):
                    outcome.records.append(cached)
                    continue
                progress.step_started(step, f"[{i}/{total}] judge {arm} q{qid}", q.question)
                rec = cached
                if not judged:
                    rec = rec.model_copy(update={"judge": judge.judge(q, rec.answer)})
                if needs_measure(rec, q):
                    rec = measured(rec)
                store.append(rec)
                outcome.records.append(rec)
                outcome.n_rejudged += 1
                progress.step_finished(step, {"judge_ok": rec.judge.ok if rec.judge else False})
                continue
            progress.step_started(step, f"[{i}/{total}] {arm} q{qid}", q.question)
            session = SessionState(
                session_id=f"eval-{manifest.run_id}-{strategy}-{mode}-{qid}",
                strategy=strategy,
                model=manifest.main_model,
                verification=mode,
            )
            sink = RecordingSink()
            result = turn_fn(q.question, session, sink)
            if turn_failed(result):
                outcome.failed.append(f"{arm}/{qid}")
                progress.step_finished(step, {"failed": True})
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise EvalRunError(
                        f"{consecutive_failures} turns in a row produced no answer (LLM/API "
                        "error, e.g. out of OpenRouter credits). Stopped; finished turns are "
                        "saved, continue with --resume once the API works again."
                    )
                continue
            consecutive_failures = 0
            rec = to_record(result, q, strategy, manifest.run_id, sink, mode)
            rec = rec.model_copy(update={"judge": judge.judge(q, rec.answer)})
            if needs_measure(rec, q):
                rec = measured(rec)
            store.append(rec)
            outcome.records.append(rec)
            outcome.n_new += 1
            progress.step_finished(
                step,
                {
                    "latency_s": round(rec.latency_ms / 1000, 1),
                    "tokens": rec.token_usage.total_tokens,
                    "judge_ok": rec.judge.ok if rec.judge else False,
                    "supported_rate": rec.supported_rate,
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


def build_agent(settings: Settings) -> ResearchAgent:
    """The real agent under test (loads the index once), checked up front so a missing API
    key fails before the first question instead of producing error answers."""
    agent = ResearchAgent(AgentDeps(eval_settings(settings), judge=build_claim_judge(settings)))
    agent.deps.llm_for(settings.models.main)
    agent.deps.answer_deps_for(settings.models.main)
    return agent


def build_turn_fn(settings: Settings, agent: ResearchAgent | None = None) -> TurnFn:
    agent = agent or build_agent(settings)

    def turn(question: str, session: SessionState, sink: EventSink) -> TurnResult:
        return run_turn(question, session, sink, agent=agent)

    return turn


def build_measure_fn(agent: ResearchAgent) -> MeasureFn:
    """Claim check of a stored answer after the turn: the cited chunks are read from the
    index by id, the judge is the agent's claim judge (same prompt as the verify node)."""
    settings = agent.settings

    def measure(rec: EvalRecord) -> ClaimMeasurement:
        index = agent.deps.retrieval.index
        chunks = []
        for number in sorted(rec.sources):
            got = index.get(rec.sources[number])
            if got is None:
                return ClaimMeasurement(error=f"chunk {rec.sources[number]} not in the index")
            chunks.append(got[0])
        try:
            v = verify_claims(
                rec.raw_answer, make_sources(chunks), agent.deps.judge_llm(), settings.verification
            )
        except VerificationError as exc:
            return ClaimMeasurement(error=str(exc))
        return ClaimMeasurement(checks=v.checks, supported_rate=v.supported_rate)

    return measure


def _judge_settings(settings: Settings) -> Settings:
    judge_model = settings.models.judge
    models = settings.models.model_copy(update={"fallback": judge_model})
    llm_cfg = settings.llm.model_copy(update={"temperature": 0.0})
    return settings.model_copy(update={"models": models, "llm": llm_cfg})


def build_claim_judge(settings: Settings) -> ChatStructuredLLM:
    """Claim-verification judge: ``models.judge`` at temperature 0, without fallback."""
    s = _judge_settings(settings)
    try:
        chat = get_chat_model("judge", s)
    except LLMConfigError as exc:
        raise AgentConfigError(str(exc)) from exc
    return ChatStructuredLLM(chat, max_attempts=s.llm.max_retries)


def build_judge(settings: Settings) -> LLMJudge:
    """The judge model at temperature 0, without fallback to another model."""
    s = _judge_settings(settings)
    try:
        chat = get_chat_model("judge", s)
    except LLMConfigError as exc:
        raise AgentConfigError(str(exc)) from exc
    return LLMJudge(ChatStructuredLLM(chat, max_attempts=s.llm.max_retries), s.models.judge)


__all__ = [
    "BASELINE",
    "ClaimMeasurement",
    "EvalRunError",
    "MeasureFn",
    "ResultStore",
    "RunManifest",
    "RunOutcome",
    "TurnFn",
    "append_test_log",
    "build_agent",
    "build_claim_judge",
    "build_judge",
    "build_measure_fn",
    "build_turn_fn",
    "eval_settings",
    "git_info",
    "load_manifest",
    "run_eval",
    "run_timestamp",
    "save_manifest",
    "to_record",
]
