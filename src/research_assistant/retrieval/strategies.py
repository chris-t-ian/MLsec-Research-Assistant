"""Configurable retrieval strategies and the :func:`retrieve` pipeline.

Stages (each one ``step_started``/``step_finished`` pair on the sink, step id
``retrieval:<strategy>:<stage>``; stages a strategy does not use are skipped):

    self_query         LLM filters + cleaned semantic query        (use_filters)
    query_translation  multi_query / hyde / decompose              (query_translation)
    retrieve           dense / bm25 / hybrid lists, fetch_k each
    fuse               Reciprocal Rank Fusion per query group      (more than one list)
    rerank             cross-encoder on the top rerank_k per group (rerank)
    select             interleave groups, deduplicate, cap per paper, top_k

The returned :class:`RetrievalResult` carries a :class:`RetrievalTrace` with rewrites,
filters, candidates and scores per stage, latency per stage and LLM token usage.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from research_assistant.config import Settings, get_settings
from research_assistant.events import EventSink, NullSink
from research_assistant.llm import LLMConfigError, StructuredLLM, get_structured_llm
from research_assistant.models import (
    CandidateScore,
    QueryFilters,
    QueryTranslation,
    RetrievalResult,
    RetrievalTrace,
    RetrieverKind,
    ScoredChunk,
    StageTrace,
    TokenUsage,
)
from research_assistant.retrieval.query_translation import translate_query
from research_assistant.retrieval.rerank import CrossEncoderReranker, Reranker, RerankError, rerank
from research_assistant.retrieval.retrievers import (
    DEFAULT_RRF_K,
    RankedLists,
    SearchIndex,
    fuse,
    interleave,
    retrieve_lists,
)
from research_assistant.retrieval.self_query import (
    IndexCatalog,
    filters_to_predicate,
    filters_to_where,
    self_query,
)

logger = logging.getLogger(__name__)


class StrategyConfigError(ValueError):
    """config/strategies.yaml is missing or invalid."""


class RetrievalError(ValueError):
    """The question cannot be retrieved for (empty or too long). Safe to show to users."""


# --------------------------------------------------------------------------- config


class StrategyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    query_translation: QueryTranslation = "none"
    use_filters: bool = False
    retriever: RetrieverKind = "hybrid"
    rerank: bool = False
    fetch_k: int = Field(default=30, gt=0, le=500, description="Candidates per ranked list.")
    top_k: int = Field(default=8, gt=0, le=100, description="Chunks returned.")
    rerank_k: int = Field(
        default=15, gt=0, le=500, description="Fused candidates per group scored by the reranker."
    )
    max_chunks_per_paper: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def _check_k(self) -> StrategyConfig:
        if self.top_k > self.fetch_k:
            raise ValueError("top_k must not exceed fetch_k")
        if self.rerank and not self.top_k <= self.rerank_k <= self.fetch_k:
            raise ValueError("rerank_k must be between top_k and fetch_k")
        return self


def load_strategies(path: Path) -> dict[str, StrategyConfig]:
    """Load strategies from YAML: ``{defaults: {...}, strategies: [{name: ..., ...}]}``."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StrategyConfigError(f"Cannot read {path.name}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("strategies"), list):
        raise StrategyConfigError(f"{path.name} must contain a 'strategies' list")
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise StrategyConfigError(f"'defaults' in {path.name} must be a mapping")
    strategies: dict[str, StrategyConfig] = {}
    for i, entry in enumerate(data["strategies"]):
        if not isinstance(entry, dict):
            raise StrategyConfigError(f"strategy #{i + 1} in {path.name} must be a mapping")
        try:
            cfg = StrategyConfig.model_validate({**defaults, **entry})
        except ValidationError as exc:
            raise StrategyConfigError(f"invalid strategy #{i + 1} in {path.name}: {exc}") from exc
        if cfg.name in strategies:
            raise StrategyConfigError(f"duplicate strategy name {cfg.name!r} in {path.name}")
        strategies[cfg.name] = cfg
    if not strategies:
        raise StrategyConfigError(f"{path.name} defines no strategies")
    return strategies


def get_strategy(name: str, settings: Settings | None = None) -> StrategyConfig:
    settings = settings or get_settings()
    strategies = load_strategies(settings.paths.resolve().strategies_file)
    if name not in strategies:
        raise StrategyConfigError(
            f"unknown strategy {name!r}; available: {', '.join(sorted(strategies))}"
        )
    return strategies[name]


# --------------------------------------------------------------------------- dependencies


class RetrievalDeps:
    """Index, LLM, reranker and settings shared across retrieval runs."""

    def __init__(
        self,
        index: SearchIndex,
        llm: StructuredLLM | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        max_question_chars: int = 2000,
    ) -> None:
        self.index = index
        self.llm = llm
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.max_question_chars = max_question_chars
        self._catalog: IndexCatalog | None = None

    @property
    def catalog(self) -> IndexCatalog:
        if self._catalog is None:
            self._catalog = IndexCatalog.from_metadata(self.index.metadata())
        return self._catalog

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RetrievalDeps:
        from research_assistant.ingest.index import ChunkIndex  # heavy import: chromadb

        settings = settings or get_settings()
        index = ChunkIndex.from_config(settings.paths.resolve().index_dir, settings.retrieval)
        llm: StructuredLLM | None
        try:
            llm = get_structured_llm("judge", settings)
        except LLMConfigError as exc:
            logger.warning("No LLM for query translation / self-query: %s", exc)
            llm = None
        return cls(
            index=index,
            llm=llm,
            reranker=CrossEncoderReranker.from_config(settings.retrieval),
            rrf_k=settings.retrieval.rrf_k,
            max_question_chars=settings.limits.max_question_chars,
        )


@lru_cache(maxsize=1)
def default_deps() -> RetrievalDeps:
    return RetrievalDeps.from_settings()


# --------------------------------------------------------------------------- tracing


class _StageRecord:
    def __init__(self) -> None:
        self.candidates: list[CandidateScore] = []
        self.notes: list[str] = []
        self.usage = TokenUsage()
        self.output: dict[str, Any] = {}


class _Tracer:
    def __init__(self, sink: EventSink, strategy: str) -> None:
        self.sink = sink
        self.strategy = strategy
        self.stages: list[StageTrace] = []

    @contextmanager
    def stage(self, name: str, input: dict[str, Any]) -> Iterator[_StageRecord]:
        step_id = f"retrieval:{self.strategy}:{name}"
        self.sink.step_started(step_id, name, input)
        rec = _StageRecord()
        start = time.perf_counter()
        try:
            yield rec
        except Exception as exc:
            rec.notes.append(f"failed: {type(exc).__name__}")
            raise
        finally:
            latency = (time.perf_counter() - start) * 1000
            self.stages.append(
                StageTrace(
                    stage=name,
                    latency_ms=latency,
                    candidates=rec.candidates,
                    token_usage=rec.usage,
                    notes=rec.notes,
                )
            )
            self.sink.step_finished(
                step_id,
                rec.output,
                {
                    "latency_ms": round(latency, 1),
                    "token_usage": rec.usage.model_dump(),
                    "n_candidates": len(rec.candidates),
                    "notes": list(rec.notes),
                },
            )

    def usage(self) -> TokenUsage:
        total = TokenUsage()
        for s in self.stages:
            total = total + s.token_usage
        return total


def _candidates(lists: dict[str, Sequence[ScoredChunk]]) -> list[CandidateScore]:
    return [
        CandidateScore(
            chunk_id=h.chunk.id, arxiv_id=h.chunk.arxiv_id, rank=pos, score=h.score, list_name=name
        )
        for name, hits in lists.items()
        for pos, h in enumerate(hits, start=1)
    ]


def _summary(hits: Sequence[ScoredChunk], n: int = 10) -> list[dict[str, Any]]:
    return [{"chunk_id": h.chunk.id, "score": round(h.score, 4)} for h in hits[:n]]


# --------------------------------------------------------------------------- selection

_WS = re.compile(r"\s+")


def select_chunks(
    hits: Sequence[ScoredChunk], top_k: int, max_chunks_per_paper: int
) -> list[ScoredChunk]:
    """Deduplicate (by id and by identical text), cap chunks per paper, keep ``top_k``."""
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    per_paper: dict[str, int] = {}
    out: list[ScoredChunk] = []
    for h in hits:
        text_key = _WS.sub(" ", h.chunk.text).strip().casefold()
        if h.chunk.id in seen_ids or text_key in seen_text:
            continue
        if per_paper.get(h.chunk.arxiv_id, 0) >= max_chunks_per_paper:
            continue
        seen_ids.add(h.chunk.id)
        seen_text.add(text_key)
        per_paper[h.chunk.arxiv_id] = per_paper.get(h.chunk.arxiv_id, 0) + 1
        out.append(h.model_copy(update={"rank": len(out) + 1}))
        if len(out) >= top_k:
            break
    return out


# --------------------------------------------------------------------------- pipeline


def _validate_question(question: str, max_chars: int) -> str:
    q = " ".join(question.split())
    if not q:
        raise RetrievalError("Please enter a question.")
    if len(q) > max_chars:
        raise RetrievalError(f"The question is too long (max {max_chars} characters).")
    return q


def retrieve(
    question: str,
    strategy: StrategyConfig | str,
    sink: EventSink | None = None,
    deps: RetrievalDeps | None = None,
) -> RetrievalResult:
    """Retrieve chunks for ``question`` with ``strategy`` (a config or a name in strategies.yaml).

    LLM or reranker failures degrade gracefully (recorded in the trace); only an invalid
    question (:class:`RetrievalError`) or a broken index raises.
    """
    if isinstance(strategy, str):
        strategy = get_strategy(strategy)
    sink = sink or NullSink()
    deps = deps or default_deps()
    q = _validate_question(question, deps.max_question_chars)
    tracer = _Tracer(sink, strategy.name)
    t0 = time.perf_counter()

    # 1. self-query
    semantic = q
    filters: QueryFilters | None = None
    if strategy.use_filters:
        with tracer.stage("self_query", {"question": q}) as st:
            sq = self_query(q, deps.llm, deps.catalog)
            semantic, filters = sq.semantic_query, sq.filters
            st.usage, st.notes = sq.token_usage, list(sq.notes)
            st.output = {
                "semantic_query": semantic,
                "filters": filters.model_dump() if filters else None,
            }
    where = filters_to_where(filters, deps.catalog)
    predicate = filters_to_predicate(filters, deps.catalog)

    # 2. query translation
    if strategy.query_translation == "none":
        translation = translate_query(semantic, "none", None)
    else:
        with tracer.stage(
            "query_translation", {"method": strategy.query_translation, "query": semantic}
        ) as st:
            translation = translate_query(semantic, strategy.query_translation, deps.llm)
            st.usage = translation.token_usage
            if translation.error:
                st.notes.append(f"fell back to the original query: {translation.error}")
            st.output = {"rewrites": translation.rewrites}
    groups = translation.groups

    # 3. retrieve
    with tracer.stage(
        "retrieve",
        {"retriever": strategy.retriever, "fetch_k": strategy.fetch_k, "groups": len(groups)},
    ) as st:

        def _run(w: dict[str, Any] | None, p: Any) -> list[RankedLists]:
            return [
                retrieve_lists(strategy.retriever, deps.index, g, strategy.fetch_k, w, p)
                for g in groups
            ]

        group_lists = _run(where, predicate)
        if filters is not None and not any(h for gl in group_lists for h in gl.values()):
            st.notes.append("filters matched no chunks; retried without filters")
            group_lists = _run(None, None)
        for gi, gl in enumerate(group_lists):
            prefix = f"g{gi}/" if len(groups) > 1 else ""
            st.candidates.extend(_candidates({prefix + n: h for n, h in gl.items()}))
        st.output = {
            "lists": {n: len(h) for gl in group_lists for n, h in gl.items()},
        }

    # 4. fuse (per group)
    if any(len(gl) > 1 for gl in group_lists):
        with tracer.stage("fuse", {"method": "rrf", "k": deps.rrf_k}) as st:
            ranked = [fuse(gl, deps.rrf_k) for gl in group_lists]
            st.candidates = _candidates(
                {(f"rrf:g{i}" if len(ranked) > 1 else "rrf"): r for i, r in enumerate(ranked)}
            )
            st.output = {"top": _summary(ranked[0])}
    else:
        ranked = [fuse(gl) if gl else [] for gl in group_lists]

    # 5. rerank (per group, against the group's query)
    if strategy.rerank:
        with tracer.stage("rerank", {"candidates_per_group": strategy.rerank_k}) as st:
            if deps.reranker is None:
                st.notes.append("no reranker configured; kept fused order")
            else:
                try:
                    # The unscored tail stays behind the reranked head, so dedup and the
                    # per-paper cap in `select` can still fill top_k.
                    n = strategy.rerank_k
                    ranked = [
                        rerank(g.rerank_query, r[:n], deps.reranker) + r[n:]
                        for g, r in zip(groups, ranked, strict=True)
                    ]
                except RerankError as exc:
                    logger.warning("Reranking failed, keeping fused order: %s", exc)
                    st.notes.append(f"reranking failed ({exc}); kept fused order")
            st.candidates = _candidates(
                {(f"rerank:g{i}" if len(ranked) > 1 else "rerank"): r for i, r in enumerate(ranked)}
            )
            st.output = {"top": _summary(ranked[0])}

    # 6. select
    with tracer.stage(
        "select",
        {"top_k": strategy.top_k, "max_chunks_per_paper": strategy.max_chunks_per_paper},
    ) as st:
        merged = interleave(ranked) if len(ranked) > 1 else ranked[0]
        if len(ranked) > 1:
            st.notes.append(f"interleaved {len(ranked)} sub-question rankings")
        chunks = select_chunks(merged, strategy.top_k, strategy.max_chunks_per_paper)
        st.candidates = _candidates({"final": chunks})
        st.output = {"chunks": _summary(chunks, strategy.top_k)}

    trace = RetrievalTrace(
        strategy=strategy.name,
        question=q,
        semantic_query=semantic,
        rewrites=translation.rewrites,
        filters=filters,
        stages=tracer.stages,
        token_usage=tracer.usage(),
        total_latency_ms=(time.perf_counter() - t0) * 1000,
    )
    return RetrievalResult(question=q, strategy=strategy.name, chunks=chunks, trace=trace)


__all__ = [
    "RetrievalDeps",
    "RetrievalError",
    "StrategyConfig",
    "StrategyConfigError",
    "default_deps",
    "get_strategy",
    "load_strategies",
    "retrieve",
    "select_chunks",
]
