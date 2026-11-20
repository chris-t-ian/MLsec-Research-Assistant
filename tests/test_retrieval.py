"""Retrieval strategies: RRF, self-query filters, strategy config, query translation, retrieve()."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from research_assistant.config import PROJECT_ROOT
from research_assistant.events import RecordingSink
from research_assistant.ingest.index import ChunkIndex, ChunkMeta, tokenize
from research_assistant.llm import ChatStructuredLLM, LLMCallError
from research_assistant.models import Chunk, QueryFilters, ScoredChunk, TokenUsage
from research_assistant.retrieval.query_translation import (
    HypotheticalPassage,
    Paraphrases,
    SubQuestions,
    translate_query,
)
from research_assistant.retrieval.rerank import RerankError, rerank
from research_assistant.retrieval.retrievers import fuse, interleave, reciprocal_rank_fusion
from research_assistant.retrieval.self_query import (
    IndexCatalog,
    RawQueryFilters,
    SelfQueryOutput,
    classify_section,
    filters_to_predicate,
    filters_to_where,
    normalize_arxiv_id,
    self_query,
    validate_filters,
)
from research_assistant.retrieval.strategies import (
    RetrievalDeps,
    RetrievalError,
    StrategyConfig,
    StrategyConfigError,
    load_strategies,
    retrieve,
    select_chunks,
)

# --------------------------------------------------------------------------- helpers


def _chunk(
    cid: str,
    text: str = "",
    *,
    section: str = "Introduction",
    year: int = 2021,
    authors: Sequence[str] = ("Ada Lovelace",),
) -> Chunk:
    text = text or f"text of {cid}"
    return Chunk(
        id=cid,
        arxiv_id=cid.split(":")[0],
        title=f"Paper {cid.split(':')[0]}",
        authors=list(authors),
        year=year,
        section=section,
        text=text,
        char_start=0,
        char_end=len(text),
    )


def _hits(ids: Sequence[str], source: str = "dense") -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=_chunk(cid), score=1.0 / r, rank=r, source=source)
        for r, cid in enumerate(ids, start=1)
    ]


class FakeLLM:
    """StructuredLLM returning canned objects per schema; records every call."""

    def __init__(self, responses: dict[type[BaseModel], BaseModel | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, str]] = []

    def invoke_structured(
        self, schema: type[Any], system: str, user: str
    ) -> tuple[Any, TokenUsage]:
        self.calls.append((schema.__name__, system, user))
        resp = self.responses[schema]
        if isinstance(resp, Exception):
            raise resp
        return resp, TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)


# --------------------------------------------------------------------------- RRF


def test_rrf_math_matches_formula() -> None:
    lists = {"dense": _hits(["a:1", "b:1", "c:1"]), "bm25": _hits(["c:1", "a:1", "d:1"], "bm25")}
    fused = reciprocal_rank_fusion(lists, k=60)
    scores = {h.chunk.id: h.score for h in fused}
    assert scores["a:1"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c:1"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b:1"] == pytest.approx(1 / 62)
    assert scores["d:1"] == pytest.approx(1 / 63)
    assert [h.chunk.id for h in fused] == ["a:1", "c:1", "b:1", "d:1"]
    assert [h.rank for h in fused] == [1, 2, 3, 4]
    assert all(h.source == "rrf" for h in fused)


def test_rrf_keeps_per_stage_scores() -> None:
    dense = _hits(["a:1", "b:1"])
    bm25 = [ScoredChunk(chunk=_chunk("a:1"), score=12.5, rank=1, source="bm25")]
    top = reciprocal_rank_fusion({"dense:q0": dense, "bm25:q0": bm25})[0]
    assert top.chunk.id == "a:1"
    assert top.stage_scores == {"dense:q0": 1.0, "bm25:q0": 12.5, "rrf": pytest.approx(2 / 61)}


def test_rrf_uses_position_counts_duplicates_once_and_breaks_ties() -> None:
    # 'rank' fields are deliberately wrong: positions in the list are what count.
    lst = [
        ScoredChunk(chunk=_chunk("x:1"), score=0.9, rank=7, source="dense"),
        ScoredChunk(chunk=_chunk("x:1"), score=0.8, rank=8, source="dense"),
        ScoredChunk(chunk=_chunk("y:1"), score=0.7, rank=9, source="dense"),
    ]
    fused = reciprocal_rank_fusion({"l": lst}, k=10)
    assert [(h.chunk.id, h.score) for h in fused] == [
        ("x:1", pytest.approx(1 / 11)),
        ("y:1", pytest.approx(1 / 13)),
    ]
    # Equal fused scores: better single rank first, then id.
    tie = reciprocal_rank_fusion({"a": _hits(["p:1", "q:1"]), "b": _hits(["q:1", "p:1"])})
    assert [h.chunk.id for h in tie] == ["p:1", "q:1"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"a": lst}, k=0)


def test_fuse_single_list_is_passthrough() -> None:
    hits = _hits(["a:1", "b:1"])
    assert fuse({"dense:q0": hits}) == hits
    assert fuse({"a": hits, "b": hits})[0].source == "rrf"


def test_interleave_round_robin() -> None:
    out = interleave([_hits(["a:1", "b:1", "c:1"]), _hits(["b:1", "d:1"])])
    assert [h.chunk.id for h in out] == ["a:1", "b:1", "d:1", "c:1"]
    assert [h.rank for h in out] == [1, 2, 3, 4]
    assert interleave([]) == []


def test_select_chunks_dedupes_and_caps_per_paper() -> None:
    hits = [
        ScoredChunk(chunk=_chunk(cid, text), score=1.0, rank=1, source="rrf")
        for cid, text in [
            ("p:1", "alpha"),
            ("p:1", "alpha"),  # duplicate id
            ("q:1", "ALPHA "),  # same text, other chunk
            ("p:2", "beta"),
            ("p:3", "gamma"),  # third chunk of paper p -> capped at 2
            ("r:1", "delta"),
        ]
    ]
    out = select_chunks(hits, top_k=10, max_chunks_per_paper=2)
    assert [h.chunk.id for h in out] == ["p:1", "p:2", "r:1"]
    assert [h.rank for h in out] == [1, 2, 3]
    assert len(select_chunks(hits, top_k=1, max_chunks_per_paper=3)) == 1


# --------------------------------------------------------------------------- self-query


def _meta(cid: str, year: int, section: str, topic: str, authors: Sequence[str]) -> ChunkMeta:
    return ChunkMeta(cid, cid.split(":")[0], tuple(authors), year, section, topic, False)


_CT = ["Nicholas Carlini", "Florian Tramèr"]
METAS = [
    _meta("2012.07805:0", 2021, "Introduction", "data_extraction", _CT),
    _meta("2012.07805:1", 2021, "Evaluation > Results", "data_extraction", _CT),
    _meta("1610.05820:0", 2017, "Related Work", "membership_inference", ["Reza Shokri"]),
    _meta("1610.05820:1", 2017, "Our Attack Methodology", "membership_inference", ["Reza Shokri"]),
    _meta("2301.00001:0", 2023, "Background and Related Work", "membership_inference",
          ["Nicolas Carlini"]),
    ChunkMeta("9999.99999:0", "9999.99999", ("Evil Author",), 2024, "Intro", "other", True),
]  # fmt: skip


@pytest.fixture
def catalog() -> IndexCatalog:
    return IndexCatalog.from_metadata(METAS)


def test_classify_section() -> None:
    assert classify_section("Related Work") == "related_work"
    assert classify_section("Background and Related Work") == "related_work"
    assert classify_section("Evaluation > Experimental Setup") == "results"
    assert classify_section("Appendix: Additional Results") == "results"
    assert classify_section("Our Attack Methodology") == "method"
    assert classify_section("Introduction > Proposed approach") == "method"
    assert classify_section("Acknowledgements") == "any"


def test_catalog_excludes_quarantined_and_matches_authors(catalog: IndexCatalog) -> None:
    assert "9999.99999" not in catalog.arxiv_ids
    assert catalog.topics == {"data_extraction", "membership_inference"}
    assert catalog.match_author("Florian Tramer") == ["Florian Tramèr"]
    assert catalog.match_author("carlini") == ["Nicholas Carlini", "Nicolas Carlini"]
    assert catalog.match_author("N. Carlini") == ["Nicholas Carlini", "Nicolas Carlini"]
    assert catalog.match_author("R. Carlini") == []
    assert catalog.match_author("Nobody") == []


def test_normalize_arxiv_id() -> None:
    assert normalize_arxiv_id("arXiv:2012.07805v2") == "2012.07805"
    assert normalize_arxiv_id(" 1610.05820 ") == "1610.05820"
    assert normalize_arxiv_id("cs/0112017v1") == "cs/0112017"
    assert normalize_arxiv_id("not an id") is None


def test_validate_filters_keeps_known_values(catalog: IndexCatalog) -> None:
    raw = RawQueryFilters(
        year_min=2022,
        year_max=2020,
        topic="Data Extraction",
        authors=["Tramer", "Somebody Unknown"],
        arxiv_ids=["arXiv:2012.07805v3", "1234.56789"],
        section_type="results",
    )
    filters, notes = validate_filters(raw, catalog)
    assert filters == QueryFilters(
        year_min=2020,
        year_max=2022,
        topic="data_extraction",
        authors=["Florian Tramèr"],
        arxiv_ids=["2012.07805"],
        section_type="results",
    )
    assert any("swapped" in n for n in notes)
    assert any("Somebody Unknown" in n for n in notes)
    assert any("1234.56789" in n for n in notes)


@pytest.mark.parametrize(
    "raw",
    [
        RawQueryFilters(),
        RawQueryFilters(topic="quantum_cooking", authors=["Nobody"], arxiv_ids=["0000.00000"]),
        RawQueryFilters(year_min=1500, year_max=3000, section_type="abstract"),
        RawQueryFilters(section_type="any", topic="other"),  # 'other' not in this catalog
        RawQueryFilters(arxiv_ids=["9999.99999"]),  # quarantined-only paper
    ],
)
def test_invalid_or_empty_filters_become_none(catalog: IndexCatalog, raw: RawQueryFilters) -> None:
    filters, _ = validate_filters(raw, catalog)
    assert filters is None
    assert filters_to_where(filters, catalog) is None
    assert filters_to_predicate(filters, catalog) is None


def test_conflicting_authors_and_ids_drop_authors(catalog: IndexCatalog) -> None:
    raw = RawQueryFilters(authors=["Reza Shokri"], arxiv_ids=["2012.07805"])
    filters, notes = validate_filters(raw, catalog)
    assert filters is not None and filters.authors == [] and filters.arxiv_ids == ["2012.07805"]
    assert any("dropped the authors" in n for n in notes)


def test_filters_to_where_single_clause(catalog: IndexCatalog) -> None:
    assert filters_to_where(QueryFilters(year_min=2020), catalog) == {"year": {"$gte": 2020}}
    assert filters_to_where(QueryFilters(topic="membership_inference"), catalog) == {
        "topic": {"$eq": "membership_inference"}
    }


def test_filters_to_where_combines_with_and(catalog: IndexCatalog) -> None:
    f = QueryFilters(
        year_min=2018,
        year_max=2022,
        topic="data_extraction",
        authors=["Nicholas Carlini"],
        section_type="related_work",
    )
    assert filters_to_where(f, catalog) == {
        "$and": [
            {"year": {"$gte": 2018}},
            {"year": {"$lte": 2022}},
            {"topic": {"$eq": "data_extraction"}},
            {"arxiv_id": {"$in": ["2012.07805"]}},
            {"section": {"$in": ["Background and Related Work", "Related Work"]}},
        ]
    }


def test_authors_and_ids_intersect(catalog: IndexCatalog) -> None:
    f = QueryFilters(authors=["Nicholas Carlini", "Nicolas Carlini"], arxiv_ids=["2301.00001"])
    assert filters_to_where(f, catalog) == {"arxiv_id": {"$in": ["2301.00001"]}}


@pytest.mark.parametrize(
    "filters",
    [
        QueryFilters(year_min=2020),
        QueryFilters(year_max=2017, section_type="method"),
        QueryFilters(topic="membership_inference", section_type="related_work"),
        QueryFilters(authors=["Nicholas Carlini", "Nicolas Carlini"]),
        QueryFilters(arxiv_ids=["2012.07805"], section_type="results"),
    ],
)
def test_predicate_matches_where(catalog: IndexCatalog, filters: QueryFilters) -> None:
    """The BM25 predicate selects exactly the chunks the Chroma where-clause describes."""
    where = filters_to_where(filters, catalog)
    pred = filters_to_predicate(filters, catalog)
    assert where is not None and pred is not None

    def chroma_match(m: ChunkMeta, clause: dict[str, Any]) -> bool:
        if "$and" in clause:
            return all(chroma_match(m, c) for c in clause["$and"])
        ((field, cond),) = clause.items()
        ((op, val),) = cond.items()
        actual = getattr(m, field)
        return {
            "$gte": lambda: actual is not None and actual >= val,
            "$lte": lambda: actual is not None and actual <= val,
            "$eq": lambda: actual == val,
            "$in": lambda: actual in val,
        }[op]()

    visible = [m for m in METAS if not m.quarantined]
    assert [m.chunk_id for m in visible if pred(m)] == [
        m.chunk_id for m in visible if chroma_match(m, where)
    ]


def test_self_query_with_mocked_llm(catalog: IndexCatalog) -> None:
    out = SelfQueryOutput(
        semantic_query="training data extraction from language models",
        filters=RawQueryFilters(authors=["Carlini"], year_min=2021, topic="nonsense"),
    )
    llm = FakeLLM({SelfQueryOutput: out})
    res = self_query("What did Carlini show in 2021 about extraction?", llm, catalog)
    assert res.semantic_query == "training data extraction from language models"
    assert res.filters == QueryFilters(
        year_min=2021, authors=["Nicholas Carlini", "Nicolas Carlini"]
    )
    assert any("nonsense" in n for n in res.notes)
    assert res.token_usage.total_tokens == 15
    assert "<question>" in llm.calls[0][2]


def test_self_query_llm_failure_means_no_filter(catalog: IndexCatalog) -> None:
    llm = FakeLLM({SelfQueryOutput: LLMCallError("boom")})
    res = self_query("q?", llm, catalog)
    assert res.filters is None and res.semantic_query == "q?"


# --------------------------------------------------------------------------- strategies


def test_load_project_strategies() -> None:
    strategies = load_strategies(PROJECT_ROOT / "config" / "strategies.yaml")
    assert list(strategies) == [
        "baseline_dense",
        "hybrid",
        "hybrid_multiquery",
        "hybrid_hyde_rerank",
        "full",
    ]
    base, full = strategies["baseline_dense"], strategies["full"]
    assert (base.retriever, base.query_translation, base.rerank, base.use_filters) == (
        "dense",
        "none",
        False,
        False,
    )
    assert (full.retriever, full.query_translation, full.rerank, full.use_filters) == (
        "hybrid",
        "multi_query",
        True,
        True,
    )
    assert strategies["hybrid_hyde_rerank"].query_translation == "hyde"
    assert all(s.max_chunks_per_paper == 3 for s in strategies.values())
    assert all(s.rerank_k == 15 for s in strategies.values())


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("strategies: []", "no strategies"),
        ("foo: 1", "'strategies' list"),
        ("strategies:\n  - {name: a}\n  - {name: a}", "duplicate"),
        ("strategies:\n  - {name: a, retriever: magic}", "invalid strategy"),
        ("strategies:\n  - {name: a, unknown_field: 1}", "invalid strategy"),
        ("strategies:\n  - {name: a, top_k: 50, fetch_k: 10}", "invalid strategy"),
        ("strategies:\n  - {name: a, rerank: true, top_k: 8, rerank_k: 5}", "invalid strategy"),
        ("strategies:\n  - {name: a, rerank: true, fetch_k: 10, rerank_k: 20}", "invalid strategy"),
        ("strategies: [", "Cannot read"),
    ],
)
def test_load_strategies_rejects_bad_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(content)
    with pytest.raises(StrategyConfigError, match=message):
        load_strategies(path)


def test_strategy_defaults_apply(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("defaults: {fetch_k: 12, top_k: 4}\nstrategies:\n  - {name: a, top_k: 2}\n")
    a = load_strategies(path)["a"]
    assert (a.fetch_k, a.top_k, a.max_chunks_per_paper) == (12, 2, 3)


# --------------------------------------------------------------------------- query translation


def test_multi_query_with_mocked_llm() -> None:
    llm = FakeLLM(
        {Paraphrases: Paraphrases(paraphrases=["  p one ", "P ONE", "orig q", "p two", "p three"])}
    )
    res = translate_query("orig q", "multi_query", llm)
    assert res.rewrites == ["p one", "p two", "p three"]  # deduped, original excluded, max 3
    assert len(res.groups) == 1
    assert res.groups[0].text_queries == ["orig q", "p one", "p two", "p three"]
    assert res.groups[0].rerank_query == "orig q"
    assert res.token_usage.total_tokens == 15 and res.error is None


def test_hyde_embeds_passage_instead_of_question() -> None:
    llm = FakeLLM({HypotheticalPassage: HypotheticalPassage(passage="We clip gradients ...")})
    res = translate_query("How does DP-SGD work?", "hyde", llm)
    group = res.groups[0]
    assert group.dense_queries() == ["We clip gradients ..."]
    assert group.text_queries == ["How does DP-SGD work?"]  # BM25 still uses the question


def test_decompose_one_group_per_sub_question() -> None:
    subs = SubQuestions(sub_questions=["What is LiRA?", "What is RMIA?", "What is LiRA?"])
    res = translate_query("Compare LiRA and RMIA", "decompose", FakeLLM({SubQuestions: subs}))
    assert [g.rerank_query for g in res.groups] == ["What is LiRA?", "What is RMIA?"]
    single = SubQuestions(sub_questions=["What is LiRA?"])
    res = translate_query("What is LiRA?", "decompose", FakeLLM({SubQuestions: single}))
    assert len(res.groups) == 1 and res.groups[0].text_queries == ["What is LiRA?"]


@pytest.mark.parametrize("method", ["multi_query", "hyde", "decompose"])
def test_translation_falls_back_on_llm_failure(method: str) -> None:
    llm = FakeLLM(
        {s: LLMCallError("down") for s in (Paraphrases, HypotheticalPassage, SubQuestions)}
    )
    res = translate_query("q?", method, llm)  # type: ignore[arg-type]
    assert res.groups[0].text_queries == ["q?"] and res.error == "down"
    assert translate_query("q?", method, None).error == "no LLM configured"  # type: ignore[arg-type]


def test_translation_none_makes_no_llm_call() -> None:
    llm = FakeLLM({})
    assert translate_query("q?", "none", llm).groups[0].text_queries == ["q?"]
    assert llm.calls == []


# --------------------------------------------------------------------------- LLM wrapper


class _FakeChatModel:
    """Stands in for a chat model: with_structured_output returns canned raw outputs."""

    def __init__(self, outputs: list[dict[str, Any] | Exception]) -> None:
        self.outputs = outputs
        self.kwargs: dict[str, Any] = {}

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        self.kwargs = kwargs

        def _invoke(_: Any) -> dict[str, Any]:
            out = self.outputs.pop(0)
            if isinstance(out, Exception):
                raise out
            return out

        return RunnableLambda(_invoke)


def _raw(tokens: int) -> AIMessage:
    return AIMessage(
        content="",
        usage_metadata={"input_tokens": tokens, "output_tokens": 1, "total_tokens": tokens + 1},
    )


def test_chat_structured_llm_retries_parse_failures_and_sums_usage() -> None:
    good = Paraphrases(paraphrases=["a"])
    model = _FakeChatModel(
        [
            {"raw": _raw(10), "parsed": None, "parsing_error": ValueError("bad json")},
            {"raw": _raw(20), "parsed": good, "parsing_error": None},
        ]
    )
    llm = ChatStructuredLLM(model, max_attempts=3, backoff_s=0)  # type: ignore[arg-type]
    parsed, usage = llm.invoke_structured(Paraphrases, "sys", "user")
    assert parsed == good
    assert usage == TokenUsage(input_tokens=30, output_tokens=2, total_tokens=32)
    assert model.kwargs == {"method": "function_calling", "include_raw": True}


def test_chat_structured_llm_raises_typed_error() -> None:
    bad = {"raw": _raw(1), "parsed": None, "parsing_error": ValueError("x")}
    llm = ChatStructuredLLM(_FakeChatModel([bad, bad]), max_attempts=2, backoff_s=0)  # type: ignore[arg-type]
    with pytest.raises(LLMCallError, match="Paraphrases"):
        llm.invoke_structured(Paraphrases, "sys", "user")
    llm = ChatStructuredLLM(_FakeChatModel([PermissionError("401")]), backoff_s=0)  # type: ignore[arg-type]
    with pytest.raises(LLMCallError, match="PermissionError"):
        llm.invoke_structured(Paraphrases, "sys", "user")


# --------------------------------------------------------------------------- rerank


class FakeReranker:
    """Scores by number of query tokens in the text."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append(query)
        if self.fail:
            raise RerankError("model missing")
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(t)))) for t in texts]


def test_rerank_reorders_and_records_score() -> None:
    hits = [
        ScoredChunk(chunk=_chunk("a:1", "nothing relevant"), score=0.9, rank=1, source="rrf"),
        ScoredChunk(chunk=_chunk("b:1", "shadow models attack"), score=0.5, rank=2, source="rrf"),
    ]
    out = rerank("shadow models", hits, FakeReranker())
    assert [h.chunk.id for h in out] == ["b:1", "a:1"]
    assert out[0].source == "rerank" and out[0].stage_scores["rerank"] == out[0].score


# --------------------------------------------------------------------------- retrieve()


class HashEmbedder:
    """Deterministic offline embedder: L2-normalized hashed bag of words."""

    model_name = "test/hash-bow"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 256
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


CORPUS = {
    "data_extraction": [
        _chunk("2012.07805:0:0", "Language models memorize training data and an adversary can "
               "extract verbatim sequences.", year=2021, authors=["Nicholas Carlini"]),
        _chunk("2012.07805:1:0", "Extraction results: we recover hundreds of memorized "
               "examples.", section="Evaluation > Results", year=2021,
               authors=["Nicholas Carlini"]),
    ],
    "membership_inference": [
        _chunk("1610.05820:0:0", "Shadow models train an attack model for membership inference.",
               year=2017, authors=["Reza Shokri"]),
        _chunk("1610.05820:1:0", "Membership inference with shadow models on ML-as-a-service.",
               year=2017, authors=["Reza Shokri"]),
        _chunk("1610.05820:2:0", "Shadow models mimic the target model behaviour.",
               year=2017, authors=["Reza Shokri"]),
        _chunk("1610.05820:3:0", "Shadow models defenses discussed in related work.",
               section="Related Work", year=2017, authors=["Reza Shokri"]),
    ],
    "differential_privacy": [
        _chunk("1607.00133:0:0", "DP-SGD clips per-example gradients and adds Gaussian noise; "
               "the moments accountant tracks privacy loss.", year=2016,
               authors=["Martin Abadi"]),
    ],
}  # fmt: skip


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> ChunkIndex:
    idx = ChunkIndex(tmp_path_factory.mktemp("idx"), HashEmbedder(), autosave=False)  # type: ignore[arg-type]
    for topic, chunks in CORPUS.items():
        idx.upsert(chunks, topic)
    return idx


def _strategy(**kw: Any) -> StrategyConfig:
    return StrategyConfig(**{"name": "t", "fetch_k": 10, "top_k": 5, "rerank_k": 10, **kw})


def test_retrieve_baseline_dense_emits_steps(index: ChunkIndex) -> None:
    sink = RecordingSink()
    deps = RetrievalDeps(index)
    res = retrieve(
        "How do shadow models enable membership inference?",
        _strategy(retriever="dense"),
        sink,
        deps,
    )
    assert res.chunks and res.chunks[0].chunk.arxiv_id == "1610.05820"
    assert sum(c.chunk.arxiv_id == "1610.05820" for c in res.chunks) == 3  # cap per paper
    assert [s.stage for s in res.trace.stages] == ["retrieve", "select"]
    started = [e.data["step_id"] for e in sink.events if e.kind == "step_started"]
    finished = [e.data["step_id"] for e in sink.events if e.kind == "step_finished"]
    assert started == finished == ["retrieval:t:retrieve", "retrieval:t:select"]
    assert all(s.latency_ms >= 0 for s in res.trace.stages)
    assert res.trace.token_usage == TokenUsage()


def test_retrieve_full_pipeline_with_mocks(index: ChunkIndex) -> None:
    llm = FakeLLM(
        {
            SelfQueryOutput: SelfQueryOutput(
                semantic_query="memorized training data extraction",
                filters=RawQueryFilters(authors=["Carlini"], year_min=2020),
            ),
            Paraphrases: Paraphrases(paraphrases=["verbatim extraction of memorized text"]),
        }
    )
    reranker = FakeReranker()
    deps = RetrievalDeps(index, llm=llm, reranker=reranker)
    strategy = _strategy(query_translation="multi_query", use_filters=True, rerank=True)
    sink = RecordingSink()
    res = retrieve("What did Carlini find about extraction since 2020?", strategy, sink, deps)

    assert [s.stage for s in res.trace.stages] == [
        "self_query", "query_translation", "retrieve", "fuse", "rerank", "select",
    ]  # fmt: skip
    assert {c.chunk.arxiv_id for c in res.chunks} == {"2012.07805"}  # filter applied
    assert res.trace.filters == QueryFilters(year_min=2020, authors=["Nicholas Carlini"])
    assert res.trace.semantic_query == "memorized training data extraction"
    assert res.trace.rewrites == ["verbatim extraction of memorized text"]
    assert res.trace.token_usage.total_tokens == 30  # two LLM calls
    assert reranker.calls == ["memorized training data extraction"]
    top = res.chunks[0]
    assert {"rrf", "rerank"} <= set(top.stage_scores)
    assert any(k.startswith("dense:") for k in top.stage_scores)
    retrieve_stage = res.trace.stages[2]
    assert {c.list_name for c in retrieve_stage.candidates} >= {"dense:q0", "bm25:q1"}
    assert len([e for e in sink.events if e.kind == "step_started"]) == 6


def test_retrieve_filters_matching_nothing_fall_back(index: ChunkIndex) -> None:
    llm = FakeLLM(
        {
            SelfQueryOutput: SelfQueryOutput(
                semantic_query="shadow models",
                filters=RawQueryFilters(topic="membership_inference", year_min=2030),
            )
        }
    )
    res = retrieve(
        "shadow models after 2030", _strategy(use_filters=True), None, RetrievalDeps(index, llm=llm)
    )
    assert res.chunks
    assert any("retried without filters" in n for n in res.trace.stages[1].notes)


def test_retrieve_decompose_and_hyde(index: ChunkIndex) -> None:
    llm = FakeLLM(
        {
            SubQuestions: SubQuestions(
                sub_questions=["How does DP-SGD add noise?", "How do shadow models work?"]
            ),
            HypotheticalPassage: HypotheticalPassage(passage="DP-SGD clips gradients."),
        }
    )
    deps = RetrievalDeps(index, llm=llm, reranker=FakeReranker())
    res = retrieve(
        "DP-SGD noise and shadow models?",
        _strategy(query_translation="decompose", rerank=True),
        None,
        deps,
    )
    papers = [c.chunk.arxiv_id for c in res.chunks[:2]]
    assert set(papers) == {"1607.00133", "1610.05820"}  # both sub-questions covered on top
    assert any("interleaved 2" in n for n in res.trace.stages[-1].notes)

    res = retrieve("How does DP-SGD work?", _strategy(query_translation="hyde"), None, deps)
    assert res.chunks[0].chunk.arxiv_id == "1607.00133"
    assert any(k.startswith("dense_hyde:") for k in res.chunks[0].stage_scores)


def test_retrieve_reranks_only_rerank_k_and_keeps_tail(index: ChunkIndex) -> None:
    class CountingReranker(FakeReranker):
        def score(self, query: str, texts: Sequence[str]) -> list[float]:
            self.sizes.append(len(texts))
            return [0.0] * len(texts)

    reranker = CountingReranker()
    reranker.sizes = []  # type: ignore[attr-defined]
    strategy = _strategy(rerank=True, top_k=2, rerank_k=2, max_chunks_per_paper=1)
    res = retrieve(
        "shadow models memorized gradients", strategy, None, RetrievalDeps(index, reranker=reranker)
    )
    assert reranker.sizes == [2]  # type: ignore[attr-defined]
    rerank_stage = next(s for s in res.trace.stages if s.stage == "rerank")
    assert [c.list_name for c in rerank_stage.candidates][:2] == ["rerank", "rerank"]
    assert len(res.chunks) == 2  # tail fills top_k despite the per-paper cap


def test_retrieve_survives_reranker_failure(index: ChunkIndex) -> None:
    deps = RetrievalDeps(index, reranker=FakeReranker(fail=True))
    res = retrieve("shadow models", _strategy(rerank=True), None, deps)
    assert res.chunks
    rerank_stage = next(s for s in res.trace.stages if s.stage == "rerank")
    assert any("reranking failed" in n for n in rerank_stage.notes)


def test_retrieve_rejects_bad_questions(index: ChunkIndex) -> None:
    deps = RetrievalDeps(index, max_question_chars=20)
    with pytest.raises(RetrievalError):
        retrieve("   ", _strategy(), None, deps)
    with pytest.raises(RetrievalError, match="too long"):
        retrieve("x" * 21, _strategy(), None, deps)
