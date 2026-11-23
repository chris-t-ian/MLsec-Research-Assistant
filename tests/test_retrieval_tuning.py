"""Retrieval-only tuning: grid expansion, rewriter cache, run/resume, metrics, summary."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from research_assistant.abtest.gold import GoldQuestion
from research_assistant.abtest.retrieval_tuning import (
    CachedLLM,
    CachedReranker,
    RecordStore,
    RetrievalRecord,
    TuningError,
    TuningManifest,
    Variant,
    check_split_selection,
    expand_grid,
    gold_coverage,
    manifest_configs,
    parse_grid,
    render_report,
    run_tuning,
    summarize_tuning,
    tuning_metrics,
    variants_from_manifest,
)
from research_assistant.llm import LLMCallError
from research_assistant.models import (
    Chunk,
    RetrievalResult,
    RetrievalTrace,
    ScoredChunk,
    TokenUsage,
)
from research_assistant.retrieval.strategies import StrategyConfig

BASE = StrategyConfig(name="baseline_dense", retriever="dense", fetch_k=30, top_k=8)
RERANK = StrategyConfig(
    name="rr", retriever="hybrid", rerank=True, fetch_k=30, top_k=8, rerank_k=15
)


def q(qid: str, gold: list[str], qtype: str = "factual") -> GoldQuestion:
    return GoldQuestion(
        id=qid,
        question=f"question {qid}?",
        type=qtype,  # type: ignore[arg-type]
        gold_arxiv_ids=gold,
        key_facts=["f"],
        split="dev",
    )


def result(papers: list[str], cfg: StrategyConfig) -> RetrievalResult:
    chunks = [
        ScoredChunk(
            chunk=Chunk(
                id=f"{p}#{i}", arxiv_id=p, title="t", text="word " * 10, char_start=0, char_end=1
            ),
            score=1.0,
            rank=i + 1,
            source="dense",
        )
        for i, p in enumerate(papers[: cfg.top_k])
    ]
    trace = RetrievalTrace(strategy=cfg.name, question="q", semantic_query="q")
    return RetrievalResult(question="q", strategy=cfg.name, chunks=chunks, trace=trace)


def words(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- grid


def test_parse_grid() -> None:
    assert parse_grid(["top_k=5,8,8", "fetch_k=60"]) == {"top_k": [5, 8], "fetch_k": [60]}
    for bad in (["top_k"], ["temperature=1"], ["top_k=a"], ["top_k=0"], ["top_k=5", "top_k=8"]):
        with pytest.raises(TuningError):
            parse_grid(bad)


def test_expand_grid_includes_base_skips_invalid_and_rerank_k_only_for_rerankers() -> None:
    variants, skipped = expand_grid([BASE, RERANK], {"top_k": [8, 40], "rerank_k": [15, 25]})
    names = [v.name for v in variants]
    # base unchanged keeps its name; top_k=8 equals the base value, so it is not a new variant
    assert "baseline_dense" in names and "rr" in names
    assert "baseline_dense__top_k40" not in names  # top_k > fetch_k
    assert not any(n.startswith("baseline_dense__rerank_k") for n in names)
    assert "rr__rerank_k25" in names
    assert any("baseline_dense__top_k40" in s for s in skipped)
    assert len(names) == len(set(names))
    v = next(v for v in variants if v.name == "rr__rerank_k25")
    assert v.params == {"rerank_k": 25} and v.config.rerank_k == 25 and v.base == "rr"


def test_manifest_roundtrip_of_variants() -> None:
    variants, _ = expand_grid([BASE], {"top_k": [5]})
    again = variants_from_manifest(
        TuningManifest.model_construct(configs=manifest_configs(variants))
    )
    assert again == variants


def test_test_split_only_for_final_confirmation_without_grid_or_subset() -> None:
    check_split_selection("dev", final=False, grid=True, subset=True)
    check_split_selection("test", final=True, grid=False, subset=False)
    for split, final, grid, subset in [
        ("test", False, False, False),  # test needs --final
        ("test", True, True, False),  # no tuning on test
        ("test", True, False, True),  # whole test split
        ("dev", True, False, False),  # --final is for test only
        ("train", False, False, False),
    ]:
        with pytest.raises(TuningError):
            check_split_selection(split, final=final, grid=grid, subset=subset)


def test_confirmation_is_logged_as_retrieval_test_run(tmp_path: Path) -> None:
    import json

    from research_assistant.abtest.runner import append_test_log

    manifest = TuningManifest(
        run_id="confirm-x",
        created_at="now",
        split="test",
        final=True,
        question_ids=["1", "2"],
        strategies=["baseline_dense", "full_tuned"],
        grid={},
        reference="baseline_dense",
        rewriter_model="m",
        seed=0,
        n_resamples=10,
        git_commit="abc",
        git_dirty=False,
        gold_file="g",
        gold_sha256="s",
        configs=[],
    )
    log = tmp_path / "test_runs.log"
    append_test_log(log, manifest, tmp_path, "start")
    entry = json.loads(log.read_text())
    assert entry["kind"] == "retrieval" and entry["run_id"] == "confirm-x"
    assert entry["n_questions"] == 2
    assert "Retrieval confirmation" in render_report(manifest, [])


# --------------------------------------------------------------------------- cache


class Out(BaseModel):
    text: str


class FakeLLM:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def invoke_structured(self, schema, system, user):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.fail:
            raise LLMCallError("down")
        return schema(text=user.upper()), TokenUsage(input_tokens=10, output_tokens=2)


def test_cached_llm_hits_persist_and_failures_are_not_cached(tmp_path: Path) -> None:
    inner = FakeLLM()
    llm = CachedLLM(inner, tmp_path / "cache.jsonl")
    a, usage = llm.invoke_structured(Out, "sys", "hello")
    b, usage2 = llm.invoke_structured(Out, "sys", "hello")
    assert a == b == Out(text="HELLO") and inner.calls == 1
    assert usage.input_tokens == 10 and usage2 == TokenUsage()
    assert (llm.misses, llm.hits) == (1, 1)

    reloaded = CachedLLM(FakeLLM(fail=True), tmp_path / "cache.jsonl")
    assert reloaded.invoke_structured(Out, "sys", "hello")[0] == Out(text="HELLO")
    with pytest.raises(LLMCallError):
        reloaded.invoke_structured(Out, "sys", "other")
    assert reloaded.failures == 1
    assert len((tmp_path / "cache.jsonl").read_text().splitlines()) == 1


class CountingReranker:
    def __init__(self) -> None:
        self.pairs = 0

    def score(self, query: str, texts):  # type: ignore[no-untyped-def]
        self.pairs += len(texts)
        return [float(len(t)) for t in texts]


def test_cached_reranker_scores_each_pair_once() -> None:
    inner = CountingReranker()
    rr = CachedReranker(inner)
    assert rr.score("q", ["aa", "b"]) == [2.0, 1.0]
    assert rr.score("q", ["b", "ccc", "aa"]) == [1.0, 3.0, 2.0]
    assert rr.score("other", ["b"]) == [1.0]
    assert inner.pairs == 4


# --------------------------------------------------------------------------- metrics


def test_gold_coverage_and_metrics() -> None:
    assert gold_coverage(["a", "b", "a"], ["a", "c"]) == 0.5
    assert gold_coverage(["a"], []) is None
    rec = RetrievalRecord(
        question_id="1",
        config="c",
        base_strategy="c",
        papers=["x", "y", "a", "c"],
        chunk_ids=["1", "2", "3", "4"],
        context_tokens=400,
        latency_s=0.5,
    )
    m = tuning_metrics(rec, q("1", ["a", "c"]))
    assert m["gold_coverage"] == 1.0
    assert m["recall@3"] == 1.0
    assert m["mrr"] == pytest.approx(1 / 3)
    assert m["context_tokens"] == 400.0


# --------------------------------------------------------------------------- run


def _variants() -> list[Variant]:
    variants, _ = expand_grid([BASE], {"top_k": [2]})
    return variants


def test_run_stores_skips_oos_and_resumes(tmp_path: Path) -> None:
    questions = [q("1", ["a"]), q("2", ["b"]), q("3", [], "out_of_scope")]
    calls: list[tuple[str, str]] = []

    def fn(question: str, cfg: StrategyConfig) -> RetrievalResult:
        calls.append((question, cfg.name))
        return result(["z", "a", "b"], cfg)

    store = RecordStore(tmp_path / "r.jsonl")
    llm = CachedLLM(FakeLLM(), tmp_path / "c.jsonl")
    out = run_tuning(questions, _variants(), fn, store, llm, words)
    assert out.done == 4 and len(calls) == 4  # 2 in-scope questions x 2 variants
    recs = store.load()
    assert recs[("baseline_dense__top_k2", "1")].papers == ["z", "a"]
    assert recs[("baseline_dense", "1")].context_tokens == 30

    calls.clear()
    again = run_tuning(questions, _variants(), fn, store, llm, words)
    assert calls == [] and again.skipped == 4


def test_run_does_not_store_retrievals_with_failed_rewriter_and_stops(tmp_path: Path) -> None:
    questions = [q(str(i), ["a"]) for i in range(5)]
    llm = CachedLLM(FakeLLM(fail=True), tmp_path / "c.jsonl")

    def fn(question: str, cfg: StrategyConfig) -> RetrievalResult:
        with pytest.raises(LLMCallError):  # the pipeline catches this and falls back
            llm.invoke_structured(Out, "s", question)
        return result(["a"], cfg)

    store = RecordStore(tmp_path / "r.jsonl")
    with pytest.raises(TuningError):
        run_tuning(questions, _variants(), fn, store, llm, words, max_consecutive_failures=3)
    assert store.load() == {}


# --------------------------------------------------------------------------- summary


def test_summary_sorts_by_coverage_marks_pareto_and_pairs_with_reference(
    tmp_path: Path,
) -> None:
    questions = [q(str(i), ["a", "b"]) for i in range(4)]
    variants, _ = expand_grid([BASE], {"top_k": [1, 4]})
    store = RecordStore(tmp_path / "r.jsonl")

    def fn(question: str, cfg: StrategyConfig) -> RetrievalResult:
        return result(["a", "x", "b", "y", "z", "w", "v", "u"], cfg)

    run_tuning(questions, variants, fn, store, CachedLLM(None, tmp_path / "c.jsonl"), words)
    rows = summarize_tuning(store.load(), questions, variants, "baseline_dense", n_resamples=200)
    by = {r.config: r for r in rows}
    # top_k=4 already finds both gold papers with fewer tokens than top_k=8
    assert by["baseline_dense__top_k4"].estimates["gold_coverage"].mean == 1.0
    assert by["baseline_dense__top_k1"].estimates["gold_coverage"].mean == 0.5
    assert rows[0].config == "baseline_dense__top_k4"
    assert by["baseline_dense__top_k4"].pareto and by["baseline_dense__top_k1"].pareto
    assert not by["baseline_dense"].pareto  # same coverage as top_k4, more tokens
    diff = by["baseline_dense__top_k1"].vs_reference["gold_coverage"]
    assert diff.n == 4 and diff.mean == -0.5
    assert by["baseline_dense"].vs_reference == {}

    manifest = TuningManifest(
        run_id="tuning-x",
        created_at="now",
        split="dev",
        question_ids=[x.id for x in questions],
        strategies=["baseline_dense"],
        grid={"top_k": [1, 4]},
        reference="baseline_dense",
        rewriter_model="m",
        seed=0,
        n_resamples=200,
        git_commit="abc",
        git_dirty=False,
        gold_file="g",
        gold_sha256="s",
        configs=manifest_configs(variants),
    )
    report = render_report(manifest, rows)
    assert "baseline_dense__top_k4" in report and "Pareto" in report
    assert "confirm" in report  # points to the one-time test-split confirmation
