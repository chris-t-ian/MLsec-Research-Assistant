"""Dense, BM25 and hybrid retrieval, with our own Reciprocal Rank Fusion.

Each retriever returns *named ranked lists* (``"dense:q0"``, ``"bm25:q1"``, ...), one per
query and retriever, so every stage's ranks and scores stay visible in the trace. Hybrid
retrieval and multi-query retrieval are both fused with :func:`reciprocal_rank_fusion`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from research_assistant.ingest.index import ChunkMeta, ChunkPredicate
from research_assistant.models import RetrieverKind, ScoredChunk
from research_assistant.retrieval.query_translation import QueryGroup

DEFAULT_RRF_K = 60


class SearchIndex(Protocol):
    """The part of :class:`~research_assistant.ingest.index.ChunkIndex` retrieval needs."""

    def dense_search(
        self,
        query: str,
        k: int,
        include_quarantined: bool = False,
        where: dict[str, Any] | None = None,
        query_is_passage: bool = False,
    ) -> list[ScoredChunk]: ...

    def bm25_search(
        self,
        query: str,
        k: int,
        include_quarantined: bool = False,
        predicate: ChunkPredicate | None = None,
    ) -> list[ScoredChunk]: ...

    def metadata(self) -> list[ChunkMeta]: ...


RankedLists = dict[str, list[ScoredChunk]]


def reciprocal_rank_fusion(
    lists: Mapping[str, Sequence[ScoredChunk]],
    k: int = DEFAULT_RRF_K,
    score_name: str = "rrf",
) -> list[ScoredChunk]:
    """Fuse ranked lists: ``score(d) = sum over lists containing d of 1 / (k + rank)``.

    Ranks are the 1-based positions in each list (the ``rank`` field is ignored, so lists
    may be pre-filtered). A chunk appearing twice in one list counts once, at its best
    position. The fused hits keep every input score in ``stage_scores`` under the list name
    and their fused score under ``score_name``. Ties are broken by best single rank, then id.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, ScoredChunk] = {}
    stage_scores: dict[str, dict[str, float]] = {}
    for name, hits in lists.items():
        seen: set[str] = set()
        for pos, hit in enumerate(hits, start=1):
            cid = hit.chunk.id
            if cid in seen:
                continue
            seen.add(cid)
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + pos)
            best_rank[cid] = min(best_rank.get(cid, pos), pos)
            first_seen.setdefault(cid, hit)
            scores = stage_scores.setdefault(cid, {})
            scores.update(hit.stage_scores)
            scores[name] = hit.score
    order = sorted(fused, key=lambda c: (-fused[c], best_rank[c], c))
    return [
        ScoredChunk(
            chunk=first_seen[cid].chunk,
            score=fused[cid],
            rank=rank,
            source=score_name,
            stage_scores={**stage_scores[cid], score_name: fused[cid]},
        )
        for rank, cid in enumerate(order, start=1)
    ]


def interleave(lists: Sequence[Sequence[ScoredChunk]]) -> list[ScoredChunk]:
    """Round-robin merge (1st of each list, then 2nd, ...), keeping the first occurrence.

    Used for decomposed questions, so every sub-question's best hits reach the top. (RRF
    would favour chunks that rank moderately for several sub-questions.) Scores are kept;
    ranks are renumbered.
    """
    out: list[ScoredChunk] = []
    seen: set[str] = set()
    for pos in range(max((len(lst) for lst in lists), default=0)):
        for lst in lists:
            if pos < len(lst) and lst[pos].chunk.id not in seen:
                seen.add(lst[pos].chunk.id)
                out.append(lst[pos].model_copy(update={"rank": len(out) + 1}))
    return out


def _with_stage(hits: list[ScoredChunk], name: str) -> list[ScoredChunk]:
    return [h.model_copy(update={"stage_scores": {**h.stage_scores, name: h.score}}) for h in hits]


def dense_lists(
    index: SearchIndex,
    group: QueryGroup,
    k: int,
    where: dict[str, Any] | None = None,
) -> RankedLists:
    """One dense list per query; HyDE passages are embedded as documents."""
    as_passage = bool(group.passages)
    prefix = "dense_hyde" if as_passage else "dense"
    return {
        f"{prefix}:q{i}": _with_stage(
            index.dense_search(q, k, where=where, query_is_passage=as_passage), f"{prefix}:q{i}"
        )
        for i, q in enumerate(group.dense_queries())
    }


def bm25_lists(
    index: SearchIndex,
    group: QueryGroup,
    k: int,
    predicate: ChunkPredicate | None = None,
) -> RankedLists:
    return {
        f"bm25:q{i}": _with_stage(index.bm25_search(q, k, predicate=predicate), f"bm25:q{i}")
        for i, q in enumerate(group.text_queries)
    }


def retrieve_lists(
    kind: RetrieverKind,
    index: SearchIndex,
    group: QueryGroup,
    k: int,
    where: dict[str, Any] | None = None,
    predicate: ChunkPredicate | None = None,
) -> RankedLists:
    """Ranked lists for one query group: dense, BM25 or both (hybrid)."""
    lists: RankedLists = {}
    if kind in ("dense", "hybrid"):
        lists.update(dense_lists(index, group, k, where))
    if kind in ("bm25", "hybrid"):
        lists.update(bm25_lists(index, group, k, predicate))
    return lists


def fuse(lists: Mapping[str, Sequence[ScoredChunk]], k: int = DEFAULT_RRF_K) -> list[ScoredChunk]:
    """RRF for several lists; a single list is returned unchanged (native scores kept)."""
    if len(lists) == 1:
        return list(next(iter(lists.values())))
    return reciprocal_rank_fusion(lists, k)


__all__ = [
    "DEFAULT_RRF_K",
    "RankedLists",
    "SearchIndex",
    "bm25_lists",
    "dense_lists",
    "fuse",
    "interleave",
    "reciprocal_rank_fusion",
    "retrieve_lists",
]
