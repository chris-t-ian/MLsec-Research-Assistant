"""Per-question metrics. Every function returns ``None`` when the metric is undefined for a
question (e.g. retrieval metrics on out-of-scope questions); undefined values are left out of
means and of paired comparisons, and the report states ``n``.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from research_assistant.abtest.gold import GoldQuestion
from research_assistant.abtest.records import EvalRecord

MetricGroup = Literal["retrieval", "answer", "abstention", "cost"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    group: MetricGroup
    higher_is_better: bool
    unit: str = ""  # "" for rates in [0, 1]
    description: str = ""


def recall_at_k(retrieved_papers: Sequence[str], gold: Collection[str], k: int) -> float | None:
    """Paper-level hit: 1.0 if a gold paper is among the papers of the top-``k`` chunks."""
    if not gold:
        return None
    if k < 1:
        raise ValueError("k must be >= 1")
    gold_set = set(gold)
    return 1.0 if any(p in gold_set for p in retrieved_papers[:k]) else 0.0


def reciprocal_rank(retrieved_papers: Sequence[str], gold: Collection[str]) -> float | None:
    """1 / rank of the first chunk from a gold paper (0.0 if none); mean over questions = MRR."""
    if not gold:
        return None
    gold_set = set(gold)
    for rank, paper in enumerate(retrieved_papers, start=1):
        if paper in gold_set:
            return 1.0 / rank
    return 0.0


def key_fact_coverage(covered: Sequence[bool]) -> float | None:
    """Share of key facts the judge marked as covered."""
    if not covered:
        return None
    return sum(1 for c in covered if c) / len(covered)


def gold_citation_rate(cited_papers: Sequence[str], gold: Collection[str]) -> float | None:
    """Share of distinct cited papers that are in the gold set (None without citations)."""
    distinct = set(cited_papers)
    if not gold or not distinct:
        return None
    return len(distinct & set(gold)) / len(distinct)


def correct_refusal(correct: bool | None) -> float | None:
    return None if correct is None else float(correct)


class ClaimVerifier(Protocol):
    """Hook for the supported-claim rate: share of answer claims that the cited chunks
    support. Return None when the rate cannot be computed for a record."""

    def supported_claim_rate(self, record: EvalRecord, question: GoldQuestion) -> float | None: ...


class StoredClaimRate:
    """:class:`ClaimVerifier` reading the claim check stored on the record (computed by the
    agent's verify node, or after the turn for runs without verification).

    Supported claims / (cited claims + uncited factual sentences); partially supported
    claims count as not supported. None if the check did not run or found no claims."""

    def supported_claim_rate(self, record: EvalRecord, question: GoldQuestion) -> float | None:
        if not record.claims_checked or record.claim_error is not None:
            return None
        return record.supported_rate


STORED_CLAIM_RATE = StoredClaimRate()


def metric_specs(k: int) -> list[MetricSpec]:
    return [
        MetricSpec(
            f"recall@{k}",
            f"Recall@{k} (paper)",
            "retrieval",
            True,
            description=f"a gold paper among the papers of the top-{k} retrieved chunks",
        ),
        MetricSpec(
            "mrr",
            "MRR (paper)",
            "retrieval",
            True,
            description="1 / rank of the first retrieved chunk from a gold paper",
        ),
        MetricSpec(
            "key_fact_coverage",
            "Key-fact coverage",
            "answer",
            True,
            description="share of key facts the judge marked as covered",
        ),
        MetricSpec(
            "gold_citation_rate",
            "Gold-citation rate",
            "answer",
            True,
            description="share of distinct cited papers that are gold papers",
        ),
        MetricSpec(
            "supported_claim_rate",
            "Supported-claim rate",
            "answer",
            True,
            description="share of answer claims (cited + uncited factual sentences) the claim "
            "judge found supported by the cited chunks; partial counts as not supported",
        ),
        MetricSpec(
            "correct_refusal",
            "Correct refusal",
            "abstention",
            True,
            description="out-of-scope questions answered with a correct refusal / correction",
        ),
        MetricSpec(
            "total_tokens",
            "Tokens / question",
            "cost",
            False,
            unit="tokens",
            description="input + output tokens of the turn (judge excluded)",
        ),
        MetricSpec(
            "latency_s",
            "Latency / question",
            "cost",
            False,
            unit="s",
            description="wall-clock time of the turn",
        ),
    ]


def compute_metrics(
    record: EvalRecord,
    question: GoldQuestion,
    k: int,
    claim_verifier: ClaimVerifier | None = None,
) -> dict[str, float | None]:
    """All metrics of one record. Retrieval and answer metrics only for in-scope questions,
    correct refusal only for out-of-scope ones; cost for all. The supported-claim rate comes
    from ``claim_verifier`` (default: the claim check stored on the record)."""
    values: dict[str, float | None] = {s.name: None for s in metric_specs(k)}
    judge = record.judge if record.judge and record.judge.ok else None
    if question.in_scope:
        papers = record.retrieved_arxiv_ids()
        values[f"recall@{k}"] = recall_at_k(papers, question.gold_arxiv_ids, k)
        values["mrr"] = reciprocal_rank(papers, question.gold_arxiv_ids)
        if judge is not None:
            values["key_fact_coverage"] = key_fact_coverage([f.covered for f in judge.facts])
        values["gold_citation_rate"] = gold_citation_rate(
            record.cited_arxiv_ids, question.gold_arxiv_ids
        )
        verifier = claim_verifier or STORED_CLAIM_RATE
        values["supported_claim_rate"] = verifier.supported_claim_rate(record, question)
    elif judge is not None and judge.refusal is not None:
        values["correct_refusal"] = correct_refusal(judge.refusal.correct)
    values["total_tokens"] = float(record.token_usage.total_tokens)
    values["latency_s"] = record.latency_ms / 1000.0
    return values


__all__ = [
    "STORED_CLAIM_RATE",
    "ClaimVerifier",
    "MetricSpec",
    "StoredClaimRate",
    "compute_metrics",
    "correct_refusal",
    "gold_citation_rate",
    "key_fact_coverage",
    "metric_specs",
    "recall_at_k",
    "reciprocal_rank",
]
