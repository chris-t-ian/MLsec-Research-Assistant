"""Cross-encoder reranking (default model: BAAI/bge-reranker-base, from settings)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol

from research_assistant.config import RetrievalConfig
from research_assistant.models import ScoredChunk

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    """The reranker model could not be loaded or failed to score."""


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance score per text (higher = more relevant). Raise RerankError on failure."""
        ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder, loaded on first use."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        max_length: int = 256,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device
        self._model: Any = None

    @classmethod
    def from_config(cls, cfg: RetrievalConfig) -> CrossEncoderReranker:
        return cls(cfg.reranker_model, cfg.reranker_max_length, cfg.reranker_batch_size)

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # heavy import: torch

                logger.info("Loading reranker model %s", self.model_name)
                self._model = CrossEncoder(
                    self.model_name, max_length=self._max_length, device=self._device
                )
            except Exception as exc:  # download / load failures
                raise RerankError(f"Cannot load reranker {self.model_name}") from exc
        return self._model

    def warm_up(self) -> None:
        self._load()

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        model = self._load()
        try:
            scores = model.predict(
                [(query, t) for t in texts],
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RerankError("Reranker failed to score candidates") from exc
        return [float(s) for s in scores]


def _rerank_text(hit: ScoredChunk) -> str:
    c = hit.chunk
    return f"{c.section}\n{c.text}" if c.section else c.text


def rerank(query: str, hits: Sequence[ScoredChunk], reranker: Reranker) -> list[ScoredChunk]:
    """Reorder ``hits`` by cross-encoder score; the score is added as ``stage_scores['rerank']``."""
    scores = reranker.score(query, [_rerank_text(h) for h in hits])
    if len(scores) != len(hits):
        raise RerankError("Reranker returned the wrong number of scores")
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    return [
        hits[i].model_copy(
            update={
                "score": scores[i],
                "rank": rank,
                "source": "rerank",
                "stage_scores": {**hits[i].stage_scores, "rerank": scores[i]},
            }
        )
        for rank, i in enumerate(order, start=1)
    ]


__all__ = ["CrossEncoderReranker", "RerankError", "Reranker", "rerank"]
