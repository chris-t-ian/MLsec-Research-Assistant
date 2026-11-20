"""Dense (Chroma) and sparse (BM25) indexes over the same chunks.

Layout under ``index_dir``:

    chroma/    persistent Chroma collection; ids are chunk ids, cosine space
    bm25.pkl   tokenized chunks + chunk data (plain builtins only, loaded with a
               restricted unpickler); the BM25 model is rebuilt from it on load

Both indexes are written per paper: :meth:`ChunkIndex.replace_paper` deletes all chunks
of a paper and upserts the new ones, so re-indexing is idempotent and never leaves stale
chunks behind. Quarantined chunks are indexed but excluded from search by default.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import shutil
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from rank_bm25 import BM25Okapi

from research_assistant.config import RetrievalConfig
from research_assistant.ingest.chunking import QUARANTINED
from research_assistant.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)

CHROMA_DIR = "chroma"
BM25_FILE = "bm25.pkl"
AUTHOR_SEP = "; "
FLAG_SEP = ","
# Bump when tokenize() changes: an index built with another version must be rebuilt.
TOKENIZER_VERSION = 1
_BM25_FORMAT = 1


class ChunkIndexError(RuntimeError):
    """The index is missing, corrupt or was built with a different embedding model/tokenizer."""


# --------------------------------------------------------------------------- embeddings


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Documents and queries may be embedded differently."""

    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers model with L2-normalized embeddings.

    ``query_instruction`` is prepended to queries only (BGE-style asymmetric retrieval).
    The model is loaded on first use, so constructing the embedder is cheap.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_instruction: str = "",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._query_instruction = query_instruction
        self._batch_size = batch_size
        self._device = device
        self._model: Any = None

    @classmethod
    def from_config(cls, cfg: RetrievalConfig) -> SentenceTransformerEmbedder:
        return cls(cfg.embedding_model, cfg.query_instruction, cfg.embedding_batch_size)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import: torch

            logger.info("Loading embedding model %s", self._model_name)
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._load().encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32).tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_instruction + text])[0]


# --------------------------------------------------------------------------- tokenizer

# Greek letters as written in converted LaTeX/PDF text, mapped to the words people query.
_GREEK_NAMES = (
    "alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta", "lambda", "mu", "pi",
    "rho", "sigma", "tau",
)  # fmt: skip
_GREEK = {
    unicodedata.lookup(f"GREEK SMALL LETTER {n.upper().replace('LAMBDA', 'LAMDA')}"): n
    for n in _GREEK_NAMES
}
_GREEK[unicodedata.lookup("GREEK LUNATE EPSILON SYMBOL")] = "epsilon"
_GREEK_TABLE = str.maketrans({k: f" {v} " for k, v in _GREEK.items()})
# Runs of letters/digits (no underscore), optionally joined by single hyphens: "dp-sgd".
_TOKEN_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens for BM25.

    Hyphenated terms are kept whole ("dp-sgd") and additionally split into their parts
    ("dp", "sgd"), so "membership-inference" also matches "membership inference".
    Greek letters become their names ("ε" -> "epsilon").
    """
    text = unicodedata.normalize("NFKC", text).lower().translate(_GREEK_TABLE)
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group()
        tokens.append(tok)
        if "-" in tok:
            tokens.extend(tok.split("-"))
    return tokens


# --------------------------------------------------------------------------- BM25 store


class _BuiltinsOnlyUnpickler(pickle.Unpickler):
    """Refuses every global, so a tampered bm25.pkl cannot execute code on load."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"forbidden global in BM25 index: {module}.{name}")


def _is_quarantined(chunk: Chunk) -> bool:
    return QUARANTINED in chunk.flags


class BM25Store:
    """Tokenized chunks persisted to a pickle of plain builtins; BM25 built lazily."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # chunk id -> {"chunk": Chunk dict, "tokens": [...], "quarantined": bool, "topic": str}
        self._docs: dict[str, dict[str, Any]] = {}
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self.dirty = False
        if path.exists():
            self._load()

    def _load(self) -> None:
        try:
            with self.path.open("rb") as fh:
                data = _BuiltinsOnlyUnpickler(fh).load()
        except (OSError, pickle.UnpicklingError, EOFError, ValueError) as exc:
            raise ChunkIndexError(f"Cannot read BM25 index {self.path.name}: {exc}") from exc
        if not isinstance(data, dict) or data.get("format") != _BM25_FORMAT:
            raise ChunkIndexError("Unknown BM25 index format; rebuild the index")
        if data.get("tokenizer") != TOKENIZER_VERSION:
            raise ChunkIndexError("BM25 index was built with another tokenizer; rebuild it")
        self._docs = data["docs"]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        payload = {"format": _BM25_FORMAT, "tokenizer": TOKENIZER_VERSION, "docs": self._docs}
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)
        self.dirty = False

    def __len__(self) -> int:
        return len(self._docs)

    def paper_ids(self) -> set[str]:
        return {d["chunk"]["arxiv_id"] for d in self._docs.values()}

    def upsert(self, chunks: Iterable[Chunk], topic: str) -> None:
        for c in chunks:
            self._docs[c.id] = {
                "chunk": c.model_dump(mode="json"),
                "tokens": tokenize(f"{c.section or ''}\n{c.text}"),
                "quarantined": _is_quarantined(c),
                "topic": topic,
            }
        self._invalidate()

    def delete_paper(self, arxiv_id: str) -> int:
        stale = [cid for cid, d in self._docs.items() if d["chunk"]["arxiv_id"] == arxiv_id]
        for cid in stale:
            del self._docs[cid]
        if stale:
            self._invalidate()
        return len(stale)

    def _invalidate(self) -> None:
        self._bm25 = None
        self.dirty = True

    def _model(self) -> BM25Okapi | None:
        if self._bm25 is None and self._docs:
            self._ids = list(self._docs)
            self._bm25 = BM25Okapi([self._docs[i]["tokens"] for i in self._ids])
        return self._bm25

    def search(
        self, query: str, k: int, include_quarantined: bool = False
    ) -> list[RetrievalResult]:
        model = self._model()
        q = tokenize(query)
        if model is None or not q or k <= 0:
            return []
        scores = np.asarray(model.get_scores(q), dtype=float)
        if not include_quarantined:
            mask = np.array([self._docs[i]["quarantined"] for i in self._ids], dtype=bool)
            scores[mask] = -np.inf
        order = np.argsort(-scores, kind="stable")[:k]
        results: list[RetrievalResult] = []
        for i in order:
            if not np.isfinite(scores[i]) or scores[i] <= 0:
                break  # no query term in this (or any later) chunk
            chunk = Chunk.model_validate(self._docs[self._ids[i]]["chunk"])
            results.append(
                RetrievalResult(
                    chunk=chunk, score=float(scores[i]), rank=len(results) + 1, source="bm25"
                )
            )
        return results


# --------------------------------------------------------------------------- Chroma


def chunk_metadata(chunk: Chunk, topic: str) -> dict[str, str | int | bool]:
    """Chroma metadata for a chunk (Chroma rejects None values, so year may be absent)."""
    meta: dict[str, str | int | bool] = {
        "arxiv_id": chunk.arxiv_id,
        "title": chunk.title,
        "authors": AUTHOR_SEP.join(chunk.authors),
        "section": chunk.section or "",
        "topic": topic,
        "quarantined": _is_quarantined(chunk),
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "flags": FLAG_SEP.join(chunk.flags),
    }
    if chunk.year is not None:
        meta["year"] = chunk.year
    return meta


def chunk_from_record(chunk_id: str, text: str, meta: dict[str, Any]) -> Chunk:
    authors = str(meta.get("authors") or "")
    flags = str(meta.get("flags") or "")
    year = meta.get("year")
    return Chunk(
        id=chunk_id,
        arxiv_id=str(meta["arxiv_id"]),
        title=str(meta.get("title") or ""),
        authors=authors.split(AUTHOR_SEP) if authors else [],
        year=int(year) if year is not None else None,
        section=str(meta.get("section") or "") or None,
        text=text,
        char_start=int(meta.get("char_start") or 0),
        char_end=int(meta.get("char_end") or 0),
        flags=flags.split(FLAG_SEP) if flags else [],
    )


class ChunkIndex:
    """Chroma (dense) + BM25 (sparse) over the same chunks, stored in ``index_dir``.

    With ``autosave`` (default) the BM25 store is written after every change; bulk
    indexing can disable it and call :meth:`save` once at the end.
    """

    def __init__(
        self,
        index_dir: Path,
        embedder: Embedder,
        collection_name: str = "chunks",
        autosave: bool = True,
    ) -> None:
        import chromadb  # heavy import, kept out of module import time
        from chromadb.config import Settings as ChromaSettings

        self.index_dir = index_dir
        self.embedder = embedder
        self.autosave = autosave
        index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(index_dir / CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # we always pass embeddings explicitly
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": embedder.model_name},
        )
        stored = (self._collection.metadata or {}).get("embedding_model")
        if stored != embedder.model_name:
            raise ChunkIndexError(
                f"Index was built with embedding model {stored!r}, not "
                f"{embedder.model_name!r}; rebuild it with scripts/ingest.py --rebuild"
            )
        self._bm25 = BM25Store(index_dir / BM25_FILE)

    @classmethod
    def from_config(
        cls, index_dir: Path, cfg: RetrievalConfig, autosave: bool = True
    ) -> ChunkIndex:
        return cls(
            index_dir, SentenceTransformerEmbedder.from_config(cfg), cfg.collection_name, autosave
        )

    @staticmethod
    def wipe(index_dir: Path) -> None:
        """Delete the Chroma directory and the BM25 file under ``index_dir``."""
        shutil.rmtree(index_dir / CHROMA_DIR, ignore_errors=True)
        (index_dir / BM25_FILE).unlink(missing_ok=True)

    # ------------------------------------------------------------------ writes

    def upsert(self, chunks: Sequence[Chunk], topic: str = "other") -> int:
        """Insert or overwrite chunks by id in both indexes. Returns the number written."""
        if not chunks:
            return 0
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        batch = self._client.get_max_batch_size()
        for lo in range(0, len(chunks), batch):
            part = chunks[lo : lo + batch]
            self._collection.upsert(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                embeddings=embeddings[lo : lo + batch],  # type: ignore[arg-type]
                metadatas=[chunk_metadata(c, topic) for c in part],  # type: ignore[misc]
            )
        self._bm25.upsert(chunks, topic)
        self._maybe_save()
        return len(chunks)

    def delete_paper(self, arxiv_id: str) -> None:
        self._collection.delete(where={"arxiv_id": arxiv_id})
        self._bm25.delete_paper(arxiv_id)
        self._maybe_save()

    def replace_paper(self, arxiv_id: str, chunks: Sequence[Chunk], topic: str = "other") -> int:
        """Drop every chunk of ``arxiv_id`` and index ``chunks`` (idempotent)."""
        if any(c.arxiv_id != arxiv_id for c in chunks):
            raise ValueError(f"all chunks must belong to {arxiv_id}")
        autosave, self.autosave = self.autosave, False
        try:
            self.delete_paper(arxiv_id)
            n = self.upsert(chunks, topic)
        finally:
            self.autosave = autosave
        self._maybe_save()
        return n

    def save(self) -> None:
        if self._bm25.dirty:
            self._bm25.save()

    def _maybe_save(self) -> None:
        if self.autosave:
            self.save()

    # ------------------------------------------------------------------ reads

    def count(self) -> int:
        return self._collection.count()

    def paper_ids(self) -> set[str]:
        """Papers present in the BM25 store (written last, so they are complete)."""
        return self._bm25.paper_ids()

    def get(self, chunk_id: str) -> tuple[Chunk, dict[str, Any]] | None:
        """A stored chunk and its raw Chroma metadata, or None."""
        res = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        meta = dict(res["metadatas"][0])  # type: ignore[index]
        return chunk_from_record(chunk_id, res["documents"][0], meta), meta  # type: ignore[index]

    def dense_search(
        self, query: str, k: int, include_quarantined: bool = False
    ) -> list[RetrievalResult]:
        n = min(k, self.count())
        if n <= 0 or not query.strip():
            return []
        res = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],  # type: ignore[arg-type]
            n_results=n,
            where=None if include_quarantined else {"quarantined": False},
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]  # type: ignore[index]
        metas = res["metadatas"][0]  # type: ignore[index]
        dists = res["distances"][0]  # type: ignore[index]
        return [
            RetrievalResult(
                chunk=chunk_from_record(cid, doc, dict(meta)),
                score=1.0 - float(dist),  # cosine similarity
                rank=rank,
                source="dense",
            )
            for rank, (cid, doc, meta, dist) in enumerate(
                zip(ids, docs, metas, dists, strict=True), start=1
            )
        ]

    def bm25_search(
        self, query: str, k: int, include_quarantined: bool = False
    ) -> list[RetrievalResult]:
        return self._bm25.search(query, k, include_quarantined)


__all__ = [
    "BM25Store",
    "ChunkIndex",
    "ChunkIndexError",
    "Embedder",
    "SentenceTransformerEmbedder",
    "chunk_from_record",
    "chunk_metadata",
    "tokenize",
]
