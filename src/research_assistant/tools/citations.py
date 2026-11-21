"""citation_graph tool: who cites a paper / what it cites, via Semantic Scholar."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from research_assistant.models import ToolResult
from research_assistant.tools.arxiv_ids import ArxivIdStr, parse_arxiv_id
from research_assistant.tools.base import (
    ToolArgs,
    ToolDeps,
    base_id,
    clean_text,
    default_deps,
    run_safely,
)
from research_assistant.tools.semantic_scholar import Direction

CITATION_GRAPH_MAX_LIMIT = 20
MAX_AUTHORS_SHOWN = 5


class CitationGraphArgs(ToolArgs):
    arxiv_id: ArxivIdStr = Field(description="arXiv ID of the paper, e.g. '1610.05820'.")
    direction: Direction = Field(
        default="citations",
        description=(
            "'citations' = later papers that cite this paper; "
            "'references' = papers this paper cites."
        ),
    )
    limit: int = Field(
        default=10, ge=1, le=CITATION_GRAPH_MAX_LIMIT, description="Number of papers (1-20)."
    )


def _paper(raw: dict[str, Any], kb_ids: set[str]) -> dict[str, Any]:
    ext = raw.get("externalIds") if isinstance(raw.get("externalIds"), dict) else {}
    arxiv = ext.get("ArXiv") if isinstance(ext.get("ArXiv"), str) else None
    authors_raw = raw.get("authors") if isinstance(raw.get("authors"), list) else []
    names = [clean_text(a.get("name")) for a in authors_raw if isinstance(a, dict)]
    names = [n for n in names if n]
    shown = names[:MAX_AUTHORS_SHOWN] + (["et al."] if len(names) > MAX_AUTHORS_SHOWN else [])
    year = raw.get("year") if isinstance(raw.get("year"), int) else None
    count = raw.get("citationCount") if isinstance(raw.get("citationCount"), int) else None
    return {
        "title": clean_text(raw.get("title")),
        "year": year,
        "authors": shown,
        "arxiv_id": arxiv,
        "doi": ext.get("DOI") if isinstance(ext.get("DOI"), str) else None,
        "citation_count": count,
        "in_kb": arxiv is not None and base_id(arxiv) in kb_ids,
    }


def citation_graph(
    arxiv_id: str,
    direction: str = "citations",
    limit: int = 10,
    deps: ToolDeps | None = None,
) -> ToolResult:
    """Citing or cited papers of an arXiv paper (Semantic Scholar, cached 24 h).

    ``data``: ``{"arxiv_id", "direction", "papers": [{title, year, authors, arxiv_id, doi,
    citation_count, in_kb}]}``.
    """
    deps = deps or default_deps()

    def run() -> dict[str, Any]:
        args = CitationGraphArgs.model_validate(
            {"arxiv_id": arxiv_id, "direction": direction, "limit": limit}
        )
        paper_id = parse_arxiv_id(args.arxiv_id).base
        raw = deps.get_semantic_scholar().paper_links(paper_id, args.direction, args.limit)
        kb_ids = deps.get_kb().paper_ids()
        return {
            "arxiv_id": paper_id,
            "direction": args.direction,
            "papers": [_paper(p, kb_ids) for p in raw[: args.limit]],
        }

    return run_safely("semantic_scholar", run)
