"""Domain tools for the agent (LangChain StructuredTools returning ToolResult)."""

from research_assistant.tools.arxiv_ids import ArxivId, ArxivIdError, parse_arxiv_id
from research_assistant.tools.arxiv_tools import (
    arxiv_ingest_request,
    arxiv_search,
    bibtex_export,
    confirm_ingest,
)
from research_assistant.tools.base import (
    IndexKnowledgeBase,
    KnowledgeBase,
    StaticKnowledgeBase,
    ToolDeps,
    default_deps,
)
from research_assistant.tools.citations import citation_graph
from research_assistant.tools.dp import dp_accountant
from research_assistant.tools.registry import build_tools

__all__ = [
    "ArxivId",
    "ArxivIdError",
    "IndexKnowledgeBase",
    "KnowledgeBase",
    "StaticKnowledgeBase",
    "ToolDeps",
    "arxiv_ingest_request",
    "arxiv_search",
    "bibtex_export",
    "build_tools",
    "citation_graph",
    "confirm_ingest",
    "default_deps",
    "dp_accountant",
    "parse_arxiv_id",
]
