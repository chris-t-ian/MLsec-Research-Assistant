"""citation_graph tool and Semantic Scholar client. All HTTP is mocked."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from tenacity import wait_none

from research_assistant.config import SemanticScholarConfig, get_settings
from research_assistant.ingest.arxiv_fetch import RateLimiter
from research_assistant.tools import StaticKnowledgeBase, ToolDeps, citation_graph
from research_assistant.tools.semantic_scholar import (
    PAPER_FIELDS,
    JsonDiskCache,
    SemanticScholarClient,
)

Handler = Callable[[httpx.Request], httpx.Response]

CITING = {
    "data": [
        {
            "citingPaper": {
                "paperId": "abc",
                "title": "Extracting Training Data from Large Language Models",
                "year": 2020,
                "authors": [{"authorId": "1", "name": f"Author {i}"} for i in range(7)],
                "externalIds": {"ArXiv": "2012.07805", "DOI": "10.1/xyz"},
                "citationCount": 1500,
            }
        },
        {
            "citingPaper": {
                "paperId": "def",
                "title": "A paper without arXiv ID",
                "year": None,
                "authors": [],
                "externalIds": {"DOI": "10.2/abc"},
                "citationCount": 3,
            }
        },
        {"citingPaper": {"paperId": None, "title": None}},  # elided entry
    ]
}


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def make(
    handler: Handler,
    tmp_path: Path,
    kb: set[str] | None = None,
    api_key: str | None = None,
    clock: Clock | None = None,
) -> tuple[ToolDeps, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = SemanticScholarClient(
        SemanticScholarConfig(max_retries=2),
        api_key=SecretStr(api_key) if api_key else None,
        cache=JsonDiskCache(tmp_path / "s2", ttl_s=24 * 3600, clock=clock or Clock()),
        http=httpx.Client(transport=httpx.MockTransport(record)),
        limiter=RateLimiter(0),
        retry_wait=wait_none(),
    )
    deps = ToolDeps(
        settings=get_settings(), kb=StaticKnowledgeBase(kb or set()), semantic_scholar=client
    )
    return deps, seen


def ok_json(body: Any) -> Handler:
    return lambda request: httpx.Response(200, json=body)


def test_citations_are_parsed_with_kb_flag(tmp_path: Path) -> None:
    deps, seen = make(ok_json(CITING), tmp_path, kb={"2012.07805"})

    result = citation_graph("1610.05820v3", "citations", limit=5, deps=deps)

    assert result.ok and result.source == "semantic_scholar"
    req = seen[0]
    assert req.url.path == "/graph/v1/paper/arXiv:1610.05820/citations"
    assert (
        req.url.params["fields"] == PAPER_FIELDS == "title,year,authors,externalIds,citationCount"
    )
    assert req.url.params["limit"] == "5"
    assert "x-api-key" not in req.headers
    assert result.data["arxiv_id"] == "1610.05820"
    first, second = result.data["papers"]
    assert first == {
        "title": "Extracting Training Data from Large Language Models",
        "year": 2020,
        "authors": ["Author 0", "Author 1", "Author 2", "Author 3", "Author 4", "et al."],
        "arxiv_id": "2012.07805",
        "doi": "10.1/xyz",
        "citation_count": 1500,
        "in_kb": True,
    }
    assert second["arxiv_id"] is None and second["in_kb"] is False and second["year"] is None


def test_references_use_cited_paper_and_api_key(tmp_path: Path) -> None:
    body = {"data": [{"citedPaper": {"title": "DP-SGD", "externalIds": {"ArXiv": "1607.00133"}}}]}
    deps, seen = make(ok_json(body), tmp_path, api_key="k-123")
    result = citation_graph("1610.05820", "references", limit=20, deps=deps)
    assert seen[0].url.path.endswith("/references")
    assert seen[0].headers["x-api-key"] == "k-123"
    assert [p["arxiv_id"] for p in result.data["papers"]] == ["1607.00133"]


def test_old_style_id_is_path_encoded(tmp_path: Path) -> None:
    deps, seen = make(ok_json({"data": []}), tmp_path)
    result = citation_graph("cs/0112017", deps=deps)
    assert result.ok and result.data["papers"] == []
    assert seen[0].url.raw_path.startswith(b"/graph/v1/paper/arXiv:cs%2F0112017/citations")


def test_backoff_on_429_then_success(tmp_path: Path) -> None:
    responses = iter([httpx.Response(429), httpx.Response(429), httpx.Response(200, json=CITING)])
    deps, seen = make(lambda r: next(responses), tmp_path)
    result = citation_graph("1610.05820", deps=deps)
    assert result.ok and len(seen) == 3


def test_persistent_429_gives_friendly_error(tmp_path: Path) -> None:
    deps, seen = make(lambda r: httpx.Response(429), tmp_path, api_key="secret-key")
    result = citation_graph("1610.05820", deps=deps)
    assert not result.ok and len(seen) == 3  # 1 + max_retries
    assert "rate-limiting" in (result.error or "")
    assert "secret-key" not in (result.error or "")


def test_unknown_paper(tmp_path: Path) -> None:
    deps, seen = make(lambda r: httpx.Response(404, json={"error": "not found"}), tmp_path)
    result = citation_graph("2112.03570", deps=deps)
    assert not result.ok and len(seen) == 1  # 404 is not retried
    assert result.error == "Semantic Scholar has no record of this paper."


def test_timeout_is_retried_and_reported(tmp_path: Path) -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    deps, seen = make(timeout, tmp_path)
    result = citation_graph("1610.05820", deps=deps)
    assert not result.ok and len(seen) == 3
    assert "not reachable" in (result.error or "")


def test_cache_hit_within_ttl_and_refetch_after(tmp_path: Path) -> None:
    clock = Clock()
    deps, seen = make(ok_json(CITING), tmp_path, kb=set(), clock=clock)

    first = citation_graph("1610.05820", limit=5, deps=deps)
    clock.now += 23 * 3600
    second = citation_graph("1610.05820v2", limit=5, deps=deps)  # same paper, cached
    assert len(seen) == 1 and first.data == second.data

    citation_graph("1610.05820", "references", limit=5, deps=deps)  # different key
    citation_graph("1610.05820", limit=6, deps=deps)
    assert len(seen) == 3

    clock.now += 2 * 3600  # first entry now older than 24 h
    citation_graph("1610.05820", limit=5, deps=deps)
    assert len(seen) == 4


def test_kb_flag_is_computed_fresh_on_cache_hit(tmp_path: Path) -> None:
    kb = StaticKnowledgeBase()
    deps, seen = make(ok_json(CITING), tmp_path)
    deps.kb = kb
    assert citation_graph("1610.05820", deps=deps).data["papers"][0]["in_kb"] is False
    kb.add("2012.07805")
    assert citation_graph("1610.05820", deps=deps).data["papers"][0]["in_kb"] is True
    assert len(seen) == 1


def test_corrupt_cache_file_is_ignored(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path, ttl_s=60)
    cache.set("k", [1, 2])
    assert cache.get("k") == [1, 2]
    for path in tmp_path.glob("*.json"):
        path.write_text("{not json")
    assert cache.get("k") is None


@pytest.mark.parametrize(
    "args",
    [
        {"arxiv_id": "1610.05820", "direction": "cited_by"},
        {"arxiv_id": "1610.05820", "limit": 21},
        {"arxiv_id": "1610.05820", "limit": 0},
        {"arxiv_id": "../../etc/passwd"},
    ],
)
def test_invalid_arguments_make_no_request(tmp_path: Path, args: dict[str, Any]) -> None:
    deps, seen = make(ok_json(CITING), tmp_path)
    result = citation_graph(**args, deps=deps)
    assert not result.ok and result.source == "validation"
    assert seen == []
