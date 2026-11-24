"""Semantic Scholar Graph API wrapper: timeout, rate limit, retries with backoff, typed errors,
and a JSON disk cache with a time-to-live.

Only the paper citation/reference endpoints are used:

    GET {api_url}/paper/arXiv:<id>/citations?fields=...&limit=N
    GET {api_url}/paper/arXiv:<id>/references?fields=...&limit=N

The API key (``SEMANTIC_SCHOLAR_API_KEY``, optional) is sent as ``x-api-key`` and never logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import httpx
from pydantic import SecretStr
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from tenacity.wait import WaitBaseT

from research_assistant.config import SemanticScholarConfig, Settings
from research_assistant.ingest.arxiv_fetch import RateLimiter

logger = logging.getLogger(__name__)

Direction = Literal["citations", "references"]
PAPER_FIELDS = "title,year,authors,externalIds,citationCount"
_CACHE_FORMAT = 1


class SemanticScholarError(RuntimeError):
    """Base class for Semantic Scholar errors. Messages never contain the API key."""


class SemanticScholarUnavailableError(SemanticScholarError):
    """Timeout, transport failure or HTTP 5xx (after retries)."""


class SemanticScholarRateLimitError(SemanticScholarUnavailableError):
    """HTTP 429 (after retries with exponential backoff)."""


class SemanticScholarNotFoundError(SemanticScholarError):
    """HTTP 404: the paper is unknown to Semantic Scholar."""


class SemanticScholarResponseError(SemanticScholarError):
    """Other HTTP errors or an unparseable body."""


class JsonDiskCache:
    """One JSON file per key under ``directory``; entries expire after ``ttl_s`` seconds."""

    def __init__(
        self, directory: Path, ttl_s: float, clock: Callable[[], float] = time.time
    ) -> None:
        self.directory = directory
        self.ttl_s = ttl_s
        self._clock = clock

    def _path(self, key: str) -> Path:
        return self.directory / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (
            not isinstance(entry, dict)
            or entry.get("format") != _CACHE_FORMAT
            or entry.get("key") != key
            or not isinstance(entry.get("stored_at"), int | float)
        ):
            return None
        if self._clock() - entry["stored_at"] > self.ttl_s:
            return None
        return entry.get("value")

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        entry = {"format": _CACHE_FORMAT, "key": key, "stored_at": self._clock(), "value": value}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(entry), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:  # a broken cache must not break the tool
            logger.warning("Could not write cache entry: %s", exc)


class SemanticScholarClient:
    """Wrapper for every Semantic Scholar request."""

    def __init__(
        self,
        config: SemanticScholarConfig,
        api_key: SecretStr | None = None,
        cache: JsonDiskCache | None = None,
        http: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
        retry_wait: WaitBaseT | None = None,
    ) -> None:
        self._cfg = config
        headers = {"User-Agent": config.user_agent}
        key = api_key.get_secret_value() if api_key is not None else ""
        if key:
            headers["x-api-key"] = key
        self.has_api_key = bool(key)
        self._http = http or httpx.Client(timeout=config.timeout_s, follow_redirects=True)
        self._headers = headers
        self._cache = cache
        self._limiter = limiter or RateLimiter(config.min_interval_s)
        self._retrying = Retrying(
            stop=stop_after_attempt(config.max_retries + 1),
            wait=retry_wait or wait_exponential(multiplier=config.backoff_s, max=60),
            retry=retry_if_exception_type(SemanticScholarUnavailableError),
            reraise=True,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> SemanticScholarClient:
        cfg = settings.semantic_scholar
        cache_dir = settings.paths.resolve().cache_dir / "semantic_scholar"
        return cls(
            cfg,
            api_key=settings.semantic_scholar_api_key,
            cache=JsonDiskCache(cache_dir, cfg.cache_ttl_h * 3600),
        )

    def close(self) -> None:
        self._http.close()

    def _get_once(self, path: str, params: dict[str, str]) -> Any:
        self._limiter.wait()
        url = f"{self._cfg.api_url}{path}"
        try:
            resp = self._http.get(url, params=params, headers=self._headers)
        except httpx.TimeoutException as exc:
            raise SemanticScholarUnavailableError(f"Timeout for {path}") from exc
        except httpx.TransportError as exc:
            raise SemanticScholarUnavailableError(f"{type(exc).__name__} for {path}") from exc
        status = resp.status_code
        if status == 429:
            raise SemanticScholarRateLimitError(f"HTTP 429 for {path}")
        if status >= 500:
            raise SemanticScholarUnavailableError(f"HTTP {status} for {path}")
        if status == 404:
            raise SemanticScholarNotFoundError(f"HTTP 404 for {path}")
        if status != 200:
            raise SemanticScholarResponseError(f"HTTP {status} for {path}")
        try:
            return resp.json()
        except ValueError as exc:
            raise SemanticScholarResponseError(f"Invalid JSON for {path}") from exc

    def _get(self, path: str, params: dict[str, str]) -> Any:
        return self._retrying(self._get_once, path, params)

    def paper_links(self, arxiv_id: str, direction: Direction, limit: int) -> list[dict[str, Any]]:
        """Citing (``citations``) or cited (``references``) papers of ``arXiv:<arxiv_id>``.

        Returns the raw paper objects (``citingPaper`` / ``citedPaper``); entries without
        data (e.g. elided by the publisher) are dropped. Results are cached on disk.
        """
        cache_key = f"{direction}:{arxiv_id}:{limit}:{PAPER_FIELDS}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if isinstance(cached, list):
                logger.debug("Semantic Scholar cache hit for %s", cache_key)
                return cached

        path = f"/paper/arXiv:{quote(arxiv_id, safe='')}/{direction}"
        body = self._get(path, {"fields": PAPER_FIELDS, "limit": str(limit), "offset": "0"})
        if not isinstance(body, dict):
            raise SemanticScholarResponseError(f"Unexpected body for {path}")
        items = body.get("data") or []
        if not isinstance(items, list):
            raise SemanticScholarResponseError(f"Unexpected 'data' for {path}")
        key = "citingPaper" if direction == "citations" else "citedPaper"
        papers = [
            item[key]
            for item in items
            if isinstance(item, dict) and isinstance(item.get(key), dict) and item[key].get("title")
        ]
        if self._cache is not None:
            self._cache.set(cache_key, papers)
        return papers
