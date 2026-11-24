"""Download arXiv metadata, LaTeX source (e-print) and PDF into data/raw/<arxiv_id>/.

Layout per paper (written atomically via a temporary ``.<id>.partial`` directory):

    data/raw/<arxiv_id>/meta.json   Paper model
    data/raw/<arxiv_id>/paper.pdf
    data/raw/<arxiv_id>/source/     extracted LaTeX (only if latex_available)

All arXiv traffic goes through :class:`ArxivClient`, which enforces at most one request
every ``min_interval_s`` (>= 3 s), a timeout, retries with exponential backoff and typed errors.
Downloaded archives are untrusted: tarballs are extracted with the ``data`` filter and size caps.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import tarfile
import threading
import time
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

import httpx
import yaml
from pydantic import BaseModel, Field, ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from tenacity.wait import WaitBaseT

from research_assistant.config import ArxivConfig
from research_assistant.events import EventSink, NullSink
from research_assistant.models import Paper, PaperListEntry

logger = logging.getLogger(__name__)

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

# New style (2007+): 0704.0001, 2401.12345, optional version. Old style: hep-th/9901001.
_ARXIV_ID_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")
_ENTRY_ID_RE = re.compile(r"arxiv\.org/abs/(?P<base>.+?)v(?P<version>\d+)$")
_TEX_MARKERS = re.compile(rb"\\(?:documentclass|documentstyle|begin\{document\}|input|section)\b")
_MAX_TAR_MEMBERS = 10_000

META_FILE = "meta.json"
PDF_FILE = "paper.pdf"
SOURCE_DIR = "source"


# --------------------------------------------------------------------------- errors


class ArxivError(RuntimeError):
    """Base class for all arXiv download errors."""


class ArxivNetworkError(ArxivError):
    """Transport failure, timeout, HTTP 429 or 5xx (retried before being raised)."""


class ArxivNotFoundError(ArxivError):
    """The ID does not exist on arXiv or the resource is unavailable (HTTP 404)."""


class ArxivHTTPError(ArxivError):
    """Non-retryable HTTP error (4xx other than 404/429)."""


class ArxivResponseError(ArxivError):
    """The response could not be parsed or is too large."""


class EprintFormatError(ArxivError):
    """The e-print payload is not a recognised or safe LaTeX/PDF format."""


# --------------------------------------------------------------------------- models


class ArxivMetadata(BaseModel):
    arxiv_id: str
    version: int = Field(ge=1)
    title: str
    authors: list[str]
    year: int | None
    abstract: str | None
    categories: list[str]
    url: str

    @property
    def versioned_id(self) -> str:
        return f"{self.arxiv_id}v{self.version}"


class FailedPaper(BaseModel):
    arxiv_id: str
    error: str


class CorpusReport(BaseModel):
    downloaded: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    failed: list[FailedPaper] = Field(default_factory=list)
    latex_available: list[str] = Field(default_factory=list)


EprintKind = Literal["tarball", "single_tex", "pdf"]


@dataclass(frozen=True)
class EprintResult:
    kind: EprintKind
    pdf_bytes: bytes | None = None  # set when the e-print itself is the PDF


# --------------------------------------------------------------------------- rate limiting


class RateLimiter:
    """Blocks so that consecutive calls to :meth:`wait` are at least ``min_interval_s`` apart.

    Thread-safe: concurrent callers are serialized, so the interval holds process-wide for
    every request made through the same limiter.
    """

    def __init__(
        self,
        min_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last is not None:
                delay = self._last + self._interval - now
                if delay > 0:
                    self._sleep(delay)
                    now = self._clock()
            self._last = now


# --------------------------------------------------------------------------- client


class ArxivClient:
    """Wrapper for every arXiv request: rate limit, timeout, retries, typed errors."""

    def __init__(
        self,
        config: ArxivConfig,
        http: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
        retry_wait: WaitBaseT | None = None,
    ) -> None:
        self._cfg = config
        self._http = http or httpx.Client(
            timeout=config.timeout_s,
            follow_redirects=True,
            headers={"User-Agent": config.user_agent},
        )
        self._limiter = limiter or RateLimiter(config.min_interval_s)
        self._retrying = Retrying(
            stop=stop_after_attempt(config.max_retries + 1),
            wait=retry_wait or wait_exponential(multiplier=config.backoff_s, max=60),
            retry=retry_if_exception_type(ArxivNetworkError),
            reraise=True,
        )
        self._max_bytes = config.max_download_mb * 1024 * 1024

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ArxivClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get_once(self, url: str, params: dict[str, str] | None) -> bytes:
        self._limiter.wait()
        try:
            with self._http.stream("GET", url, params=params) as resp:
                status = resp.status_code
                if status == 429 or status >= 500:
                    raise ArxivNetworkError(f"HTTP {status} from {url}")
                if status == 404:
                    raise ArxivNotFoundError(f"HTTP 404 from {url}")
                if status != 200:
                    raise ArxivHTTPError(f"HTTP {status} from {url}")
                buf = bytearray()
                for chunk in resp.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > self._max_bytes:
                        raise ArxivResponseError(
                            f"Response from {url} exceeds {self._cfg.max_download_mb} MB"
                        )
                return bytes(buf)
        except httpx.TransportError as exc:
            raise ArxivNetworkError(f"{type(exc).__name__} for {url}") from exc

    def _get(self, url: str, params: dict[str, str] | None = None) -> bytes:
        return self._retrying(self._get_once, url, params)

    def fetch_metadata(self, arxiv_id: str) -> ArxivMetadata:
        data = self._get(self._cfg.api_url, {"id_list": arxiv_id, "max_results": "1"})
        return parse_metadata(data, arxiv_id)

    def fetch_metadata_many(self, arxiv_ids: list[str]) -> list[ArxivMetadata]:
        """Metadata for several IDs in one API request; unknown IDs are simply missing."""
        if not arxiv_ids:
            return []
        params = {"id_list": ",".join(arxiv_ids), "max_results": str(len(arxiv_ids))}
        return parse_feed(self._get(self._cfg.api_url, params))

    def search(self, search_query: str, max_results: int) -> list[ArxivMetadata]:
        """Run an arXiv API ``search_query`` (already in arXiv query syntax), by relevance."""
        params = {
            "search_query": search_query,
            "max_results": str(max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        return parse_feed(self._get(self._cfg.api_url, params))

    def fetch_eprint(self, versioned_id: str) -> bytes:
        return self._get(f"{self._cfg.eprint_url}/{versioned_id}")

    def fetch_pdf(self, versioned_id: str) -> bytes:
        data = self._get(f"{self._cfg.pdf_url}/{versioned_id}")
        if not data.startswith(b"%PDF"):
            raise ArxivResponseError(f"PDF for {versioned_id} is not a PDF file")
        return data


# --------------------------------------------------------------------------- parsing


def _text(el: ET.Element | None) -> str | None:
    if el is None or el.text is None:
        return None
    return " ".join(el.text.split()) or None


def _parse_root(xml_bytes: bytes, what: str) -> ET.Element:
    try:
        return ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ArxivResponseError(f"Invalid XML from arXiv API for {what}") from exc


def _parse_entry(entry: ET.Element) -> ArxivMetadata | None:
    """One Atom ``<entry>``; None if it is not a paper (e.g. an API error entry)."""
    entry_id = _text(entry.find(f"{_ATOM}id")) or ""
    match = _ENTRY_ID_RE.search(entry_id)
    title = _text(entry.find(f"{_ATOM}title"))
    if "/api/errors" in entry_id or match is None or not title:
        return None

    published = _text(entry.find(f"{_ATOM}published"))
    year = int(published[:4]) if published and published[:4].isdigit() else None

    categories: list[str] = []
    primary = entry.find(f"{_ARXIV}primary_category")
    if primary is not None and primary.get("term"):
        categories.append(primary.get("term", ""))
    for cat in entry.findall(f"{_ATOM}category"):
        term = cat.get("term")
        if term and term not in categories:
            categories.append(term)

    authors = [
        name
        for a in entry.findall(f"{_ATOM}author")
        if (name := _text(a.find(f"{_ATOM}name"))) is not None
    ]
    base = match["base"]
    return ArxivMetadata(
        arxiv_id=base,
        version=int(match["version"]),
        title=title,
        authors=authors,
        year=year,
        abstract=_text(entry.find(f"{_ATOM}summary")),
        categories=categories,
        url=f"https://arxiv.org/abs/{base}v{match['version']}",
    )


def parse_metadata(xml_bytes: bytes, requested_id: str) -> ArxivMetadata:
    """Parse an arXiv API Atom response for a single ID."""
    root = _parse_root(xml_bytes, requested_id)
    entries = root.findall(f"{_ATOM}entry")
    if not entries:
        raise ArxivNotFoundError(f"arXiv has no entry for {requested_id}")
    entry = entries[0]
    entry_id = _text(entry.find(f"{_ATOM}id")) or ""
    if "/api/errors" in entry_id:
        summary = _text(entry.find(f"{_ATOM}summary")) or "unknown error"
        raise ArxivNotFoundError(f"arXiv API error for {requested_id}: {summary}")
    meta = _parse_entry(entry)
    if meta is None:
        raise ArxivNotFoundError(f"arXiv has no entry for {requested_id}")
    requested_base = re.sub(r"v\d+$", "", requested_id)
    if meta.arxiv_id != requested_base:
        raise ArxivResponseError(f"arXiv returned {meta.arxiv_id} when asked for {requested_id}")
    return meta


def parse_feed(xml_bytes: bytes) -> list[ArxivMetadata]:
    """Parse every paper entry of an arXiv API Atom feed (search or multi-ID lookup).

    An API error entry (e.g. malformed query) raises :class:`ArxivHTTPError`.
    """
    root = _parse_root(xml_bytes, "feed")
    results: list[ArxivMetadata] = []
    for entry in root.findall(f"{_ATOM}entry"):
        entry_id = _text(entry.find(f"{_ATOM}id")) or ""
        if "/api/errors" in entry_id:
            summary = _text(entry.find(f"{_ATOM}summary")) or "unknown error"
            raise ArxivHTTPError(f"arXiv API rejected the request: {summary}")
        meta = _parse_entry(entry)
        if meta is not None:
            results.append(meta)
    return results


def _gunzip_limited(data: bytes, max_bytes: int) -> bytes:
    d = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        out = d.decompress(data, max_bytes + 1)
    except zlib.error as exc:
        raise EprintFormatError("Corrupt gzip data in e-print") from exc
    if len(out) > max_bytes:
        raise EprintFormatError(f"Decompressed e-print exceeds {max_bytes // (1024 * 1024)} MB")
    return out


def _open_tar(payload: bytes) -> tarfile.TarFile | None:
    try:
        return tarfile.open(fileobj=io.BytesIO(payload), mode="r:")
    except tarfile.TarError:
        return None


def extract_eprint(data: bytes, source_dir: Path, max_extract_bytes: int) -> EprintResult:
    """Classify an e-print payload and extract LaTeX into ``source_dir``.

    Handles: gzipped tarball, single gzipped .tex, and PDF-only submissions.
    The payload may already have been gunzipped by the HTTP layer (Content-Encoding).
    """
    payload = data
    if payload[:2] == b"\x1f\x8b":
        payload = _gunzip_limited(payload, max_extract_bytes)

    if payload.startswith(b"%PDF"):
        return EprintResult(kind="pdf", pdf_bytes=payload)

    tar = _open_tar(payload)
    if tar is not None:
        with tar:
            members = tar.getmembers()
            if not members:
                raise EprintFormatError("Empty tarball")
            if len(members) > _MAX_TAR_MEMBERS:
                raise EprintFormatError(f"Tarball has more than {_MAX_TAR_MEMBERS} members")
            if sum(m.size for m in members) > max_extract_bytes:
                raise EprintFormatError("Tarball content exceeds extraction size limit")
            source_dir.mkdir(parents=True, exist_ok=True)
            try:
                tar.extractall(source_dir, filter="data")
            except (tarfile.TarError, OSError) as exc:
                raise EprintFormatError(f"Unsafe or corrupt tarball: {exc}") from exc
        return EprintResult(kind="tarball")

    if _TEX_MARKERS.search(payload[:200_000]):
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "main.tex").write_bytes(payload)
        return EprintResult(kind="single_tex")

    raise EprintFormatError("E-print is neither a tarball, a TeX file nor a PDF")


# --------------------------------------------------------------------------- corpus


def load_paper_list(path: Path) -> list[PaperListEntry]:
    """Load and validate data/papers.yaml. Duplicate IDs are dropped with a warning."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a YAML list")
    entries: list[PaperListEntry] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        try:
            entry = PaperListEntry.model_validate(item)
        except ValidationError as exc:
            raise ValueError(f"{path}: invalid entry #{i + 1}: {item!r}") from exc
        if not is_valid_arxiv_id(entry.arxiv_id):
            raise ValueError(f"{path}: invalid arXiv ID {entry.arxiv_id!r}")
        if entry.arxiv_id in seen:
            logger.warning("Duplicate arXiv ID %s in %s; ignoring", entry.arxiv_id, path)
            continue
        seen.add(entry.arxiv_id)
        entries.append(entry)
    return entries


def is_valid_arxiv_id(arxiv_id: str) -> bool:
    """True for new-style (``2401.12345``) and old-style (``hep-th/9901001``) IDs."""
    return bool(_ARXIV_ID_RE.match(arxiv_id))


def paper_dir(raw_dir: Path, arxiv_id: str) -> Path:
    # Old-style IDs contain '/', which must not create nested directories.
    return raw_dir / arxiv_id.replace("/", "_")


def load_complete(target: Path) -> Paper | None:
    """Return the stored Paper if ``target`` holds a complete download, else None."""
    meta_path = target / META_FILE
    if not meta_path.is_file() or not (target / PDF_FILE).is_file():
        return None
    try:
        paper = Paper.model_validate_json(meta_path.read_text(encoding="utf-8"))
    except (ValidationError, OSError):
        return None
    if paper.latex_available and not (target / SOURCE_DIR).is_dir():
        return None
    return paper


def download_paper(
    entry: PaperListEntry, client: ArxivClient, raw_dir: Path, max_extract_bytes: int
) -> Paper:
    """Download one paper into ``raw_dir``. Raises ArxivError/OSError on failure."""
    target = paper_dir(raw_dir, entry.arxiv_id)
    tmp = target.with_name(f".{target.name}.partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        meta = client.fetch_metadata(entry.arxiv_id)
        vid = meta.versioned_id

        eprint: EprintResult | None = None
        try:
            eprint = extract_eprint(client.fetch_eprint(vid), tmp / SOURCE_DIR, max_extract_bytes)
        except ArxivNetworkError:
            raise  # transient: fail the paper so a re-run retries it
        except ArxivError as exc:
            logger.warning("No LaTeX source for %s: %s", vid, exc)
            shutil.rmtree(tmp / SOURCE_DIR, ignore_errors=True)

        pdf = eprint.pdf_bytes if eprint and eprint.pdf_bytes else client.fetch_pdf(vid)
        (tmp / PDF_FILE).write_bytes(pdf)

        paper = Paper(
            arxiv_id=meta.arxiv_id,
            title=meta.title,
            authors=meta.authors,
            year=meta.year,
            abstract=meta.abstract,
            url=meta.url,
            categories=meta.categories,
            version=meta.version,
            topic=entry.topic,
            latex_available=eprint is not None and eprint.kind != "pdf",
        )
        (tmp / META_FILE).write_text(paper.model_dump_json(indent=2), encoding="utf-8")

        if target.exists():
            shutil.rmtree(target)  # incomplete earlier attempt
        tmp.rename(target)
        return paper
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def build_corpus(
    entries: Iterable[PaperListEntry],
    client: ArxivClient,
    raw_dir: Path,
    config: ArxivConfig,
    sink: EventSink | None = None,
) -> CorpusReport:
    """Download every paper in ``entries``; skip complete ones, log and continue on failure."""
    sink = sink or NullSink()
    raw_dir.mkdir(parents=True, exist_ok=True)
    report = CorpusReport()
    max_extract = config.max_extract_mb * 1024 * 1024

    for entry in entries:
        step = f"arxiv_fetch:{entry.arxiv_id}"
        sink.step_started(step, "arxiv_fetch", {"arxiv_id": entry.arxiv_id})

        existing = load_complete(paper_dir(raw_dir, entry.arxiv_id))
        if existing is not None:
            report.skipped.append(entry.arxiv_id)
            if existing.latex_available:
                report.latex_available.append(entry.arxiv_id)
            sink.step_finished(step, "skipped", {"latex_available": existing.latex_available})
            continue

        try:
            paper = download_paper(entry, client, raw_dir, max_extract)
        except (ArxivError, OSError) as exc:
            logger.error("Failed to download %s: %s", entry.arxiv_id, exc)
            report.failed.append(FailedPaper(arxiv_id=entry.arxiv_id, error=str(exc)))
            sink.step_finished(step, "failed", {"error": str(exc)})
            continue

        report.downloaded.append(entry.arxiv_id)
        if paper.latex_available:
            report.latex_available.append(entry.arxiv_id)
        logger.info(
            "Downloaded %s v%s (latex=%s): %s",
            paper.arxiv_id,
            paper.version,
            paper.latex_available,
            paper.title,
        )
        sink.step_finished(step, "downloaded", {"latex_available": paper.latex_available})

    return report


__all__ = [
    "ArxivClient",
    "ArxivError",
    "ArxivHTTPError",
    "ArxivMetadata",
    "ArxivNetworkError",
    "ArxivNotFoundError",
    "ArxivResponseError",
    "CorpusReport",
    "EprintFormatError",
    "EprintResult",
    "FailedPaper",
    "RateLimiter",
    "build_corpus",
    "download_paper",
    "extract_eprint",
    "is_valid_arxiv_id",
    "load_complete",
    "load_paper_list",
    "parse_feed",
    "parse_metadata",
]
