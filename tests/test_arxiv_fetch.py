"""Unit tests for the arXiv corpus downloader. All HTTP is mocked via httpx.MockTransport."""

from __future__ import annotations

import gzip
import io
import tarfile
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import httpx
import pytest
from tenacity import wait_none

from research_assistant.config import ArxivConfig
from research_assistant.events import RecordingSink
from research_assistant.ingest.arxiv_fetch import (
    ArxivClient,
    ArxivNotFoundError,
    EprintFormatError,
    RateLimiter,
    build_corpus,
    extract_eprint,
    load_complete,
    load_paper_list,
    parse_metadata,
)
from research_assistant.models import Paper, PaperListEntry

CFG = ArxivConfig(max_retries=2)
PDF = b"%PDF-1.5\n fake pdf body\n%%EOF"
TEX = b"\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}\n"


def atom(arxiv_id: str, version: int = 2, title: str = "A  Test\n Paper") -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}v{version}</id>
    <published>2016-10-18T00:00:00Z</published>
    <title>{title}</title>
    <summary>  An abstract
      on two lines. </summary>
    <author><name>Alice Example</name></author>
    <author><name>Bob Example</name></author>
    <arxiv:primary_category term="cs.CR" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CR" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>""".encode()


EMPTY_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""


def tarball(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


Handler = Callable[[httpx.Request], httpx.Response]


def arxiv_handler(eprints: dict[str, bytes | int], calls: list[str]) -> Handler:
    """Serve metadata for any ID, e-prints from `eprints` (bytes or HTTP status), PDFs always."""

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        path = request.url.path
        if path.endswith("/api/query"):
            arxiv_id = request.url.params["id_list"]
            return httpx.Response(200, content=atom(arxiv_id))
        if "/e-print/" in path:
            base = path.rsplit("/", 1)[-1].rsplit("v", 1)[0]
            body = eprints[base]
            if isinstance(body, int):
                return httpx.Response(body)
            return httpx.Response(200, content=body)
        if "/pdf/" in path:
            return httpx.Response(200, content=PDF)
        return httpx.Response(404)

    return handle


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(handler: Handler, clock: FakeClock | None = None) -> ArxivClient:
    clock = clock or FakeClock()
    return ArxivClient(
        CFG,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        limiter=RateLimiter(CFG.min_interval_s, clock=clock, sleep=clock.sleep),
        retry_wait=wait_none(),
    )


def run(
    tmp_path: Path, ids: list[str], handler: Handler, clock: FakeClock | None = None
) -> tuple[Path, object]:
    raw = tmp_path / "raw"
    entries = [PaperListEntry(arxiv_id=i, topic="membership_inference") for i in ids]
    with make_client(handler, clock) as client:
        report = build_corpus(entries, client, raw, CFG, sink=RecordingSink())
    return raw, report


# --------------------------------------------------------------------------- e-print variants


def test_gzipped_tarball(tmp_path: Path) -> None:
    calls: list[str] = []
    eprint = tarball({"main.tex": TEX, "sections/intro.tex": b"\\section{Intro}"})
    raw, report = run(tmp_path, ["1610.05820"], arxiv_handler({"1610.05820": eprint}, calls))

    assert report.downloaded == ["1610.05820"] and report.latex_available == ["1610.05820"]
    d = raw / "1610.05820"
    assert (d / "source" / "main.tex").read_bytes() == TEX
    assert (d / "source" / "sections" / "intro.tex").is_file()
    assert (d / "paper.pdf").read_bytes() == PDF
    paper = Paper.model_validate_json((d / "meta.json").read_text())
    assert paper.title == "A Test Paper"
    assert paper.authors == ["Alice Example", "Bob Example"]
    assert paper.year == 2016 and paper.version == 2
    assert paper.categories == ["cs.CR", "cs.LG"]
    assert paper.abstract == "An abstract on two lines."
    assert paper.topic == "membership_inference" and paper.latex_available
    # Versioned URLs are used for source and PDF.
    assert any(c.endswith("/e-print/1610.05820v2") for c in calls)
    assert any(c.endswith("/pdf/1610.05820v2") for c in calls)
    assert not list(raw.glob(".*.partial"))


def test_single_gzipped_tex(tmp_path: Path) -> None:
    calls: list[str] = []
    eprint = gzip.compress(TEX)
    raw, report = run(tmp_path, ["1709.01604"], arxiv_handler({"1709.01604": eprint}, calls))

    assert report.downloaded == ["1709.01604"] and report.latex_available == ["1709.01604"]
    assert (raw / "1709.01604" / "source" / "main.tex").read_bytes() == TEX
    assert (raw / "1709.01604" / "paper.pdf").is_file()


def test_pdf_only_eprint(tmp_path: Path) -> None:
    calls: list[str] = []
    raw, report = run(tmp_path, ["1806.01246"], arxiv_handler({"1806.01246": PDF}, calls))

    assert report.downloaded == ["1806.01246"] and report.latex_available == []
    d = raw / "1806.01246"
    assert (d / "paper.pdf").read_bytes() == PDF
    assert not (d / "source").exists()
    assert not Paper.model_validate_json((d / "meta.json").read_text()).latex_available
    # The e-print already was the PDF, so no separate PDF request.
    assert not any("/pdf/" in c for c in calls)


def test_eprint_404_falls_back_to_pdf(tmp_path: Path) -> None:
    calls: list[str] = []
    raw, report = run(tmp_path, ["1908.11229"], arxiv_handler({"1908.11229": 404}, calls))
    assert report.downloaded == ["1908.11229"] and report.latex_available == []
    assert (raw / "1908.11229" / "paper.pdf").is_file()


# --------------------------------------------------------------------------- failures


def test_network_failure_logs_and_continues(tmp_path: Path) -> None:
    calls: list[str] = []
    ok = arxiv_handler({"2112.03570": tarball({"main.tex": TEX})}, calls)
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("id_list") == "1610.05820":
            attempts.append(str(request.url))
            raise httpx.ConnectError("boom", request=request)
        return ok(request)

    raw, report = run(tmp_path, ["1610.05820", "2112.03570"], handler)

    assert [f.arxiv_id for f in report.failed] == ["1610.05820"]
    assert report.downloaded == ["2112.03570"]
    assert len(attempts) == CFG.max_retries + 1  # retried, then gave up
    assert not (raw / "1610.05820").exists()
    assert not list(raw.glob(".*.partial"))


def test_server_error_then_success_is_retried(tmp_path: Path) -> None:
    calls: list[str] = []
    ok = arxiv_handler({"1610.05820": tarball({"main.tex": TEX})}, calls)
    failures = iter([503])

    def handler(request: httpx.Request) -> httpx.Response:
        if "/pdf/" in request.url.path:
            status = next(failures, None)
            if status:
                return httpx.Response(status)
        return ok(request)

    _, report = run(tmp_path, ["1610.05820"], handler)
    assert report.downloaded == ["1610.05820"] and not report.failed


def test_unknown_id_fails_without_retry(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=EMPTY_FEED)

    _, report = run(tmp_path, ["2609.37344"], handler)
    assert [f.arxiv_id for f in report.failed] == ["2609.37344"]
    assert len(calls) == 1


def test_eprint_network_failure_fails_paper(tmp_path: Path) -> None:
    calls: list[str] = []
    ok = arxiv_handler({}, calls)

    def handler(request: httpx.Request) -> httpx.Response:
        if "/e-print/" in request.url.path:
            raise httpx.ReadTimeout("slow", request=request)
        return ok(request)

    raw, report = run(tmp_path, ["1610.05820"], handler)
    assert [f.arxiv_id for f in report.failed] == ["1610.05820"]
    assert not (raw / "1610.05820").exists()


# --- idempotency & rate limit


def test_second_run_skips_downloaded(tmp_path: Path) -> None:
    calls: list[str] = []
    handler = arxiv_handler({"1610.05820": tarball({"main.tex": TEX})}, calls)
    run(tmp_path, ["1610.05820"], handler)
    calls.clear()

    _, report = run(tmp_path, ["1610.05820"], handler)
    assert report.skipped == ["1610.05820"] and report.downloaded == []
    assert report.latex_available == ["1610.05820"]
    assert calls == []


def test_incomplete_download_is_redone(tmp_path: Path) -> None:
    calls: list[str] = []
    handler = arxiv_handler({"1610.05820": PDF}, calls)
    raw, _ = run(tmp_path, ["1610.05820"], handler)
    (raw / "1610.05820" / "paper.pdf").unlink()
    assert load_complete(raw / "1610.05820") is None

    _, report = run(tmp_path, ["1610.05820"], handler)
    assert report.downloaded == ["1610.05820"]


def test_requests_are_at_least_3s_apart(tmp_path: Path) -> None:
    calls: list[str] = []
    clock = FakeClock()
    times: list[float] = []
    inner = arxiv_handler({"1610.05820": tarball({"main.tex": TEX}), "1709.01604": PDF}, calls)

    def handler(request: httpx.Request) -> httpx.Response:
        times.append(clock.now)
        return inner(request)

    run(tmp_path, ["1610.05820", "1709.01604"], handler, clock)
    assert len(times) == 5  # meta+eprint+pdf, meta+eprint(pdf)
    gaps = [b - a for a, b in pairwise(times)]
    assert all(g >= 3.0 for g in gaps), gaps


def test_min_interval_cannot_be_below_3s() -> None:
    with pytest.raises(ValueError):
        ArxivConfig(min_interval_s=1.0)


# --------------------------------------------------------------------------- helpers


def test_extract_rejects_path_traversal(tmp_path: Path) -> None:
    evil = tarball({"../escape.tex": TEX})
    with pytest.raises(EprintFormatError):
        extract_eprint(evil, tmp_path / "src", 10_000_000)
    assert not (tmp_path / "escape.tex").exists()


def test_extract_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(EprintFormatError):
        extract_eprint(gzip.compress(b"just some words"), tmp_path / "src", 10_000_000)


def test_extract_rejects_oversized_gzip(tmp_path: Path) -> None:
    with pytest.raises(EprintFormatError):
        extract_eprint(gzip.compress(b"\\section{x}" + b"a" * 5000), tmp_path / "src", 1000)


def test_parse_metadata_api_error_entry() -> None:
    xml = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/api/errors#incorrect_id_format_for_x</id>
      <title>Error</title><summary>incorrect id format for x</summary></entry></feed>"""
    with pytest.raises(ArxivNotFoundError):
        parse_metadata(xml, "x")


def test_load_paper_list(tmp_path: Path) -> None:
    f = tmp_path / "papers.yaml"
    f.write_text(
        '- {arxiv_id: "2212.06470", topic: differential_privacy}\n'
        '- {arxiv_id: "2212.06470", topic: differential_privacy}\n'
        '- {arxiv_id: "hep-th/9901001", topic: other}\n'
    )
    entries = load_paper_list(f)
    assert [e.arxiv_id for e in entries] == ["2212.06470", "hep-th/9901001"]

    f.write_text('- {arxiv_id: "2212.06470", topic: not_a_topic}\n')
    with pytest.raises(ValueError):
        load_paper_list(f)


def test_project_paper_list_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    entries = load_paper_list(root / "data" / "papers.yaml")
    assert entries
