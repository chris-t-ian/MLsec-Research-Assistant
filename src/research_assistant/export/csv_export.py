"""CSV exports: one row per message, and one row per citation.

Files are UTF-8 with BOM so spreadsheet apps detect the encoding (epsilon, delta, accents).
Cell values that a spreadsheet would evaluate as formulas are prefixed with an apostrophe:
answers and excerpts contain text from papers, which is untrusted.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from typing import Any

from research_assistant.export.models import Conversation

MESSAGE_COLUMNS = ("message_id", "timestamp", "role", "content", "strategy", "cited_arxiv_ids")
CITATION_COLUMNS = ("message_id", "marker", "arxiv_id", "section", "excerpt")

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
ID_SEPARATOR = ";"


def safe_cell(value: Any) -> str:
    """Stringify a cell and neutralize spreadsheet formula injection."""
    text = "" if value is None else str(value)
    if text.startswith(_FORMULA_PREFIXES):
        return "'" + text
    return text


def _to_csv(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([safe_cell(v) for v in row])
    return buf.getvalue().encode("utf-8-sig")


def export_messages_csv(conversation: Conversation) -> bytes:
    rows = (
        (
            m.id,
            m.created_at.isoformat() if m.created_at else "",
            m.role,
            m.content,
            m.strategy or "",
            ID_SEPARATOR.join(m.cited_arxiv_ids()),
        )
        for m in conversation.messages
    )
    return _to_csv(MESSAGE_COLUMNS, rows)


def export_citations_csv(conversation: Conversation) -> bytes:
    rows = (
        (m.id, f"[{c.number}]", c.arxiv_id, c.section or "", c.excerpt)
        for m in conversation.messages
        for c in m.citations
    )
    return _to_csv(CITATION_COLUMNS, rows)
