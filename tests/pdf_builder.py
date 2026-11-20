"""Minimal PDF writer for tests: text (size, colour, bold) and filled rectangles.

Avoids a PDF-generation dependency. Coordinates are PDF points with the origin at the
bottom-left of a US-letter page; drawing order is list order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PAGE_W, PAGE_H = 612, 792


@dataclass(frozen=True)
class Text:
    x: float
    y: float
    text: str
    size: float = 10
    color: tuple[float, float, float] = (0, 0, 0)
    bold: bool = False
    rotated: bool = False  # 90° counter-clockwise, like the arXiv side stamp


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float
    fill: tuple[float, float, float] = (0, 0, 0)


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _rgb(c: tuple[float, float, float]) -> str:
    return " ".join(f"{v:g}" for v in c)


def _content(items: list[Text | Rect]) -> bytes:
    ops: list[str] = []
    for it in items:
        if isinstance(it, Rect):
            ops.append(f"{_rgb(it.fill)} rg {it.x:g} {it.y:g} {it.w:g} {it.h:g} re f")
        else:
            font = "/F2" if it.bold else "/F1"
            place = f"0 1 -1 0 {it.x:g} {it.y:g} Tm" if it.rotated else f"{it.x:g} {it.y:g} Td"
            ops.append(
                f"BT {font} {it.size:g} Tf {_rgb(it.color)} rg {place} ({_escape(it.text)}) Tj ET"
            )
    return "\n".join(ops).encode("latin-1")


def build_pdf(path: Path, pages: list[list[Text | Rect]]) -> None:
    n = len(pages)
    page_ids = [5 + 2 * i for i in range(n)]
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    }
    for pid, items in zip(page_ids, pages, strict=True):
        content = _content(items)
        objects[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {pid + 1} 0 R >>"
        ).encode()
        objects[pid + 1] = (
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n"
    xref = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for num in range(1, size):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
