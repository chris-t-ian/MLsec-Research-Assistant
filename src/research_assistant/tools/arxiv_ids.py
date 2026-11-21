"""Strict arXiv identifier validation.

Accepted forms (an optional ``arXiv:`` prefix is stripped, case-insensitive):

* new style (April 2007 on): ``YYMM.NNNN`` up to 1412, ``YYMM.NNNNN`` from 1501, e.g.
  ``0704.0001``, ``2112.03570``, optionally versioned: ``2112.03570v2``;
* old style (1991 to March 2007): ``archive[.XX]/YYMMNNN``, e.g. ``cs/0112017``,
  ``hep-th/9901001``, ``math.GT/0309136``, optionally versioned.
"""

from __future__ import annotations

import re
from typing import Annotated, NamedTuple

from pydantic import AfterValidator

_NEW = re.compile(r"(?P<yy>\d{2})(?P<mm>\d{2})\.(?P<num>\d{4,5})(?:v(?P<v>[1-9]\d{0,2}))?")
_OLD = re.compile(
    r"(?P<archive>[a-z]+(?:-[a-z]+)?)(?:\.(?P<subj>[A-Z]{2}))?/"
    r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<num>\d{3})(?:v(?P<v>[1-9]\d{0,2}))?"
)
_PREFIX = re.compile(r"^arxiv:", re.IGNORECASE)
MAX_ID_LEN = 40


class ArxivIdError(ValueError):
    """The string is not a valid arXiv identifier (message is user-safe)."""


class ArxivId(NamedTuple):
    base: str  # versionless, e.g. "2112.03570" or "cs/0112017"
    version: int | None

    def __str__(self) -> str:
        return self.base if self.version is None else f"{self.base}v{self.version}"


def parse_arxiv_id(raw: str) -> ArxivId:
    """Validate and normalize an arXiv ID. Raises :class:`ArxivIdError`."""
    text = _PREFIX.sub("", raw.strip())
    shown = text[:MAX_ID_LEN]
    if not text or len(text) > MAX_ID_LEN:
        raise ArxivIdError(f"{shown!r} is not a valid arXiv ID")

    if m := _NEW.fullmatch(text):
        yy, mm, num = int(m["yy"]), int(m["mm"]), m["num"]
        if not 1 <= mm <= 12 or (yy, mm) < (7, 4):
            raise ArxivIdError(f"{shown!r} is not a valid arXiv ID (bad year/month)")
        if (yy, mm) < (15, 1) and len(num) != 4:
            raise ArxivIdError(f"{shown!r}: IDs before 2015 have a 4-digit number")
        if (yy, mm) >= (15, 1) and len(num) != 5:
            raise ArxivIdError(f"{shown!r}: IDs since 2015 have a 5-digit number")
        base = f"{m['yy']}{m['mm']}.{num}"
    elif m := _OLD.fullmatch(text):
        yy, mm = int(m["yy"]), int(m["mm"])
        year = 1900 + yy if yy >= 91 else 2000 + yy
        if not 1 <= mm <= 12 or not (1991, 1) <= (year, mm) <= (2007, 3):
            raise ArxivIdError(f"{shown!r} is not a valid old-style arXiv ID (bad year/month)")
        subj = f".{m['subj']}" if m["subj"] else ""
        base = f"{m['archive']}{subj}/{m['yy']}{m['mm']}{m['num']}"
    else:
        raise ArxivIdError(
            f"{shown!r} is not a valid arXiv ID (expected e.g. 2112.03570, 2112.03570v2 "
            "or cs/0112017)"
        )
    return ArxivId(base=base, version=int(m["v"]) if m["v"] else None)


def _validate(value: str) -> str:
    return str(parse_arxiv_id(value))


# Pydantic field type: validates and normalizes (e.g. strips an "arXiv:" prefix).
ArxivIdStr = Annotated[str, AfterValidator(_validate)]
