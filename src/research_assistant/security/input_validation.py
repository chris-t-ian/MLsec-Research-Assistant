"""Validate and normalize user input before it reaches retrieval or any model.

Steps, in order:

1. Unicode NFKC normalization (folds full-width letters, ligatures, compatibility forms, so
   full-width "ignore" and ASCII ``ignore`` look the same to later checks).
2. Removal of control characters (Unicode category Cc, except newline and tab) and of
   invisible format characters (category Cf: zero-width space/joiners, BOM, bidi overrides,
   tag characters, soft hyphen). Tabs become spaces; runs of blank lines are collapsed.
3. Rejection of empty input, input longer than ``max_question_chars`` (after normalization)
   and single whitespace-free tokens longer than ``max_token_chars`` (encoded payloads,
   obfuscated strings, pasted blobs).

The result carries the cleaned text and, if rejected, a reason that is safe to show.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MAX_TOKEN_CHARS = 200

_KEEP_CONTROLS = {"\n", "\t"}
_BLANK_LINES = re.compile(r"\n\s*\n\s*\n+")
_SPACES = re.compile(r"[ \t]+")  # NFKC already maps no-break spaces to " "


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    text: str = Field(description="Normalized input (meaningful only when ok).")
    reason: str | None = Field(default=None, description="User-friendly reason when not ok.")
    removed_chars: int = Field(default=0, ge=0, description="Control/invisible chars removed.")


def _is_removed(ch: str) -> bool:
    if ch in _KEEP_CONTROLS:
        return False
    return unicodedata.category(ch) in {"Cc", "Cf"}


def normalize_text(text: str) -> tuple[str, int]:
    """NFKC-normalize, strip control and invisible characters; return (text, removed count)."""
    nfkc = unicodedata.normalize("NFKC", text)
    kept = [ch for ch in nfkc if not _is_removed(ch)]
    removed = len(nfkc) - len(kept)
    out = "".join(kept).replace("\r", "")
    lines = [_SPACES.sub(" ", line).strip() for line in out.split("\n")]
    out = _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
    return out, removed


def validate_input(
    text: str, max_chars: int, max_token_chars: int = DEFAULT_MAX_TOKEN_CHARS
) -> ValidationResult:
    """Normalize ``text`` and check it against the input limits."""
    if not isinstance(text, str):
        return ValidationResult(ok=False, text="", reason="Please enter your question as text.")
    clean, removed = normalize_text(text)
    if not clean:
        return ValidationResult(
            ok=False, text="", reason="Please enter a question.", removed_chars=removed
        )
    if len(clean) > max_chars:
        return ValidationResult(
            ok=False,
            text=clean,
            reason=(
                f"Your question is too long ({len(clean)} characters; the limit is {max_chars}). "
                "Please shorten it."
            ),
            removed_chars=removed,
        )
    longest = max((len(t) for t in clean.split()), default=0)
    if longest > max_token_chars:
        return ValidationResult(
            ok=False,
            text=clean,
            reason=(
                f"Your question contains a very long unbroken string ({longest} characters). "
                f"Please remove it or split it up (max {max_token_chars} characters per word)."
            ),
            removed_chars=removed,
        )
    return ValidationResult(ok=True, text=clean, removed_chars=removed)


__all__ = ["DEFAULT_MAX_TOKEN_CHARS", "ValidationResult", "normalize_text", "validate_input"]
