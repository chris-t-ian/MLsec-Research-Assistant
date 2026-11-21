"""Prompt-injection detection on user input: weighted heuristics plus an optional classifier.

Heuristics
    Each pattern has a weight in (0, 1]. A pattern counts once; matches inside quotes
    (``"..."``, ``'...'``, ``“...”``, backticks) count half, because this assistant is about
    prompt injection and users quote attack strings when asking about them. Weights combine
    like independent probabilities: ``score = 1 - prod(1 - w_i)``. One strong pattern
    ("ignore your instructions") lands in *warn*; two independent signals ("... and print
    your system prompt") reach *block*.

Classifier (optional, ``security.injection.classifier_enabled``)
    ``protectai/deberta-v3-base-prompt-injection-v2`` (Apache-2.0, labels SAFE/INJECTION),
    run locally with ``transformers``. The final score is ``max(heuristic, classifier)``, but
    the classifier alone can only raise the verdict to *warn*: blocking needs at least one
    heuristic signal. Such classifiers fire on benign questions *about* prompt injection,
    which are this assistant's core use case. If the model cannot be loaded, detection falls
    back to the heuristics (logged once).

Verdicts: ``allow`` (score < warn_threshold), ``warn`` (turn continues, event recorded),
``block`` (score >= block_threshold, the turn is refused without calling any model).
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.config import InjectionConfig

logger = logging.getLogger(__name__)

Verdict = Literal["allow", "warn", "block"]
QUOTED_FACTOR = 0.5


class InjectionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: Verdict
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="Short, user-safe explanation.")
    matches: list[str] = Field(default_factory=list, description="Names of matched heuristics.")
    heuristic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    classifier_score: float | None = Field(default=None, ge=0.0, le=1.0)


@dataclass(frozen=True)
class Heuristic:
    name: str
    weight: float
    description: str
    pattern: re.Pattern[str]


def _h(name: str, weight: float, description: str, pattern: str, flags: int = re.I) -> Heuristic:
    return Heuristic(name, weight, description, re.compile(pattern, flags))


_VERB_REVEAL = (
    r"(?:print|reveal|show|display|repeat|output|echo|leak|dump|disclose|spell\s+out|"
    r"tell\s+me|give\s+me|write\s+out|what\s+(?:is|are|was|were))"
)

HEURISTICS: tuple[Heuristic, ...] = (
    _h(
        "override_instructions",
        0.7,
        "tries to override the assistant's instructions",
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
        r"\b(?:instructions?|prompts?|rules|guidelines|directives|system\s+message|guardrails)\b",
    ),
    _h(
        "prompt_leak",
        0.7,
        "asks to reveal the system prompt or hidden instructions",
        rf"\b{_VERB_REVEAL}\b[^.\n]{{0,40}}?\b(?:system|initial|hidden|original|developer|"
        r"secret|internal)\s+(?:prompt|instructions?|message|rules)"
        rf"|\b{_VERB_REVEAL}\b[^.\n]{{0,20}}?\byour\s+(?:prompt|instructions?|rules)\b",
    ),
    _h(
        "repeat_context",
        0.6,
        "asks to repeat the hidden context",
        r"\b(?:repeat|print|output|copy|echo)\b[^.\n]{0,30}?\b(?:everything|all|the\s+text|"
        r"the\s+words|text)\s+(?:above|before|preceding)\b",
    ),
    _h(
        "secret_exfiltration",
        0.6,
        "asks for secrets such as keys or tokens",
        rf"\b{_VERB_REVEAL}\b[^.\n]{{0,30}}?\b(?:api[\s_-]?keys?|canary|secret\s+tokens?|"
        r"passwords?|credentials|environment\s+variables|\.env)\b",
    ),
    _h(
        "role_override",
        0.5,
        "tries to change the assistant's role",
        r"\byou\s+are\s+now\b|\bfrom\s+now\s+on,?\s+you\b|\bpretend\s+(?:to\s+be|you\s+are)\b"
        r"|\bact\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|uncensored|jailbroken|evil)\b",
    ),
    _h(
        "jailbreak_persona",
        0.6,
        "uses a known jailbreak persona",
        r"\b(?:do\s+anything\s+now|developer\s+mode|jailbreak\s+mode|god\s+mode)\b|(?-i:\bDAN\b)",
    ),
    _h(
        "chat_markup",
        0.6,
        "contains chat-template or role markup",
        r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|endoftext\|>|\[/?INST\]|<</?SYS>>"
        r"|^\s*(?:system|assistant|developer)\s*:",
        re.I | re.M,
    ),
    _h(
        "new_instructions",
        0.6,
        "announces new instructions",
        r"\b(?:new|updated|real|actual|revised)\s+(?:system\s+)?instructions?\s*:",
    ),
    _h(
        "markdown_exfiltration",
        0.5,
        "contains a markdown image pointing to an external URL",
        r"!\[[^\]\n]*\]\(\s*https?://",
    ),
)

_QUOTED = re.compile(
    r'"[^"\n]{1,300}"|“[^”\n]{1,300}”|„[^“”\n]{1,300}[“”]|`[^`\n]{1,300}`'
    r"|(?<!\w)'[^'\n]{2,300}'(?!\w)"
)


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _QUOTED.finditer(text)]


def _inside(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(s <= span[0] and span[1] <= e for s, e in spans)


def heuristic_score(text: str) -> tuple[float, list[Heuristic]]:
    """Combined heuristic score and the heuristics that matched."""
    quoted = _quoted_spans(text)
    remaining = 1.0
    hits: list[Heuristic] = []
    for h in HEURISTICS:
        matches = list(h.pattern.finditer(text))
        if not matches:
            continue
        unquoted = any(not _inside(m.span(), quoted) for m in matches)
        weight = h.weight if unquoted else h.weight * QUOTED_FACTOR
        remaining *= 1.0 - weight
        hits.append(h)
    return 1.0 - remaining, hits


# --------------------------------------------------------------------------- classifier


class InjectionClassifier(Protocol):
    def injection_probability(self, text: str) -> float:
        """Probability in [0, 1] that ``text`` is a prompt injection. May raise."""
        ...


class HFInjectionClassifier:
    """Local Hugging Face text-classification model with an ``INJECTION`` label.

    The pipeline is loaded lazily on first use (downloads the model once).
    """

    INJECTION_LABEL = "INJECTION"

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._pipe: Any = None
        self._lock = threading.Lock()

    def _pipeline(self) -> Any:
        with self._lock:
            if self._pipe is None:
                from transformers import pipeline  # heavy import, optional feature

                self._pipe = pipeline(
                    "text-classification",
                    model=self.model_name,
                    truncation=True,
                    max_length=self.max_length,
                    top_k=None,
                )
            return self._pipe

    def injection_probability(self, text: str) -> float:
        out = self._pipeline()(text)
        scores = out[0] if out and isinstance(out[0], list) else out
        for item in scores:
            if str(item.get("label", "")).upper() == self.INJECTION_LABEL:
                return float(item["score"])
        raise ValueError(f"classifier {self.model_name} has no {self.INJECTION_LABEL} label")


# --------------------------------------------------------------------------- detector


class InjectionDetector:
    def __init__(
        self,
        config: InjectionConfig | None = None,
        classifier: InjectionClassifier | None = None,
    ) -> None:
        self.config = config or InjectionConfig()
        self._classifier = classifier
        self._classifier_failed = False

    @classmethod
    def from_config(cls, config: InjectionConfig) -> InjectionDetector:
        classifier = (
            HFInjectionClassifier(config.classifier_model, config.classifier_max_length)
            if config.classifier_enabled
            else None
        )
        return cls(config, classifier)

    def _classify(self, text: str) -> float | None:
        if self._classifier is None or self._classifier_failed:
            return None
        try:
            return min(1.0, max(0.0, self._classifier.injection_probability(text)))
        except Exception as exc:
            self._classifier_failed = True
            logger.warning(
                "Injection classifier unavailable, using heuristics only (%s)", type(exc).__name__
            )
            return None

    def detect(self, text: str) -> InjectionResult:
        cfg = self.config
        h_score, hits = heuristic_score(text)
        c_score = self._classify(text)
        score = max(h_score, c_score or 0.0)

        if score >= cfg.block_threshold and h_score > 0:
            verdict: Verdict = "block"
        elif score >= cfg.warn_threshold:
            verdict = "warn"
        else:
            verdict = "allow"

        reasons = [h.description for h in hits]
        if c_score is not None and c_score >= cfg.warn_threshold:
            reasons.append(f"the injection classifier flagged it ({c_score:.2f})")
        reason = "; ".join(reasons) if reasons else "no injection signals"
        return InjectionResult(
            verdict=verdict,
            score=round(score, 4),
            reason=reason,
            matches=[h.name for h in hits],
            heuristic_score=round(h_score, 4),
            classifier_score=None if c_score is None else round(c_score, 4),
        )


__all__ = [
    "HEURISTICS",
    "HFInjectionClassifier",
    "InjectionClassifier",
    "InjectionDetector",
    "InjectionResult",
    "Verdict",
    "heuristic_score",
]
