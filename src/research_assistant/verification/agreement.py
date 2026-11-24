"""Judge validation: sample judged claims from an eval run, collect hand labels, and report
the judge's accuracy and Cohen's kappa against them (used by ``eval/label_claims.py``).

Only cited claims the judge labelled are sampled (uncited sentences get no verdict). The judge
is compared twice: with its raw label and with the final status after the evidence check
(fabricated evidence -> unsupported).

Flag precision / recall treat the claim check as a detector of bad claims: a claim is
*flagged* when the judge does not label it supported (the UI shows ◐ or ✗), and it is *bad*
when the human label is not supported (partial counts as not supported, as in the
supported-claim rate). Precision = flagged and bad / flagged; recall = flagged and bad / bad.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from research_assistant.abtest.records import EvalRecord
from research_assistant.abtest.stats import accuracy, cohens_kappa
from research_assistant.models import JudgeLabel

LABELS: tuple[JudgeLabel, ...] = ("supported", "partially_supported", "unsupported")
SHORT = {"supported": "S", "partially_supported": "P", "unsupported": "U"}


class ClaimSample(BaseModel):
    """One judged claim of one eval record."""

    key: str  # "<arm>|<question id>|<claim index>"
    run_id: str
    arm: str
    question_id: str
    question: str
    claim_index: int
    sentence: str
    cited: list[int]
    source_chunk_ids: dict[int, str]
    judge_label: JudgeLabel
    final_label: JudgeLabel


class HumanLabel(BaseModel):
    """A hand label (one line of claim_labels.jsonl)."""

    key: str
    human: JudgeLabel
    judge_label: JudgeLabel
    final_label: JudgeLabel
    sentence: str = ""
    labeled_at: str = ""


def candidates(records: Iterable[EvalRecord]) -> list[ClaimSample]:
    """Every judged claim, deduplicated by (question, sentence) across arms, in a stable
    order (by key)."""
    out: dict[tuple[str, str], ClaimSample] = {}
    for rec in records:
        for c in rec.claim_checks:
            if c.judge_label is None or c.status == "uncited":
                continue
            final: JudgeLabel = c.status  # type: ignore[assignment]
            sample = ClaimSample(
                key=f"{rec.arm}|{rec.question_id}|{c.index}",
                run_id=rec.run_id,
                arm=rec.arm,
                question_id=rec.question_id,
                question=rec.question,
                claim_index=c.index,
                sentence=c.sentence,
                cited=c.cited,
                source_chunk_ids={n: rec.sources[n] for n in c.cited if n in rec.sources},
                judge_label=c.judge_label,
                final_label=final,
            )
            dedup = (rec.question_id, " ".join(c.sentence.split()))
            if dedup not in out or sample.key < out[dedup].key:
                out[dedup] = sample
    return sorted(out.values(), key=lambda s: s.key)


def sample_claims(pool: Sequence[ClaimSample], n: int, seed: int) -> list[ClaimSample]:
    """``n`` claims drawn without replacement (all if fewer), reproducible for a seed."""
    if n >= len(pool):
        return list(pool)
    return random.Random(seed).sample(list(pool), n)


# --------------------------------------------------------------------------- label store


def load_labels(path: Path) -> dict[str, HumanLabel]:
    labels: dict[str, HumanLabel] = {}
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            lab = HumanLabel.model_validate_json(line)
        except ValidationError:
            continue
        labels[lab.key] = lab
    return labels


def append_label(path: Path, sample: ClaimSample, human: JudgeLabel) -> HumanLabel:
    lab = HumanLabel(
        key=sample.key,
        human=human,
        judge_label=sample.judge_label,
        final_label=sample.final_label,
        sentence=sample.sentence,
        labeled_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(lab.model_dump_json() + "\n")
    return lab


# --------------------------------------------------------------------------- agreement


class Agreement(BaseModel):
    name: str
    n: int
    accuracy: float | None
    kappa: float | None
    # confusion[human][judge] = count
    confusion: dict[str, dict[str, int]]
    flag_precision: float | None = None
    flag_recall: float | None = None
    n_flagged: int = 0
    n_bad: int = 0


def agreement(
    labels: Sequence[HumanLabel], which: str = "final_label", name: str = ""
) -> Agreement:
    human = [lab.human for lab in labels]
    judge = [getattr(lab, which) for lab in labels]
    confusion = {h: dict.fromkeys(LABELS, 0) for h in LABELS}
    for h, j in zip(human, judge, strict=True):
        confusion[h][j] += 1
    flagged = [j != "supported" for j in judge]
    bad = [h != "supported" for h in human]
    hits = sum(f and b for f, b in zip(flagged, bad, strict=True))
    n_flagged, n_bad = sum(flagged), sum(bad)
    return Agreement(
        name=name or which,
        n=len(labels),
        accuracy=accuracy(human, judge),
        kappa=cohens_kappa(human, judge),
        confusion=confusion,
        flag_precision=hits / n_flagged if n_flagged else None,
        flag_recall=hits / n_bad if n_bad else None,
        n_flagged=n_flagged,
        n_bad=n_bad,
    )


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def render_agreement(results: Sequence[Agreement], run_dir: str, judge_model: str) -> str:
    lines = [
        "# Claim judge vs. hand labels",
        "",
        f"- **Run:** `{run_dir}`",
        f"- **Judge model:** `{judge_model}`",
        f"- **Hand-labelled claims:** {results[0].n if results else 0}",
        "",
        "Labels: supported (S), partially supported (P), unsupported (U). Cohen's kappa "
        "corrects the raw agreement for agreement expected by chance from each rater's label "
        "frequencies (3 classes, unweighted). With about 20 claims both numbers are rough.",
        "",
        "| Judge label | n | Accuracy | Cohen's kappa |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {r.n} | {_fmt(r.accuracy)} | {_fmt(r.kappa)} |")
    lines += [
        "",
        "## Flags (judge: not supported) vs. hand labels (not supported)",
        "",
        "| Judge label | Flagged | Bad (human) | Flag precision | Flag recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.n_flagged} | {r.n_bad} | {_fmt(r.flag_precision)} "
            f"| {_fmt(r.flag_recall)} |"
        )
    for r in results:
        lines += [
            "",
            f"### Confusion matrix: {r.name} (rows: human, columns: judge)",
            "",
            "| human \\ judge | " + " | ".join(SHORT[x] for x in LABELS) + " |",
            "|---|" + "---:|" * len(LABELS),
        ]
        for h in LABELS:
            lines.append(
                f"| {SHORT[h]} | " + " | ".join(str(r.confusion[h][j]) for j in LABELS) + " |"
            )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "LABELS",
    "Agreement",
    "ClaimSample",
    "HumanLabel",
    "agreement",
    "append_label",
    "candidates",
    "load_labels",
    "render_agreement",
    "sample_claims",
]
