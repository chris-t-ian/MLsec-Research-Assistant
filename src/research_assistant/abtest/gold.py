"""Gold question set: schema, loading, topic derivation and the one-time dev/test split.

The gold file (``eval/gold_questions.yaml``) is written by hand. This module never generates
questions, gold papers or key facts; it only reads them and, once, adds ``topic`` and
``split`` fields (see :func:`write_splits`).
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

QuestionType = Literal["factual", "comparison", "method", "out_of_scope"]
Split = Literal["dev", "test"]

OUT_OF_SCOPE: QuestionType = "out_of_scope"
OUT_OF_SCOPE_TOPIC = "out_of_scope"
UNKNOWN_TOPIC = "other"


class GoldFileError(ValueError):
    """The gold file is missing, malformed or would be changed in a forbidden way."""


class GoldQuestion(BaseModel):
    """One hand-written evaluation question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    type: QuestionType
    gold_arxiv_ids: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(
        default_factory=list, description="Short statements a good answer should contain."
    )
    notes: str = ""
    topic: str | None = Field(
        default=None, description="Stratum for the split; derived by scripts/split_gold.py."
    )
    split: Split | None = Field(default=None, description="Assigned once by split_gold.py.")

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v: Any) -> Any:
        return str(v) if isinstance(v, int) and not isinstance(v, bool) else v

    @field_validator("gold_arxiv_ids", mode="before")
    @classmethod
    def _ids_are_strings(cls, v: Any) -> Any:
        if isinstance(v, list):
            for item in v:
                if not isinstance(item, str):
                    raise ValueError(
                        f"arXiv ID {item!r} is not a string; quote it in the YAML "
                        "(unquoted IDs such as 2212.06470 lose their trailing zero)"
                    )
        return v

    @property
    def in_scope(self) -> bool:
        return self.type != OUT_OF_SCOPE


def _validate_set(questions: Sequence[GoldQuestion]) -> None:
    dupes = [i for i, n in Counter(q.id for q in questions).items() if n > 1]
    if dupes:
        raise GoldFileError(f"duplicate question ids: {', '.join(sorted(dupes))}")
    for q in questions:
        if not q.key_facts:
            raise GoldFileError(f"question {q.id}: key_facts is empty")
        if q.in_scope and not q.gold_arxiv_ids:
            raise GoldFileError(f"question {q.id}: in-scope questions need gold_arxiv_ids")


def parse_gold(data: Any) -> list[GoldQuestion]:
    """Validate parsed YAML (a list of mappings) as a gold question set."""
    if not isinstance(data, list):
        raise GoldFileError("the gold file must contain a YAML list of questions")
    questions: list[GoldQuestion] = []
    for n, item in enumerate(data, start=1):
        try:
            questions.append(GoldQuestion.model_validate(item))
        except ValidationError as exc:
            raise GoldFileError(f"entry {n} is invalid: {exc}") from exc
    _validate_set(questions)
    return questions


def load_gold(path: Path) -> list[GoldQuestion]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GoldFileError(f"cannot read {path}: {exc.strerror}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise GoldFileError(f"{path} is not valid YAML") from exc
    return parse_gold(data)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- topics


def load_paper_topics(papers_file: Path) -> dict[str, str]:
    """``arxiv_id -> topic`` from data/papers.yaml."""
    try:
        data = yaml.safe_load(papers_file.read_text(encoding="utf-8")) or []
    except (OSError, yaml.YAMLError) as exc:
        raise GoldFileError(f"cannot read paper topics from {papers_file}") from exc
    return {
        str(e["arxiv_id"]): str(e["topic"])
        for e in data
        if isinstance(e, dict) and "arxiv_id" in e and "topic" in e
    }


def derive_topic(question: GoldQuestion, paper_topics: Mapping[str, str]) -> str:
    """Majority topic of the question's gold papers.

    Out-of-scope questions form their own stratum. Ties go to the topic that appears first in
    ``gold_arxiv_ids`` order; questions without any known gold paper get ``"other"``.
    """
    if not question.in_scope:
        return OUT_OF_SCOPE_TOPIC
    topics = [paper_topics[a] for a in question.gold_arxiv_ids if a in paper_topics]
    if not topics:
        return UNKNOWN_TOPIC
    counts = Counter(topics)
    best = max(counts.values())
    return next(t for t in topics if counts[t] == best)


# --------------------------------------------------------------------------- splits


def assign_splits(strata: Mapping[str, tuple[str, str]], n_dev: int, seed: int) -> dict[str, Split]:
    """Stratified dev/test assignment.

    ``strata`` maps question id -> (type, topic). Each stratum gets a dev quota proportional
    to its size (largest-remainder rounding, ties broken by the seeded RNG), then its members
    are shuffled with the seeded RNG and the first ``quota`` become dev. Deterministic for a
    given input and seed.
    """
    total = len(strata)
    if not 0 <= n_dev <= total:
        raise GoldFileError(f"cannot put {n_dev} of {total} questions into dev")
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for qid, key in strata.items():
        groups[key].append(qid)
    keys = sorted(groups)
    exact = {k: n_dev * len(groups[k]) / total if total else 0.0 for k in keys}
    quota = {k: int(exact[k]) for k in keys}
    tiebreak = {k: rng.random() for k in keys}
    remaining = n_dev - sum(quota.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - quota[k]), tiebreak[k]))[:remaining]:
        quota[k] += 1
    splits: dict[str, Split] = {}
    for k in keys:
        members = sorted(groups[k], key=_natural_key)
        rng.shuffle(members)
        for i, qid in enumerate(members):
            splits[qid] = "dev" if i < quota[k] else "test"
    return splits


def _natural_key(qid: str) -> tuple[int, int | str]:
    return (0, int(qid)) if qid.isdigit() else (1, qid)


_ID_LINE = re.compile(r"^- id:\s*(?P<id>.+?)\s*$")
_TYPE_LINE = re.compile(r"^  type:\s*\S")
_TOPIC_LINE = re.compile(r"^  topic:")


def _yaml_scalar(raw: str) -> str:
    value = yaml.safe_load(raw)
    return str(value)


def insert_fields(text: str, splits: Mapping[str, Split], topics: Mapping[str, str]) -> str:
    """Add ``topic`` (if absent) and ``split`` lines after each question's ``type`` line.

    Works on the raw text so comments and formatting of the hand-written file survive.
    Expects the layout of the gold file: ``- id: ...`` items with two-space-indented keys.
    """
    lines = text.splitlines(keepends=True)
    blocks: list[tuple[str, int, int]] = []  # id, start line, end line (exclusive)
    current: tuple[str, int] | None = None
    for i, line in enumerate(lines):
        m = _ID_LINE.match(line)
        if m:
            if current:
                blocks.append((current[0], current[1], i))
            current = (_yaml_scalar(m.group("id")), i)
    if current:
        blocks.append((current[0], current[1], len(lines)))

    out = list(lines)
    for qid, start, end in reversed(blocks):
        if qid not in splits:
            raise GoldFileError(f"no split computed for question {qid}")
        type_idx = next((i for i in range(start, end) if _TYPE_LINE.match(lines[i])), None)
        if type_idx is None:
            raise GoldFileError(f"question {qid}: no 'type:' line in the expected layout")
        has_topic = any(_TOPIC_LINE.match(lines[i]) for i in range(start, end))
        new = [] if has_topic else [f"  topic: {topics[qid]}\n"]
        new.append(f"  split: {splits[qid]}\n")
        if not out[type_idx].endswith("\n"):
            out[type_idx] += "\n"
        out[type_idx + 1 : type_idx + 1] = new
    return "".join(out)


class SplitReport(BaseModel):
    n_dev: int
    n_test: int
    seed: int
    by_stratum: dict[str, dict[str, int]]  # "type/topic" -> {"dev": n, "test": n}


def write_splits(
    gold_file: Path, papers_file: Path, n_dev: int, seed: int, *, dry_run: bool = False
) -> SplitReport:
    """Assign splits once and write them into the gold file.

    Refuses (GoldFileError) if any question already has a split. The rewritten file is
    re-parsed and checked to differ from the original only in ``topic``/``split``.
    """
    questions = load_gold(gold_file)
    already = [q.id for q in questions if q.split is not None]
    if already:
        raise GoldFileError(
            f"{len(already)} question(s) already have a split; splits are assigned only once "
            "and are not overwritten"
        )
    paper_topics = load_paper_topics(papers_file)
    topics = {q.id: q.topic or derive_topic(q, paper_topics) for q in questions}
    strata = {q.id: (q.type, topics[q.id]) for q in questions}
    splits = assign_splits(strata, n_dev, seed)

    original = gold_file.read_text(encoding="utf-8")
    updated = insert_fields(original, splits, topics)
    check = parse_gold(yaml.safe_load(updated))
    expected = [
        q.model_copy(update={"topic": topics[q.id], "split": splits[q.id]}) for q in questions
    ]
    if check != expected:
        raise GoldFileError("rewriting the gold file would change more than topic/split")
    if not dry_run:
        tmp = gold_file.with_suffix(gold_file.suffix + ".tmp")
        tmp.write_text(updated, encoding="utf-8")
        tmp.replace(gold_file)

    by: dict[str, dict[str, int]] = defaultdict(lambda: {"dev": 0, "test": 0})
    for qid, (qtype, topic) in strata.items():
        by[f"{qtype}/{topic}"][splits[qid]] += 1
    return SplitReport(
        n_dev=sum(1 for s in splits.values() if s == "dev"),
        n_test=sum(1 for s in splits.values() if s == "test"),
        seed=seed,
        by_stratum=dict(sorted(by.items())),
    )


__all__ = [
    "GoldFileError",
    "GoldQuestion",
    "QuestionType",
    "Split",
    "SplitReport",
    "assign_splits",
    "derive_topic",
    "file_sha256",
    "insert_fields",
    "load_gold",
    "load_paper_topics",
    "parse_gold",
    "write_splits",
]
