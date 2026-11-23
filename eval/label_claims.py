"""Hand-label claims of dev-set answers and measure how well the claim judge agrees.

Usage:
    uv run python eval/label_claims.py eval/results/<run>            # label 20 claims
    uv run python eval/label_claims.py eval/results/<run> --n 30 --seed 1
    uv run python eval/label_claims.py eval/results/<run> --report    # agreement only

Claims are sampled (seeded) from the judged, cited claims of the run's answers; uncited
sentences have no judge verdict and are not sampled. For each claim the terminal shows the
question, the sentence and the full text of the chunks it cites, but not the judge's label.
Answer s (supported), p (partially supported), u (unsupported), k (skip) or q (quit). Labels
are appended to <run>/claim_labels.jsonl right away, so you can stop and continue later. The
end prints accuracy and Cohen's kappa of the judge against your labels and writes
<run>/judge_agreement.md.

The run must be a dev-split run whose records carry claim checks (eval/run_eval.py after
step 14). Paper text is printed as plain text with control characters removed.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

from research_assistant.abtest.runner import RESULTS_FILE, EvalRunError, ResultStore, load_manifest
from research_assistant.config import ConfigError, get_settings
from research_assistant.ingest.index import ChunkIndex, ChunkIndexError
from research_assistant.models import JudgeLabel
from research_assistant.verification.agreement import (
    ClaimSample,
    agreement,
    append_label,
    candidates,
    load_labels,
    render_agreement,
    sample_claims,
)
from research_assistant.verification.claims import strip_markers

LABELS_FILE = "claim_labels.jsonl"
REPORT_FILE = "judge_agreement.md"
KEYS: dict[str, JudgeLabel] = {"s": "supported", "p": "partially_supported", "u": "unsupported"}
# C0/C1 control characters incl. ESC: untrusted paper text must not drive the terminal.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
MAX_CHUNK_CHARS = 4000


def plain(text: str) -> str:
    return _CONTROL.sub(" ", text)


def wrap(text: str, indent: str = "    ") -> str:
    paras = [p for p in plain(text).split("\n") if p.strip()]
    return "\n".join(
        textwrap.fill(
            " ".join(p.split()), width=96, initial_indent=indent, subsequent_indent=indent
        )
        for p in paras
    )


def show(sample: ClaimSample, index: ChunkIndex | None, i: int, total: int) -> None:
    print("\n" + "=" * 100)
    print(f"Claim {i}/{total}   (question {sample.question_id}, {sample.arm})")
    print(f"\nQuestion:\n{wrap(sample.question)}")
    print(f"\nClaim (cites {', '.join(f'[{n}]' for n in sample.cited)}):")
    print(wrap(strip_markers(sample.sentence)))
    for n in sample.cited:
        chunk_id = sample.source_chunk_ids.get(n)
        got = index.get(chunk_id) if index is not None and chunk_id else None
        print(f"\n--- Source [{n}] {chunk_id or '(unknown chunk)'}")
        if got is None:
            print("    (chunk text not available)")
            continue
        chunk = got[0]
        print(f"    {plain(chunk.title)} ({chunk.year}), section: {plain(chunk.section or '-')}")
        text = chunk.text
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + " …[truncated]"
        print(wrap(text))


def ask() -> str:
    while True:
        try:
            answer = input("\n[s]upported / [p]artially / [u]nsupported / [k] skip / [q] quit: ")
        except EOFError:
            return "q"
        answer = answer.strip().lower()
        if answer in (*KEYS, "k", "q"):
            return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="eval/results/<run>")
    parser.add_argument("--n", type=int, default=20, help="claims to label (default 20)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--report", action="store_true", help="only compute the agreement")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.run_dir)
        if manifest.split != "dev":
            raise EvalRunError("label claims from a dev-split run (keep the test split unseen)")
        records = list(ResultStore(args.run_dir / RESULTS_FILE).load().values())
        pool = candidates(records)
        if not pool:
            raise EvalRunError(
                "no judged claims in this run; run eval/run_eval.py first (claim checks are "
                "stored since step 14)"
            )
        samples = sample_claims(pool, args.n, args.seed)
        labels_path = args.run_dir / LABELS_FILE
        labels = load_labels(labels_path)
        todo = [s for s in samples if s.key not in labels]
        if todo and not args.report:
            settings = get_settings()
            try:
                index: ChunkIndex | None = ChunkIndex.from_config(
                    settings.paths.resolve().index_dir, settings.retrieval
                )
            except ChunkIndexError as exc:
                print(f"Warning: index unavailable ({exc}); chunk text not shown.", file=sys.stderr)
                index = None
            print(f"{len(samples)} claims sampled, {len(samples) - len(todo)} already labelled.")
            for i, sample in enumerate(samples, start=1):
                if sample.key in labels:
                    continue
                show(sample, index, i, len(samples))
                key = ask()
                if key == "q":
                    break
                if key == "k":
                    continue
                labels[sample.key] = append_label(labels_path, sample, KEYS[key])
    except (ConfigError, EvalRunError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped. Labels so far are saved; run again to continue.", file=sys.stderr)
        return 130

    wanted = {s.key for s in samples}
    done = [lab for key, lab in labels.items() if key in wanted]
    if not done:
        print("No labels yet.", file=sys.stderr)
        return 1
    results = [
        agreement(done, "judge_label", "raw judge label"),
        agreement(done, "final_label", "after evidence check"),
    ]
    report = render_agreement(results, str(args.run_dir), manifest.judge_model)
    (args.run_dir / REPORT_FILE).write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Written to {args.run_dir / REPORT_FILE} ({len(done)}/{len(samples)} labelled).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
