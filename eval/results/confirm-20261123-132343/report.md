# Retrieval confirmation confirm-20261123-132343

- Split: **test**, 39 in-scope questions with gold papers (out-of-scope questions are skipped)
- Strategies: baseline_dense, full, full_tuned; grid: none
- Configurations: 3; reference: `baseline_dense`
- Rewriter model (query translation, self-query): `anthropic/claude-haiku-4.5`. Production uses `models.judge`; strategies with query translation or filters may behave somewhat differently there.
- Commit `bdae963b1d` (dirty tree), gold sha256 `d411e0ecc0ed`, bootstrap 10000 resamples, seed 0
- Rewriter calls this session: 78 new, 78 from cache; 78,698 input / 6,644 output tokens

> Test split, fixed configurations chosen on dev; no parameter was tuned on these questions, so the differences are an unbiased check of the dev choice. A CI that includes 0 means *no detectable difference*, not *no difference*.

## Metrics

Paper level: a chunk counts as a hit if it comes from one of the question's gold papers.

- **Gold papers found**: share of the question's gold papers among the selected chunks (matters for comparison questions with several gold papers). Primary sort key.
- **Recall (any)**: 1 if any gold paper is among the selected chunks.
- **Recall@3**: same, within the top 3 chunks.
- **MRR**: 1 / rank of the first gold chunk (0 if none).
- **Context tokens**: tokens of the selected chunk texts, i.e. what one search adds to the answer model's prompt (the cost side of top_k).
- Latency is not compared: rewrites and reranker scores are cached across grid variants, so only the first variant of a strategy pays for them.
- **Pareto**: no other configuration finds at least as many gold papers with fewer context tokens.

## Ranking

| # | Configuration | Pareto | Gold found | Recall (any) | Recall@3 | MRR | Context tok |
|---|---|---|---|---|---|---|---|
| 1 | `full_tuned` | ✓ | 0.53 | 0.92 | 0.85 | 0.75 | 5,257 |
| 2 | `baseline_dense` | ✓ | 0.46 | 0.95 | 0.87 | 0.78 | 3,856 |
| 3 | `full` | ✓ | 0.43 | 0.90 | 0.87 | 0.77 | 3,698 |

## Difference to `baseline_dense` (paired bootstrap, 95% CI)

| Configuration | n | Δ Gold found | Δ MRR | Δ Context tok |
|---|---|---|---|---|
| `full_tuned` | 39 | +0.07 [-0.04, 0.17] | -0.03 [-0.18, 0.13] | +1,401 [755, 1,997] |
| `full` | 39 | -0.03 [-0.12, 0.05] | -0.01 [-0.16, 0.14] | -158 [-542, 185] |
