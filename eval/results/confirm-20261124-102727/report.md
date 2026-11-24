# Retrieval confirmation confirm-20261124-102727

- Split: **test**, 39 in-scope questions with gold papers (out-of-scope questions are skipped)
- Strategies: baseline_dense, baseline_dense_tuned; grid: none
- Configurations: 2; reference: `baseline_dense`
- Rewriter model (query translation, self-query): `anthropic/claude-haiku-4.5`. Production uses `models.judge`; strategies with query translation or filters may behave somewhat differently there.
- Commit `714b4da810` (dirty tree), gold sha256 `d411e0ecc0ed`, bootstrap 10000 resamples, seed 0
- Rewriter calls this session: 0 new, 0 from cache; 0 input / 0 output tokens

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
| 1 | `baseline_dense_tuned` | ✓ | 0.58 | 0.95 | 0.87 | 0.78 | 5,866 |
| 2 | `baseline_dense` | ✓ | 0.49 | 0.95 | 0.87 | 0.78 | 3,848 |

## Difference to `baseline_dense` (paired bootstrap, 95% CI)

| Configuration | n | Δ Gold found | Δ MRR | Δ Context tok |
|---|---|---|---|---|
| `baseline_dense_tuned` | 39 | +0.09 [0.05, 0.12] | +0.00 [0.00, 0.00] | +2,018 [1,819, 2,201] |
