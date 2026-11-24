# Retrieval tuning tuning-20261123-102932

- Split: **dev**, 18 in-scope questions with gold papers (out-of-scope questions are skipped)
- Strategies: baseline_dense, hybrid, hybrid_multiquery, hybrid_hyde_rerank, full; grid: top_k=[5, 8, 12], fetch_k=[30, 60], rerank_k=[15, 30], max_chunks_per_paper=[2, 3]
- Configurations: 84; reference: `baseline_dense`
- Rewriter model (query translation, self-query): `anthropic/claude-haiku-4.5`. Production uses `models.judge`; strategies with query translation or filters may behave somewhat differently there.
- Commit `bdae963b1d` (dirty tree), gold sha256 `d411e0ecc0ed`, bootstrap 10000 resamples, seed 0
- Rewriter calls this session: 68 new, 1444 from cache; 62,191 input / 8,692 output tokens

> Dev questions only, and few of them: paired bootstrap CIs are wide, so only large differences are detectable, and a CI that includes 0 means *no detectable difference*. Picking the best of many grid variants on the same questions overfits to them; confirm the chosen configuration once on the test split with eval/run_eval.py.

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
| 1 | `hybrid_multiquery__fetch_k60__max_chunks_per_paper2__top_k12` | ✓ | 0.70 | 0.94 | 0.78 | 0.68 | 6,600 |
| 2 | `full__max_chunks_per_paper2__rerank_k30__top_k12` | ✓ | 0.70 | 1.00 | 0.94 | 0.85 | 5,446 |
| 3 | `full__fetch_k60__max_chunks_per_paper2__rerank_k30__top_k12` |  | 0.70 | 1.00 | 0.94 | 0.83 | 5,529 |
| 4 | `full__fetch_k60__max_chunks_per_paper2__top_k12` |  | 0.69 | 0.94 | 0.94 | 0.92 | 5,851 |
| 5 | `full__rerank_k30__top_k12` |  | 0.67 | 0.94 | 0.94 | 0.84 | 5,655 |
| 6 | `hybrid_hyde_rerank__fetch_k60__max_chunks_per_paper2__top_k12` |  | 0.67 | 0.94 | 0.89 | 0.72 | 5,857 |
| 7 | `hybrid_hyde_rerank__max_chunks_per_paper2__top_k12` |  | 0.66 | 0.94 | 0.89 | 0.68 | 5,728 |
| 8 | `hybrid_hyde_rerank__fetch_k60__max_chunks_per_paper2__rerank_k30__top_k12` |  | 0.66 | 0.94 | 0.83 | 0.65 | 5,615 |
| 9 | `hybrid_multiquery__max_chunks_per_paper2__top_k12` |  | 0.65 | 1.00 | 0.83 | 0.70 | 6,614 |
| 10 | `hybrid__max_chunks_per_paper2__top_k12` |  | 0.65 | 1.00 | 0.83 | 0.67 | 6,477 |
| 11 | `full__fetch_k60__rerank_k30__top_k12` |  | 0.65 | 0.94 | 0.94 | 0.82 | 5,622 |
| 12 | `full__max_chunks_per_paper2__top_k12` |  | 0.65 | 0.94 | 0.89 | 0.84 | 5,651 |
| 13 | `full__fetch_k60__top_k12` |  | 0.65 | 0.94 | 0.94 | 0.92 | 6,070 |
| 14 | `hybrid__fetch_k60__max_chunks_per_paper2__top_k12` |  | 0.65 | 1.00 | 0.83 | 0.68 | 6,440 |
| 15 | `hybrid_hyde_rerank__max_chunks_per_paper2__rerank_k30__top_k12` |  | 0.65 | 0.94 | 0.83 | 0.66 | 5,656 |
| 16 | `hybrid_hyde_rerank__fetch_k60__top_k12` |  | 0.64 | 0.94 | 0.83 | 0.71 | 5,858 |
| 17 | `hybrid_hyde_rerank__fetch_k60__rerank_k30__top_k12` |  | 0.64 | 0.94 | 0.83 | 0.65 | 5,594 |
| 18 | `hybrid_hyde_rerank__top_k12` |  | 0.63 | 0.94 | 0.83 | 0.68 | 5,765 |
| 19 | `baseline_dense__max_chunks_per_paper2__top_k12` |  | 0.62 | 1.00 | 0.78 | 0.57 | 6,009 |
| 20 | `baseline_dense__fetch_k60__max_chunks_per_paper2__top_k12` |  | 0.62 | 1.00 | 0.78 | 0.57 | 6,009 |
| 21 | `full__fetch_k60__max_chunks_per_paper2__rerank_k30` | ✓ | 0.61 | 0.94 | 0.94 | 0.82 | 3,649 |
| 22 | `hybrid_hyde_rerank__rerank_k30__top_k12` |  | 0.61 | 0.94 | 0.83 | 0.65 | 5,652 |
| 23 | `full__top_k12` |  | 0.60 | 0.94 | 0.89 | 0.84 | 5,854 |
| 24 | `hybrid__fetch_k60__top_k12` |  | 0.60 | 1.00 | 0.83 | 0.68 | 6,351 |
| 25 | `full__max_chunks_per_paper2__rerank_k30` |  | 0.60 | 0.94 | 0.94 | 0.84 | 3,686 |
| 26 | `hybrid_multiquery__fetch_k60__top_k12` |  | 0.60 | 0.94 | 0.78 | 0.68 | 6,642 |
| 27 | `hybrid__top_k12` |  | 0.60 | 1.00 | 0.83 | 0.67 | 6,492 |
| 28 | `full__fetch_k60__rerank_k30` |  | 0.59 | 0.94 | 0.94 | 0.82 | 3,734 |
| 29 | `full__fetch_k60__max_chunks_per_paper2` |  | 0.58 | 0.94 | 0.94 | 0.92 | 3,966 |
| 30 | `baseline_dense__top_k12` |  | 0.58 | 1.00 | 0.78 | 0.57 | 5,951 |
| 31 | `baseline_dense__fetch_k60__top_k12` |  | 0.58 | 1.00 | 0.78 | 0.57 | 5,951 |
| 32 | `full__rerank_k30` |  | 0.56 | 0.94 | 0.94 | 0.84 | 3,765 |
| 33 | `hybrid_multiquery__top_k12` |  | 0.56 | 1.00 | 0.83 | 0.70 | 6,507 |
| 34 | `hybrid__fetch_k60__max_chunks_per_paper2` |  | 0.56 | 1.00 | 0.83 | 0.68 | 4,237 |
| 35 | `full__max_chunks_per_paper2` |  | 0.55 | 0.89 | 0.89 | 0.83 | 3,929 |
| 36 | `hybrid_hyde_rerank__fetch_k60__max_chunks_per_paper2` |  | 0.54 | 0.94 | 0.89 | 0.72 | 3,744 |
| 37 | `hybrid_hyde_rerank__max_chunks_per_paper2__rerank_k30` |  | 0.53 | 0.94 | 0.83 | 0.66 | 3,744 |
| 38 | `baseline_dense__max_chunks_per_paper2` |  | 0.53 | 0.94 | 0.78 | 0.57 | 4,140 |
| 39 | `baseline_dense__fetch_k60__max_chunks_per_paper2` |  | 0.53 | 0.94 | 0.78 | 0.57 | 4,140 |
| 40 | `full__fetch_k60` |  | 0.52 | 0.94 | 0.94 | 0.92 | 3,972 |
| 41 | `hybrid_hyde_rerank__max_chunks_per_paper2` |  | 0.52 | 0.94 | 0.89 | 0.68 | 3,696 |
| 42 | `hybrid_hyde_rerank__rerank_k30` |  | 0.52 | 0.94 | 0.83 | 0.65 | 3,716 |
| 43 | `hybrid__fetch_k60` |  | 0.52 | 1.00 | 0.83 | 0.68 | 4,304 |
| 44 | `hybrid_hyde_rerank__fetch_k60__max_chunks_per_paper2__rerank_k30` |  | 0.51 | 0.94 | 0.83 | 0.65 | 3,782 |
| 45 | `hybrid_hyde_rerank__fetch_k60` |  | 0.50 | 0.94 | 0.83 | 0.71 | 3,745 |
| 46 | `hybrid_multiquery__max_chunks_per_paper2` |  | 0.50 | 0.94 | 0.83 | 0.70 | 4,393 |
| 47 | `hybrid_multiquery__fetch_k60__max_chunks_per_paper2` |  | 0.49 | 0.94 | 0.78 | 0.68 | 4,467 |
| 48 | `hybrid_hyde_rerank__fetch_k60__rerank_k30` |  | 0.49 | 0.94 | 0.83 | 0.65 | 3,712 |
| 49 | `hybrid_multiquery` |  | 0.48 | 0.94 | 0.83 | 0.70 | 4,355 |
| 50 | `full__fetch_k60__max_chunks_per_paper2__top_k5` | ✓ | 0.48 | 0.94 | 0.94 | 0.92 | 2,359 |
| 51 | `hybrid_hyde_rerank` |  | 0.48 | 0.94 | 0.83 | 0.68 | 3,718 |
| 52 | `full__max_chunks_per_paper2__rerank_k30__top_k5` | ✓ | 0.48 | 0.94 | 0.94 | 0.84 | 2,300 |
| 53 | `hybrid__max_chunks_per_paper2` |  | 0.47 | 0.94 | 0.83 | 0.67 | 4,318 |
| 54 | `full` |  | 0.47 | 0.89 | 0.89 | 0.83 | 3,968 |
| 55 | `full__rerank_k30__top_k5` | ✓ | 0.47 | 0.94 | 0.94 | 0.84 | 2,280 |
| 56 | `hybrid` |  | 0.46 | 0.94 | 0.83 | 0.67 | 4,355 |
| 57 | `hybrid_multiquery__fetch_k60` |  | 0.46 | 0.94 | 0.78 | 0.68 | 4,414 |
| 58 | `full__fetch_k60__max_chunks_per_paper2__rerank_k30__top_k5` | ✓ | 0.46 | 0.94 | 0.94 | 0.82 | 2,203 |
| 59 | `baseline_dense` |  | 0.46 | 0.94 | 0.78 | 0.57 | 4,061 |
| 60 | `baseline_dense__fetch_k60` |  | 0.46 | 0.94 | 0.78 | 0.57 | 4,061 |
| 61 | `full__fetch_k60__top_k5` |  | 0.45 | 0.94 | 0.94 | 0.92 | 2,463 |
| 62 | `hybrid__fetch_k60__max_chunks_per_paper2__top_k5` |  | 0.43 | 0.94 | 0.83 | 0.67 | 2,629 |
| 63 | `full__fetch_k60__rerank_k30__top_k5` |  | 0.43 | 0.94 | 0.94 | 0.82 | 2,214 |
| 64 | `hybrid_hyde_rerank__max_chunks_per_paper2__rerank_k30__top_k5` |  | 0.42 | 0.94 | 0.83 | 0.66 | 2,339 |
| 65 | `hybrid_hyde_rerank__fetch_k60__max_chunks_per_paper2__rerank_k30__top_k5` |  | 0.42 | 0.94 | 0.83 | 0.65 | 2,279 |
| 66 | `full__max_chunks_per_paper2__top_k5` |  | 0.42 | 0.89 | 0.89 | 0.83 | 2,421 |
| 67 | `baseline_dense__top_k5` |  | 0.41 | 0.94 | 0.78 | 0.57 | 2,554 |
| 68 | `baseline_dense__fetch_k60__top_k5` |  | 0.41 | 0.94 | 0.78 | 0.57 | 2,554 |
| 69 | `baseline_dense__max_chunks_per_paper2__top_k5` |  | 0.41 | 0.94 | 0.78 | 0.57 | 2,598 |
| 70 | `baseline_dense__fetch_k60__max_chunks_per_paper2__top_k5` |  | 0.41 | 0.94 | 0.78 | 0.57 | 2,598 |
| 71 | `full__top_k5` |  | 0.40 | 0.89 | 0.89 | 0.83 | 2,404 |
| 72 | `hybrid_hyde_rerank__fetch_k60__max_chunks_per_paper2__top_k5` |  | 0.40 | 0.94 | 0.89 | 0.72 | 2,254 |
| 73 | `hybrid_multiquery__max_chunks_per_paper2__top_k5` |  | 0.39 | 0.89 | 0.83 | 0.69 | 2,720 |
| 74 | `hybrid_hyde_rerank__rerank_k30__top_k5` |  | 0.39 | 0.94 | 0.83 | 0.65 | 2,393 |
| 75 | `hybrid_hyde_rerank__fetch_k60__rerank_k30__top_k5` |  | 0.39 | 0.94 | 0.83 | 0.65 | 2,320 |
| 76 | `hybrid_hyde_rerank__fetch_k60__top_k5` |  | 0.38 | 0.94 | 0.83 | 0.71 | 2,270 |
| 77 | `hybrid_multiquery__top_k5` |  | 0.37 | 0.89 | 0.83 | 0.69 | 2,744 |
| 78 | `hybrid_multiquery__fetch_k60__max_chunks_per_paper2__top_k5` |  | 0.37 | 0.78 | 0.78 | 0.66 | 2,782 |
| 79 | `hybrid__fetch_k60__top_k5` |  | 0.37 | 0.89 | 0.83 | 0.66 | 2,680 |
| 80 | `hybrid__max_chunks_per_paper2__top_k5` |  | 0.37 | 0.89 | 0.83 | 0.66 | 2,728 |
| 81 | `hybrid_hyde_rerank__max_chunks_per_paper2__top_k5` |  | 0.37 | 0.94 | 0.89 | 0.68 | 2,252 |
| 82 | `hybrid_hyde_rerank__top_k5` |  | 0.34 | 0.94 | 0.83 | 0.68 | 2,294 |
| 83 | `hybrid__top_k5` |  | 0.34 | 0.83 | 0.83 | 0.65 | 2,781 |
| 84 | `hybrid_multiquery__fetch_k60__top_k5` |  | 0.32 | 0.78 | 0.78 | 0.66 | 2,836 |

## Difference to `baseline_dense` (paired bootstrap, 95% CI)

| Configuration | n | Δ Gold found | Δ MRR | Δ Context tok |
|---|---|---|---|---|
| `hybrid_multiquery__fetch_k60__max_chunks_per_paper2__top_k12` | 18 | +0.24 [0.14, 0.33] | +0.12 [-0.01, 0.26] | +2,540 [2,184, 2,880] |
| `full__max_chunks_per_paper2__rerank_k30__top_k12` | 18 | +0.24 [0.10, 0.37] | +0.28 [0.11, 0.46] | +1,385 [623, 1,962] |
| `full__fetch_k60__max_chunks_per_paper2__rerank_k30__top_k12` | 18 | +0.24 [0.10, 0.37] | +0.26 [0.11, 0.42] | +1,468 [696, 2,057] |
| `full__fetch_k60__max_chunks_per_paper2__top_k12` | 18 | +0.24 [0.10, 0.35] | +0.35 [0.18, 0.51] | +1,790 [952, 2,462] |
| `full__rerank_k30__top_k12` | 18 | +0.21 [0.06, 0.35] | +0.28 [0.10, 0.45] | +1,595 [949, 2,108] |
| `hybrid_hyde_rerank__fetch_k60__max_chunks_per_paper2__top_k12` | 18 | +0.21 [0.14, 0.29] | +0.15 [-0.03, 0.32] | +1,796 [1,467, 2,123] |
| `hybrid_hyde_rerank__max_chunks_per_paper2__top_k12` | 18 | +0.20 [0.13, 0.28] | +0.11 [-0.06, 0.28] | +1,667 [1,278, 2,058] |
| `hybrid_hyde_rerank__fetch_k60__max_chunks_per_paper2__rerank_k30__top_k12` | 18 | +0.20 [0.13, 0.27] | +0.08 [-0.11, 0.26] | +1,554 [1,193, 1,910] |
| `hybrid_multiquery__max_chunks_per_paper2__top_k12` | 18 | +0.20 [0.11, 0.28] | +0.14 [0.02, 0.27] | +2,553 [2,284, 2,851] |
| `hybrid__max_chunks_per_paper2__top_k12` | 18 | +0.19 [0.13, 0.26] | +0.11 [-0.02, 0.24] | +2,416 [2,176, 2,663] |
| `full__fetch_k60__rerank_k30__top_k12` | 18 | +0.19 [0.04, 0.34] | +0.26 [0.10, 0.42] | +1,561 [928, 2,065] |
| `full__max_chunks_per_paper2__top_k12` | 18 | +0.19 [0.05, 0.33] | +0.27 [0.11, 0.44] | +1,591 [737, 2,342] |
| `full__fetch_k60__top_k12` | 18 | +0.19 [0.05, 0.32] | +0.35 [0.18, 0.51] | +2,009 [1,263, 2,600] |
| `hybrid__fetch_k60__max_chunks_per_paper2__top_k12` | 18 | +0.19 [0.11, 0.27] | +0.12 [-0.01, 0.24] | +2,379 [2,018, 2,714] |
| `hybrid_hyde_rerank__max_chunks_per_paper2__rerank_k30__top_k12` | 18 | +0.19 [0.11, 0.27] | +0.09 [-0.10, 0.27] | +1,595 [1,248, 1,959] |
| `hybrid_hyde_rerank__fetch_k60__top_k12` | 18 | +0.18 [0.10, 0.27] | +0.15 [-0.03, 0.32] | +1,797 [1,489, 2,112] |
| `hybrid_hyde_rerank__fetch_k60__rerank_k30__top_k12` | 18 | +0.18 [0.11, 0.25] | +0.08 [-0.11, 0.26] | +1,533 [1,148, 1,942] |
| `hybrid_hyde_rerank__top_k12` | 18 | +0.17 [0.10, 0.25] | +0.11 [-0.06, 0.28] | +1,704 [1,293, 2,121] |
| `baseline_dense__max_chunks_per_paper2__top_k12` | 18 | +0.16 [0.09, 0.23] | +0.01 [0.00, 0.02] | +1,948 [1,740, 2,168] |
| `baseline_dense__fetch_k60__max_chunks_per_paper2__top_k12` | 18 | +0.16 [0.09, 0.23] | +0.01 [0.00, 0.02] | +1,948 [1,740, 2,168] |
| `full__fetch_k60__max_chunks_per_paper2__rerank_k30` | 18 | +0.15 [-0.00, 0.31] | +0.26 [0.10, 0.42] | -412 [-915, 37] |
| `hybrid_hyde_rerank__rerank_k30__top_k12` | 18 | +0.15 [0.04, 0.25] | +0.09 [-0.10, 0.27] | +1,591 [1,251, 1,965] |
| `full__top_k12` | 18 | +0.15 [0.01, 0.28] | +0.27 [0.11, 0.44] | +1,793 [1,047, 2,443] |
| `hybrid__fetch_k60__top_k12` | 18 | +0.14 [0.07, 0.22] | +0.11 [-0.01, 0.24] | +2,290 [1,961, 2,599] |
| `full__max_chunks_per_paper2__rerank_k30` | 18 | +0.14 [-0.02, 0.30] | +0.28 [0.10, 0.45] | -375 [-861, 47] |
| `hybrid_multiquery__fetch_k60__top_k12` | 18 | +0.14 [0.01, 0.26] | +0.12 [-0.01, 0.26] | +2,581 [2,253, 2,889] |
| `hybrid__top_k12` | 18 | +0.14 [0.06, 0.22] | +0.11 [-0.02, 0.23] | +2,432 [2,249, 2,621] |
| `full__fetch_k60__rerank_k30` | 18 | +0.13 [-0.03, 0.29] | +0.26 [0.10, 0.42] | -327 [-744, 75] |
| `full__fetch_k60__max_chunks_per_paper2` | 18 | +0.13 [-0.01, 0.26] | +0.35 [0.18, 0.51] | -94 [-682, 409] |
| `baseline_dense__top_k12` | 18 | +0.13 [0.07, 0.18] | +0.01 [0.00, 0.02] | +1,890 [1,676, 2,125] |
| `baseline_dense__fetch_k60__top_k12` | 18 | +0.13 [0.07, 0.18] | +0.01 [0.00, 0.02] | +1,890 [1,676, 2,125] |
| `full__rerank_k30` | 18 | +0.11 [-0.05, 0.26] | +0.28 [0.10, 0.45] | -296 [-685, 43] |
| `hybrid_multiquery__top_k12` | 18 | +0.11 [-0.01, 0.21] | +0.14 [0.02, 0.27] | +2,446 [2,171, 2,719] |
| `hybrid__fetch_k60__max_chunks_per_paper2` | 18 | +0.11 [0.02, 0.19] | +0.12 [-0.01, 0.24] | +176 [-95, 471] |
| `full__max_chunks_per_paper2` | 18 | +0.09 [-0.05, 0.23] | +0.27 [0.10, 0.44] | -132 [-731, 434] |
| `hybrid_hyde_rerank__fetch_k60__max_chunks_per_paper2` | 18 | +0.08 [-0.03, 0.18] | +0.15 [-0.03, 0.32] | -317 [-580, -38] |
| `hybrid_hyde_rerank__max_chunks_per_paper2__rerank_k30` | 18 | +0.07 [-0.05, 0.18] | +0.09 [-0.10, 0.27] | -317 [-564, -64] |
| `baseline_dense__max_chunks_per_paper2` | 18 | +0.07 [0.02, 0.12] | +0.00 [0.00, 0.00] | +79 [-40, 203] |
| `baseline_dense__fetch_k60__max_chunks_per_paper2` | 18 | +0.07 [0.02, 0.12] | +0.00 [0.00, 0.00] | +79 [-40, 203] |
| `full__fetch_k60` | 18 | +0.07 [-0.05, 0.18] | +0.35 [0.18, 0.51] | -88 [-567, 350] |
| `hybrid_hyde_rerank__max_chunks_per_paper2` | 18 | +0.06 [-0.05, 0.17] | +0.11 [-0.06, 0.28] | -365 [-719, -7] |
| `hybrid_hyde_rerank__rerank_k30` | 18 | +0.06 [-0.05, 0.17] | +0.09 [-0.10, 0.27] | -345 [-616, -63] |
| `hybrid__fetch_k60` | 18 | +0.06 [-0.02, 0.14] | +0.11 [-0.01, 0.24] | +243 [-54, 552] |
| `hybrid_hyde_rerank__fetch_k60__max_chunks_per_paper2__rerank_k30` | 18 | +0.05 [-0.06, 0.16] | +0.08 [-0.11, 0.26] | -278 [-568, 25] |
| `hybrid_hyde_rerank__fetch_k60` | 18 | +0.04 [-0.05, 0.13] | +0.15 [-0.03, 0.32] | -316 [-616, -13] |
| `hybrid_multiquery__max_chunks_per_paper2` | 18 | +0.04 [-0.08, 0.15] | +0.13 [0.01, 0.27] | +332 [-35, 709] |
| `hybrid_multiquery__fetch_k60__max_chunks_per_paper2` | 18 | +0.04 [-0.07, 0.14] | +0.12 [-0.01, 0.26] | +406 [132, 708] |
| `hybrid_hyde_rerank__fetch_k60__rerank_k30` | 18 | +0.03 [-0.08, 0.14] | +0.08 [-0.11, 0.26] | -349 [-635, -57] |
| `hybrid_multiquery` | 18 | +0.02 [-0.08, 0.12] | +0.13 [0.01, 0.27] | +294 [-43, 644] |
| `full__fetch_k60__max_chunks_per_paper2__top_k5` | 18 | +0.02 [-0.10, 0.14] | +0.35 [0.18, 0.51] | -1,702 [-2,104, -1,292] |
| `hybrid_hyde_rerank` | 18 | +0.02 [-0.09, 0.13] | +0.11 [-0.06, 0.28] | -343 [-659, -45] |
| `full__max_chunks_per_paper2__rerank_k30__top_k5` | 18 | +0.02 [-0.12, 0.14] | +0.28 [0.10, 0.45] | -1,761 [-2,135, -1,380] |
| `hybrid__max_chunks_per_paper2` | 18 | +0.02 [-0.07, 0.10] | +0.10 [-0.03, 0.23] | +257 [43, 480] |
| `full` | 18 | +0.01 [-0.11, 0.13] | +0.27 [0.10, 0.44] | -93 [-581, 367] |
| `full__rerank_k30__top_k5` | 18 | +0.01 [-0.12, 0.13] | +0.28 [0.10, 0.45] | -1,781 [-2,089, -1,444] |
| `hybrid` | 18 | +0.00 [-0.07, 0.07] | +0.10 [-0.03, 0.23] | +294 [38, 574] |
| `hybrid_multiquery__fetch_k60` | 18 | +0.00 [-0.10, 0.09] | +0.12 [-0.01, 0.26] | +354 [120, 611] |
| `full__fetch_k60__max_chunks_per_paper2__rerank_k30__top_k5` | 18 | +0.00 [-0.13, 0.12] | +0.26 [0.10, 0.42] | -1,858 [-2,256, -1,446] |
| `baseline_dense__fetch_k60` | 18 | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +0 [0, 0] |
| `full__fetch_k60__top_k5` | 18 | -0.01 [-0.13, 0.11] | +0.35 [0.18, 0.51] | -1,598 [-1,950, -1,223] |
| `hybrid__fetch_k60__max_chunks_per_paper2__top_k5` | 18 | -0.03 [-0.12, 0.06] | +0.11 [-0.02, 0.23] | -1,432 [-1,732, -1,113] |
| `full__fetch_k60__rerank_k30__top_k5` | 18 | -0.03 [-0.15, 0.08] | +0.26 [0.10, 0.42] | -1,847 [-2,176, -1,483] |
| `hybrid_hyde_rerank__max_chunks_per_paper2__rerank_k30__top_k5` | 18 | -0.03 [-0.13, 0.06] | +0.09 [-0.10, 0.27] | -1,722 [-2,009, -1,420] |
| `hybrid_hyde_rerank__fetch_k60__max_chunks_per_paper2__rerank_k30__top_k5` | 18 | -0.03 [-0.13, 0.06] | +0.08 [-0.11, 0.26] | -1,781 [-2,069, -1,476] |
| `full__max_chunks_per_paper2__top_k5` | 18 | -0.04 [-0.16, 0.08] | +0.27 [0.10, 0.44] | -1,640 [-2,056, -1,209] |
| `baseline_dense__top_k5` | 18 | -0.05 [-0.09, -0.02] | +0.00 [0.00, 0.00] | -1,507 [-1,675, -1,325] |
| `baseline_dense__fetch_k60__top_k5` | 18 | -0.05 [-0.09, -0.02] | +0.00 [0.00, 0.00] | -1,507 [-1,675, -1,325] |
| `baseline_dense__max_chunks_per_paper2__top_k5` | 18 | -0.05 [-0.09, -0.02] | +0.00 [0.00, 0.00] | -1,463 [-1,628, -1,287] |
| `baseline_dense__fetch_k60__max_chunks_per_paper2__top_k5` | 18 | -0.05 [-0.09, -0.02] | +0.00 [0.00, 0.00] | -1,463 [-1,628, -1,287] |
| `full__top_k5` | 18 | -0.06 [-0.18, 0.06] | +0.27 [0.10, 0.44] | -1,657 [-2,006, -1,268] |
| `hybrid_hyde_rerank__fetch_k60__max_chunks_per_paper2__top_k5` | 18 | -0.06 [-0.16, 0.03] | +0.15 [-0.03, 0.32] | -1,806 [-2,049, -1,557] |
| `hybrid_multiquery__max_chunks_per_paper2__top_k5` | 18 | -0.06 [-0.16, 0.03] | +0.12 [0.00, 0.26] | -1,340 [-1,670, -990] |
| `hybrid_hyde_rerank__rerank_k30__top_k5` | 18 | -0.07 [-0.16, 0.02] | +0.09 [-0.10, 0.27] | -1,668 [-1,943, -1,380] |
| `hybrid_hyde_rerank__fetch_k60__rerank_k30__top_k5` | 18 | -0.07 [-0.16, 0.02] | +0.08 [-0.11, 0.26] | -1,740 [-2,024, -1,439] |
| `hybrid_hyde_rerank__fetch_k60__top_k5` | 18 | -0.08 [-0.18, 0.02] | +0.15 [-0.03, 0.32] | -1,791 [-2,046, -1,534] |
| `hybrid_multiquery__top_k5` | 18 | -0.08 [-0.18, 0.01] | +0.12 [0.00, 0.26] | -1,317 [-1,619, -994] |
| `hybrid_multiquery__fetch_k60__max_chunks_per_paper2__top_k5` | 18 | -0.09 [-0.19, 0.01] | +0.09 [-0.06, 0.24] | -1,279 [-1,593, -940] |
| `hybrid__fetch_k60__top_k5` | 18 | -0.09 [-0.17, -0.02] | +0.10 [-0.03, 0.23] | -1,381 [-1,672, -1,073] |
| `hybrid__max_chunks_per_paper2__top_k5` | 18 | -0.09 [-0.17, -0.02] | +0.09 [-0.04, 0.22] | -1,332 [-1,630, -1,013] |
| `hybrid_hyde_rerank__max_chunks_per_paper2__top_k5` | 18 | -0.09 [-0.19, 0.00] | +0.11 [-0.06, 0.28] | -1,809 [-2,105, -1,516] |
| `hybrid_hyde_rerank__top_k5` | 18 | -0.11 [-0.21, -0.02] | +0.11 [-0.06, 0.28] | -1,767 [-2,076, -1,464] |
| `hybrid__top_k5` | 18 | -0.12 [-0.20, -0.04] | +0.08 [-0.05, 0.22] | -1,279 [-1,559, -974] |
| `hybrid_multiquery__fetch_k60__top_k5` | 18 | -0.13 [-0.22, -0.05] | +0.09 [-0.06, 0.24] | -1,225 [-1,504, -925] |
