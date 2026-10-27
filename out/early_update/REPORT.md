# Next step 3: Bayesian update with early 17lands data

2026-10-27. Script: `scripts/early_update.py` (defaults). Prior: the Phase 2
ensemble avg(FT EmbeddingGemma-2, gbt_tfidf), with the FT model's predicted SD.
Data: PremierDraft games from the first D days after release (day one = first
day with >= 10% of the median daily volume). **Target: labels refitted on games
after day 7 only**, so no estimate is scored on the games it used. Within-set
metrics, mean over the 7 test sets (`metrics_summary.csv`, subset `label_late`;
`label_full` = official full-data label, same ordering).

| estimate | day 1 | day 2 | day 3 | day 5 | day 7 |
|---|---|---|---|---|---|
| prior (model only, pre-release) | 0.621 | | | | |
| early data only (Phase 0 model, group shrinkage) | 0.633 | 0.808 | 0.844 | 0.872 | 0.886 |
| naive precision-weighted update | 0.718 | **0.822** | **0.851** | **0.875** | **0.886** |
| model as prior inside the label model (`model_prior`) | **0.732** | 0.816 | 0.836 | 0.855 | 0.866 |
| same, plus rarity x colour group means (`model_prior_grp`) | 0.685 | 0.819 | 0.837 | 0.859 | 0.870 |
| ATA over the same days | 0.603 | 0.611 | 0.621 | 0.635 | 0.644 |
| model_prior + ATA (z-average) | 0.712 | 0.760 | 0.776 | 0.789 | 0.796 |

Spearman vs the late label. NDCG@20 for `model_prior` on day 1: 0.905 (prior
0.803, early data 0.835, ATA 0.778).

Findings
- **Day 1 is where the model matters.** One day of games (4k-40k per set) alone
  is about as good as the model alone (0.63 vs 0.62); combining them in the label
  model gives 0.73 Spearman and 0.91 NDCG@20, +0.11 Spearman over either.
- From day 2 the games dominate; the model still adds ~+0.01 (naive update vs
  data only) up to day 5, and nothing by day 7.
- `model_prior` (theta_c = mu + b m_c + s_c u_c, b and tau^2 by empirical Bayes;
  b is 1.0-1.2 in every set, so the model's scale is right) is best on day 1 but
  falls slightly behind the naive update later: with one global mean and a tight
  residual tau^2 it shrinks well-measured cards towards the model too much.
  A product should switch to the naive update (or let tau^2 grow) after day 1.
- **ATA stops helping once games exist.** It is weaker than one day of game data
  and z-averaging it in lowers every estimator. (The earlier 0.69 for model +
  week-one ATA was against the full label and without game data.)

Product recommendation: pre-release, show the ensemble; on day 1 show
`model_prior`; from day 2 the naive precision-weighted update.

Caveats: 7 sets; the late label is itself an estimate (noise ceiling ~0.96);
`naive_update` treats the shrunk early SD as a likelihood SD (approximate).
