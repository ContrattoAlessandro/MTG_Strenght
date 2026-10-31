# Early update with the final prior

Rerun with the final prior (`ens_g_final+gbt_tfidf`, calibrated `g_final_sd`),
all 7 test sets, Kaggle CPU job c1; `out/early_update_final/`. Target = labels
from games after day 7 (Spearman, mean over sets; old prior in brackets):

| day | prior | data only | naive update | model_prior | ATA |
|---|---|---|---|---|---|
| 0 | **0.640** (0.621) | | | | |
| 1 | | 0.633 | 0.719 (0.718) | **0.727** (0.732), NDCG@20 0.901 | 0.603 |
| 2 | | 0.808 | **0.818** (0.822) | 0.817 | 0.611 |
| 3 | | 0.844 | **0.847** (0.851) | 0.839 | 0.621 |
| 5 | | 0.872 | 0.871 (0.875) | 0.859 | 0.635 |
| 7 | | **0.886** | 0.884 (0.886) | 0.871 | — |

The better prior (+0.019 before release) washes out as soon as one day of games
exists: day 1 and later are within ±0.005 of the old run. Product rule
unchanged: ensemble before release, model_prior on day 1, naive update from
day 2 (equal to data alone by day 5). Adding ATA still hurts once game data
exists.

Methods: `out/early_update/REPORT.md` (same script, older prior).
