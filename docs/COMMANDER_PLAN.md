# Commander strength: plan and pre-registration (step I)

2026-11-01. Follows the protocol of `docs/RESEARCH_PLAN.md` (selection on
validation only, paired statistics, logged test looks). Written **before**
any new model was scored.

## Problem

The shipped Commander head (step F2/G: 50/50 fine-tuned EmbeddingGemma-2 +
TF-IDF ridge, test 0.764) has three weaknesses:

1. **Wrong training population.** It is trained only on the ~6k cards first
   printed in the 24 Limited sets. Most Commander cards (Commander precons,
   Masters sets, older expansions, Secret Lairs) are not in that pool, yet the
   app scores all 33k cards with it. Its error on those cards was never
   measured.
2. **No age adjustment.** The label is a current EDHREC snapshot, so a card
   released last month has had little time to be adopted. Within-set ranking
   is unaffected, but the app compares cards across sets (percentiles).
3. **No strength value in the app.** The app shows an inclusion rate and a
   percentile "among cards from recent Limited sets", not a strength score or
   grade comparable to the Limited one.

## Target

Commander strength of card *c* = logit of its EDHREC inclusion rate
(decks playing it / decks whose colour identity allows it), at a reference
age of 2 years after first printing. There are no game outcomes for
Commander, so this is a revealed-preference (play-rate) target; the
Commander Game Changers list (WotC's own judgment of the strongest cards,
Scryfall `game_changer`) is used as an external check that it tracks power and
not only popularity. It is never a feature.

## Data (new, `scripts/cmd_data.py`)

- EDHREC num_decks / potential_decks for **every** Commander-legal card
  (32k), fetched 2026-11-01. Label rows: potential_decks > 1000.
- First paper printing (date, set, set type) from Scryfall Default Cards.
  The oracle bulk file only has one recent printing per card.
- All models below, old ones included, are scored against the **new**
  snapshot, so comparisons are paired on the same labels.

## Split (by first printing, so reprints cannot leak)

| part | first printed |
|---|---|
| fold A: train / validate | < 2024-08-01 / 2024-08-01 … 2025-01-31 |
| fold B: train / validate | < 2025-02-01 / 2025-02-01 … 2025-10-31 |
| final refit | < 2025-11-01 |
| **test** (one look) | 2025-11-01 … 2026-10-31 (same window as the Limited test) |

Groups: a card's first-printing set. Metric: Spearman within group, averaged
over groups with ≥ 20 labelled cards (as the Limited metric is within set).
Secondary: NDCG@20 within group, pooled Spearman.
Test strata: **T1** = cards of the 7 Limited test sets (comparable with the
shipped 0.764); **T2** = all test groups (precons, Secret Lairs, …).

## Hypotheses and decision rules (validation only)

A difference is real if Δ > 2 SE across validation groups and it holds in
≥ 2/3 of groups (as in `RESEARCH_PLAN.md`). GBT models: 3 seeds.

- **H1 (training pool).** TF-IDF ridge trained on all Commander-legal cards
  beats the same ridge trained on the Limited-set cards only, on all groups and
  on Limited groups alone. Ablation: the full pool subsampled to the Limited
  pool's size, to separate "more cards" from "the right cards".
- **H2 (structured features).** Adding colour identity, card types, legendary
  / commander-capable, mana value and log age to the ridge helps.
- **H3 (non-linear model).** GBT on engineered features + TF-IDF SVD, and its
  blend with the ridge (weight chosen on validation), beat the ridge.
- **H4 (age).** A log-age term improves pooled (cross-set) Spearman on
  validation; it is what makes the age-adjusted strength value possible.
- **H5 (fine-tuned encoder on the full pool, GPU).** Only if H1 holds: the
  F2 recipe retrained on the full pool. Prepared as a Kaggle job and run only
  after Alessandro says yes.

The shipped model (fine-tuned + Limited-pool ridge) cannot be scored on
validation (its fine-tuned half saw the validation sets), so it is compared on
test only, once, against the model chosen on validation.

## App

Commander strength shown as a grade (A+ … F, same cut-offs as Limited) and a
0–100 percentile among all Commander-legal cards, with the expected inclusion
rate at maturity and an 80% range. Measured strength (EDHREC, age-adjusted)
is shown next to the model's when it exists.

## Log

(filled in as steps finish)
- **13:35 data fix.** Prerelease promo printings (`pblb`, `pdsk`, …) came out
  days before their set and were taken as first printings, which split each
  Limited set into two groups and put Limited cards in "non-Limited" groups.
  First printing is now the first non-promo paper printing. All runs redone.
  Validation has 17 groups with ≥ 20 cards (7 Limited sets, 8 groups with no
  Limited-pool card: Commander precons, Jumpstart, Spider-Man, …).
- **13:45 validation decisions** (`out/commander/VAL.md`, ref = shipped
  ridge recipe `ridge_lim`, mean within-group Spearman, all 17 groups):
  - H1 alone (`ridge_all`, same features): +0.019 ± 0.018, tie. With the
    structured features, the full pool beats the Limited pool by
    +0.038 ± 0.020 (`ridge_struct` vs `ridge_struct_lim`). **Adopted with H2.**
  - H2+H4 (`ridge_struct`: TF-IDF + colour identity, types, legendary /
    commander-capable, set type, mana value, log-age curve): **+0.056 ± 0.022,
    13/17, REAL**; Limited groups +0.020 ± 0.008 (6/7); non-Limited groups
    0.41 → 0.52 (+0.109 ± 0.037, 7/8). **Adopted.**
  - H4 age term: removing it costs 0.016 within group and 0.019 pooled
    Spearman. Kept (it also gives the age adjustment).
  - H3 GBT: alone a tie with `ridge_lim` and worse than the ridge; best blend
    0.7 ridge + 0.3 GBT is +0.003 ± 0.005 over `ridge_struct`. **Not adopted.**
  - Recency (since 2015, half-life 3 or 6 years): ties. Not adopted.
  - Shipped fine-tuned head `ft_cmd`, on the 8 groups where it is
    out-of-sample: 0.490 vs `ridge_struct` 0.516 (tie); 50/50 blend 0.534,
    +0.018 ± 0.014 (6/8), tie. **Not adopted**; this is what H5 (fine-tune on
    the full pool) would test.
  - **Chosen model: `ridge_struct`.** Test look (one): `ridge_struct` vs the
    shipped `ridge_lim*0.5+ft_cmd*0.5` on T1 and T2; secondary, reported but
    not used for any choice: `ridge_struct*0.5+ft_cmd*0.5`, `ridge_lim`,
    `ft_cmd`, `ridge_struct_noage`.
- **13:47 test look (pre-registered).** `ridge_struct` 0.656 vs shipped
  0.674 (−0.018 ± 0.019, tie). Its final refit chose alpha 30 by inner-split
  MSE (validation folds had chosen 3 and 1). Secondary blend with the
  fine-tuned head 0.688 (+0.014 ± 0.008, 11/13).
- **13:50 post-hoc (logged as such).** Alpha fixed by the protocol's metric
  (validation Spearman → 1). `ridge_struct_a1` + fine-tuned head: 0.698,
  +0.025 ± 0.006, 12/13. **Shipped**, with the caveat that this choice was made
  after a test look. Report: `out/commander/REPORT.md`.
- **14:10 app.** `scripts/commander.py final` (fit on all 31.7k cards,
  out-of-fold scores, age curve, fine-tuned head offset), `app/build.py` and
  the card page / Card Lab show Commander strength (grade + score /100).
- **Open: H5** (fine-tune on the full pool, GPU) not run; needs Alessandro's go.

## H5: fine-tuned encoder on the full pool (pre-registered 2026-11-01 17:30, before the job ran)

Kaggle job c2 (`gpu/kaggle/lanes_cmd.json`), EmbeddingGemma-2 top-12, the
step-D recipe unchanged (10 epochs, default learning rates), struct = the
Commander features + age curve, card tables from `commander.py p2prep`
(`phase2.py --res commander/p2[_all]`). Runs:
- `g_cmd_full` folds A and B, seed 0, per-epoch validation predictions;
- `g_cmd_full` trained before the test window, 3 seeds, test predictions;
- `g_cmd_all` trained on all labelled cards, 3 seeds, weights saved (app).

Decisions, on validation only (`commander.py import-ft`, `compare`):
1. **Adopt** `g_cmd_full` in place of the shipped head `ft_cmd` if the blend
   `ridge_struct_a1` + `g_cmd_full` (epoch 10) beats `ridge_struct_a1` +
   `ft_cmd` (both 0.5) by > 2 SE and on ≥ 2/3 of the groups where `ft_cmd`
   is out-of-sample ("nonlimited"). On all 17 groups `ft_cmd` is partly
   in-sample, so that comparison is reported but not used.
2. If adopted, the blend weight of the fine-tuned head is chosen from
   {0.3, 0.5, 0.7} on all 17 validation groups (`g_cmd_full` is out-of-sample
   on every group).
3. The epoch count stays 10 (no second job); the per-epoch validation curve is
   reported, and a peak well before 10 is noted as a follow-up.
Test: one look, the adopted blend vs the shipped one (0.698).

### H5 result (2026-11-02 00:00, job c2, ~6.3 GPU h)

- Epoch curve (`g_cmd_full`, mean within-set Spearman, folds A / B): epoch 1
  0.646 / 0.561, 3 0.686 / 0.627, 5 0.694 / 0.586, 10 0.690 / 0.614. Flat
  after epoch 3, so 10 epochs cost nothing in accuracy.
- **Decision rule 1 (pre-registered): not adopted.** On the 8 groups where the
  shipped head is out-of-sample, `ridge_struct_a1` + `g_cmd_full` (0.5) 0.538
  vs `ridge_struct_a1` + `ft_cmd` 0.550: −0.012 ± 0.017, 2/8. Weights 0.3 /
  0.7: −0.008 / −0.018. The head trained on 29k cards is no better a blend
  partner than the one trained on 6k Limited cards.
- Clean comparisons (all 17 groups, every model out-of-sample): `g_cmd_full`
  alone 0.641 vs the ridge 0.656 (tie); ridge + `g_cmd_full` 0.680,
  +0.024 ± 0.012 over the ridge (12/17, REAL). So a fine-tuned head adds about
  as much to the ridge whichever pool it was trained on (+0.019 for `ft_cmd`
  on the clean groups).
- No test look (the rule called for one only if adopted). The test
  predictions of `g_cmd_full` are saved in `out/commander/p2/runs/` unscored.
  The shipped model is unchanged. `g_cmd_all` weights (3 seeds, trained on all
  labelled cards) are in the Kaggle output of job c2 (`models/ft_cmd/g_cmd_all`),
  not used by the app.
