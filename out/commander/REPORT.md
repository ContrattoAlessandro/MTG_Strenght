# Commander strength (step I), 2026-11-01

Plan and pre-registration: `docs/COMMANDER_PLAN.md`. Full tables:
`VAL.md`, `TEST.md` (both from `python scripts/commander.py report`).
Test looks are logged in `out/TEST_LOG.md`.

## Summary

- **Data.** EDHREC play rates for every Commander-legal card (31,694 labelled,
  was 8,046), first paper printing from Scryfall's full printing list, and a
  strict temporal split by first printing.
- **Model.** A ridge on TF-IDF + structured Commander features (colour
  identity, types, legendary / can be a commander, product type such as
  Commander precon, mana value, an age curve), trained on **all**
  Commander-legal cards, averaged 50/50 with the existing fine-tuned
  EmbeddingGemma-2 Commander head (`g_commander`).
- **Result on cards first printed in the last 12 months** (13 sets, nothing
  of which either model saw): mean within-set Spearman **0.698** vs **0.674**
  for the shipped model, +0.025 ± 0.006, better on 12/13 sets, card bootstrap
  [+0.004, +0.045]. On the 7 Limited test sets 0.780 vs 0.761 (7/7).
  **Caveat: this number comes from a second, post-hoc look** (see "Protocol").
  The pre-registered model, the same ridge alone with its regularization
  chosen by MSE on an inner split, tied the shipped model (−0.018 ± 0.019).
- **App.** Every card now shows a **Commander strength** grade (A+ … F) and
  score out of 100: its percentile among all 31.7k Commander cards, measured
  (EDHREC, adjusted to 2 years after release) and predicted (out-of-fold for
  cards with data, live for Card Lab cards).

## Target

Commander has no game outcomes. Strength = logit of the EDHREC inclusion rate
(decks playing the card / decks whose colour identity allows it), the share of
deckbuilders who choose the card when they can. Snapshot 2026-11-01.

**External check.** The Commander Game Changers list (53 cards WotC judged the
strongest; Scryfall `game_changer`) against each score, AUC over 31.7k cards:

| score | AUC |
|---|---|
| observed inclusion (age-adjusted) | 0.932 |
| model, shipped blend | 0.883 |
| model, out-of-fold ridge | 0.855 |
| EDHREC salt | 0.994 |

Play rate identifies the strongest cards well, but salt (how much people hate
facing a card) identifies them almost perfectly: the list is as much about
"unfun" as about power. Play rate stays the target because it exists for every
card and is not a judgment about fun.

**Age.** The fitted age curve (logit, relative to 1 month after release) is
−0.26 at 3 months, −0.16 at 6, −0.05 at 12, +0.06 at 24, −0.33 at 4 years,
−0.85 at 10 years. New cards are over-played in their first month (hype), dip,
then recover; older cards are played less, conditional on their text (power
creep). Strength is reported at max(age, 2 years); measured values of cards
younger than 2 years are lifted by the curve (at most ~0.3 logit).

## Validation (folds A: Aug 2024 – Jan 2025, B: Feb – Oct 2025; 17 sets)

Δ vs the shipped ridge recipe (`ridge_lim`: TF-IDF + rarity, Limited-set
cards only), mean within-set Spearman:

| model | all 17 | 7 Limited sets | 8 other sets | verdict |
|---|---|---|---|---|
| `ridge_lim` (shipped recipe) | 0.597 | 0.767 | 0.408 | ref |
| H1 same ridge, all cards (`ridge_all`) | +0.019 ± 0.018 | −0.002 | +0.053 | tie |
| H1 control: all cards, subsampled to Limited size | +0.023 ± 0.023 | −0.008 | +0.069 | tie |
| H2 structured features, Limited cards only | +0.018 ± 0.009 | −0.003 | +0.044 | tie |
| **H1+H2+H4 `ridge_struct`** | **+0.056 ± 0.022 (13/17)** | +0.020 (6/7) | +0.109 (7/8) | **REAL** |
| same without the age curve | +0.040 ± 0.023 | +0.005 | +0.086 | tie |
| H3 GBT (3 seeds) | −0.002 ± 0.019 | −0.015 | +0.013 | tie |
| best GBT blend (0.7 ridge + 0.3 GBT) vs `ridge_struct` | +0.003 ± 0.005 | | | tie |
| recency (since 2015, half-life 3/6 y) vs `ridge_struct` | ≤ +0.005 | | | tie |

- More cards alone do not help; more cards **with** product type and age do
  (+0.038 ± 0.020 over the same features on Limited cards). Old and precon
  cards are played differently for reasons the text does not show, and the
  model needs to be told.
- The gain is largest where the old model was weakest: sets with no Limited
  cards (Commander precons, Jumpstart, Spider-Man, …), 0.41 → 0.52.
- The shipped fine-tuned head, on the 8 sets where it is out-of-sample: 0.490
  alone, and 0.550 averaged with the alpha-1 ridge, vs 0.516 for the
  pre-registered ridge (+0.034 ± 0.013, 7/8) and 0.531 for the alpha-1 ridge alone. Not adopted before the test look under the
  pre-registered rule (it was a tie with the MSE-chosen alpha).

## Test (first printed Nov 2025 – Oct 2026; 13 sets, 2,404 cards)

| model | all 13 | Limited 7 | other 4 | Δ vs shipped |
|---|---|---|---|---|
| shipped: `ridge_lim` + fine-tuned | 0.674 | 0.761 | 0.502 | ref |
| `ridge_lim` alone | 0.646 | 0.741 | 0.449 | −0.028 ± 0.018 |
| fine-tuned head alone | 0.652 | 0.746 | 0.476 | −0.022 ± 0.008 |
| **pre-registered: `ridge_struct` (alpha 30 by inner MSE)** | 0.656 | 0.749 | 0.452 | −0.018 ± 0.019, tie |
| `ridge_struct` + fine-tuned (secondary, pre-declared) | 0.688 | 0.773 | 0.516 | +0.014 ± 0.008, 11/13 |
| post-hoc: `ridge_struct`, alpha 1 | 0.690 | 0.763 | 0.531 | +0.016 ± 0.010, 9/13 |
| **post-hoc, shipped now: alpha 1 + fine-tuned** | **0.698** | **0.780** | 0.531 | **+0.025 ± 0.006, 12/13** |

### Protocol

The pre-registered model chose its ridge penalty by MSE on the last 6 months
of its training window. That rule picked alpha 3 / 1 on the validation folds
but **30** on the final refit, and alpha 30 is much worse for ranking
(validation: 0.586 at alpha 30 vs 0.655 at 1, fold B). The selection metric
of the protocol is within-set Spearman on the validation folds, which picks
alpha 1 (mean of A and B 0.656, vs 0.656 at 3 and 0.605 at 30). The shipped model uses that
alpha. This decision was made after the test look and is logged there as a
post-hoc diagnostic, so the +0.025 is not a clean held-out number; the clean
statements are: validation +0.056 over the old recipe (REAL), and the
pre-registered test look a tie. Every model here also gets one fresh check:
the next Commander products (EDHREC snapshot in a few months).

### Uncertainty

The 80% range comes from the validation residuals of the ridge
(−1.35 … +1.01 logit). On the test cards it covers **71%**, so it is somewhat
too narrow for new cards. The prediction shrinks a lot toward the mean: true
staples (Sol Ring, Rhystic Study) are predicted in the top 15% but with a play
rate of ~1%, so read the grade and range, not the expected rate, for extreme
cards.

## What did not work or was not done

- GBT on engineered features + TF-IDF SVD: below the ridge everywhere.
- Weighting recent cards more: no gain.
- 303 cards (mostly split cards) have no EDHREC page under the slug used.
- **Not run (GPU): fine-tuning EmbeddingGemma-2 on the full card pool (H5).**
  The fine-tuned head only ever saw ~6k Limited cards and is still half of
  the shipped score; on precon-type sets it is the weaker half. This is the
  most promising next step and needs a Kaggle job (`phase2.py` would need a
  card table that is not tied to Limited sets).
