# Final card model (research plan step G), 2026-10-31

Everything below follows `docs/RESEARCH_PLAN.md`: the model was chosen on the
validation folds only (steps C-F), then scored once on the 7 test sets (TLA,
ECL, TMT, SOS, MSH, HOB, FRA). Test looks are listed in `out/TEST_LOG.md`.

## Model

EmbeddingGemma-2, top 12 of 24 layers fine-tuned on cached lower-layer states,
plus the structured MLP; Gaussian NLL with label variance + within-set
Bradley-Terry loss; default hyperparameters (enc-lr 5e-5, head-lr 1e-3,
batch 32), 10 epochs, original card text, 4 set-context features, no
environment encoder. Trained on all 24 training sets, 5 seeds
(`g_final`, weights in `/mnt/project-files/mtg_strength/models/ft_card/g_final/`).
Shipped as the 50/50 average with `gbt_tfidf`.

What led here (validation, 3 seeds × 9 sets): top-12 beat top-4 by +0.020 ±
0.006; a 10-config sweep kept the defaults; Qwen3-Embedding 0.6B / 4B were
+0.011 / +0.012 (not significant, more expensive); env encoder, no-ctx and the
deck-value aux head were ties.

## Limited (test, mean within-set Spearman over 7 sets)

| model | Spearman | NDCG@20 | MAE (pp) |
|---|---|---|---|
| final model `g_final` | **0.652** | 0.814 | 1.71 |
| final model + gbt_tfidf (shipped) | 0.650 | 0.820 | **1.66** |
| previous model (top-4, Oct 27) | 0.626 | 0.788 | |
| previous ensemble | 0.629 | 0.813 | |
| gbt_tfidf | 0.591 | **0.830** | 1.75 |
| first-week ATA (humans, after release) | 0.652 | 0.819 | |
| Draftsim pre-release (humans) | 0.551 | 0.810 | |
| final model + first-week ATA (z-average) | **0.702** | | |

`g_final` vs the previous model: +0.025 ± 0.006, 7/7 sets, card bootstrap
[+0.008, +0.039]. Per set: TLA 0.693, ECL 0.686, TMT 0.670, SOS 0.600, MSH
0.671, HOB 0.545, FRA 0.697. Against first-week ATA: −0.000 ± 0.010 (2/7
sets), i.e. a pre-release model now matches the first week of human drafting;
against Draftsim it wins 6/7.

**Rolling backtest** (each of MKM … EOE predicted from all earlier sets, 1
seed): 0.649 vs gbt_tfidf 0.616 (+0.033 ± 0.013, 8/10); the blend 0.661
(10/10).

**Uncertainty.** With 10 epochs on 12 layers the Gaussian head is
overconfident (val z SD 4.0). One scale factor fitted on the validation folds
(`scripts/calibrate_sd.py`, k = 7.8) gives test z SD 1.03 and 90% interval
coverage 0.90. The SD tracks the error only weakly (Spearman 0.23 vs 0.40 for
the old model).

## Commander head

Fine-tuned on logit EDHREC inclusion (first-printing labels), same recipe,
3 seeds (`g_commander`). Validation: 0.795 vs TF-IDF ridge 0.768; 50/50 blend
0.798 (9/9 sets). Test (one look): fine-tuned 0.746, ridge 0.749, **blend
0.764** (+0.014 ± 0.004 over the ridge, 6/7). The app uses the blend.

## Behavioural probes (`out/probes/`, 1,748 test cards)

| edit | cards | final model: drops > 0.01 | mean Δ (logit) | gbt_tfidf: drops |
|---|---|---|---|---|
| power +1 | 993 | 3.2% | +0.015 | 0.4% |
| toughness +1 | 1002 | 4.7% | +0.008 | 0.6% |
| one less generic mana | 1327 | 16.7% | +0.012 | 10.9% |

On average the edits help, as they should, but cheaper cards get a lower score
in 1 case out of 6. This is the case for the parked monotonicity regularizer
(augmented counterfactual pairs).

Reprints (96 cards in several labelled sets): the model's change between
environments correlates 0.34 with the label's change, but moves 5× less (SD
0.007 vs 0.031 logits). New mechanics (385 test cards with a keyword no
training card has): predicted SD barely higher (+3%), error 5% higher. The
model neither adapts much to environments nor flags new mechanics.

## App

`out/app/ft_scores.parquet` and `out/app/ft_cmd_scores.parquet`: every Scryfall
card (33,483) scored by the final model and the Commander head (typical
environment), computed on a Kaggle T4. `app/build.py` uses them; the live
engine reproduces the training predictions to 6e-5 logits.

## Early update

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
