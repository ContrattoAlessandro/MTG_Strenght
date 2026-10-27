# Phase 2: fine-tuned card encoder, SOTA encoders, human baselines

2026-10-26

## Summary

- **A fine-tuned EmbeddingGemma-2 is the best card model so far: Spearman 0.62** within set, against 0.60 for the best Phase 1 model (TF-IDF + ridge). It also has the best within-rarity ranking (0.51 against 0.47). It is better on 5 of the 7 test sets, tied on one and worse on one (ECL).
- **The text does most of the work.** Fine-tuned on text alone, EmbeddingGemma-2 gets 0.615 (one seed). The engineered features add about 0.01 on top, and on their own (`struct_mlp`) they get 0.59.
- **Fine-tuning is what helps, not a newer encoder.** Zero-shot, the four 2025–2026 encoders score 0.515–0.530, no better than the 2021–2025 ones (0.508–0.519). Fine-tuning the top 4 layers of EmbeddingGemma-2 takes it from 0.52 to 0.62.
- **No clear sign of contamination**, within what 7 test sets can show. The newer encoders' small zero-shot gains are about the same on sets they could have seen and sets they could not. Section "Contamination check" has the details and the limits.
- **The model beats Draftsim's pre-release ratings** (0.62 against 0.55 Spearman, better on 6 of 7 sets). **It doesn't beat the drafting community after one week** (first-week ATA, 0.65). The two are complementary: averaging the model with first-week ATA gives 0.69, and averaging it with the pre-release ratings gives 0.65.
- **Weak spot: the very top of the ranking.** The fine-tuned model's NDCG@20 is 0.78, below the Phase 1 GBT (0.83) and the human ratings (0.81–0.82). Averaging it with the GBT gets 0.81 NDCG@20 and keeps 0.63 Spearman. That average is the best all-round model.
- **Uncertainty is roughly calibrated.** Standardized errors have SD 1.12 (1.0 is ideal), and 88% of labels fall inside the 90% interval. The predicted SD correlates 0.40 with the actual error.

## Setup

Data, labels, temporal split, metrics and test sets are the same as Phase 1:

- Train on 24 sets (KHM to EOE). Choose epochs on TDM, FIN and EOE, then refit on all 24.
- Test on TLA, ECL, TMT, SOS, MSH, HOB and FRA (Nov 2025 to Oct 2026): 1,748 cards.
- The CPU-only constraint holds throughout (4 cores, no GPU).

### Card model (design section 3, card encoder + card head)

- **Text tower.** A pretrained encoder reads the type line, mana cost, P/T and rules text (own name replaced by `CARDNAME`), mean- or last-token pooled as the encoder was trained.
- **Structured tower.** An MLP reads the 167 Phase 1 engineered features, which include the four set-context features.
- **Head.** The two towers are concatenated, and the head predicts a mean and a variance for θ.
- **Large encoders on CPU.** Only the top k layers are fine-tuned. The frozen layers below run once, and their output is cached (`mtgstrength/partial.py`). A check confirms the cached path matches the full model to within 1e-4 relative error. This makes a 0.3–0.6B encoder trainable in about 8 minutes per epoch.

### Loss (design section 4)

- **Gaussian NLL.** The variance is the predicted variance plus the label's own posterior variance, so noisy labels count for less without a hand-set weight.
- **Within-set Bradley–Terry pairwise loss.** Each batch holds cards from one set. A pair's target is P(θ_i > θ_j) given both labels' SDs.

### Protocol

- **Epoch count.** Picked on the validation sets by Spearman + NDCG@20, smoothed over neighbouring epochs.
- **Seeds.** Each fine-tuned model is trained with 2–3 seeds, and the reported prediction is their mean.
- **Text-only ablation.** It reuses the epoch count chosen for the full model. It has one seed so far: a second was cut short when the machine ran out of memory.

## Results

The protocol and metrics match Phase 1: each metric is computed per test set, then averaged over the 7 sets.

**All test cards**

| model | Spearman | within rarity | NDCG@20 | top-20 overlap | MAE (pts) |
|---|---|---|---|---|---|
| rarity_only | 0.42 | 0 | 0.68 | 0.39 | 2.04 |
| Phase 1 tfidf (ridge) | 0.60 | 0.47 | 0.80 | 0.51 | 1.80 |
| Phase 1 gbt_tfidf | 0.59 | 0.47 | **0.83** | **0.56** | 1.75 |
| structured MLP only (`struct_mlp`) | 0.59 | 0.45 | 0.79 | 0.49 | 1.80 |
| **EmbeddingGemma-2, top 4 layers fine-tuned + structured** | **0.62** | **0.51** | 0.78 | 0.49 | 1.77 |
| EmbeddingGemma-2, top 4 layers fine-tuned, text only (1 seed) | 0.62 | 0.49 | 0.78 | 0.47 | 1.80 |
| Harrier-0.6B, top 2 layers fine-tuned + structured | 0.61 | 0.49 | 0.78 | 0.50 | 1.82 |
| avg(EmbeddingGemma-2 fine-tuned, gbt_tfidf) | **0.63** | **0.51** | 0.81 | 0.52 | **1.71** |
| avg(Harrier fine-tuned, gbt_tfidf) | 0.62 | 0.50 | 0.82 | 0.55 | 1.72 |
| *Human:* Draftsim pre-release ratings | 0.55 | 0.38 | 0.81 | 0.52 | – |
| *Human:* first-week ATA (17lands) | 0.65 | 0.56 | 0.82 | 0.54 | – |

The well-measured subset (≥ 2,000 games in hand) and the new-cards subset give the same ordering. On the well-measured subset, the fine-tuned EmbeddingGemma-2 scores 0.64 and the average with gbt_tfidf 0.65.

**Per-set Spearman**

| set | gbt_tfidf | tfidf | FT Gemma-2 | FT Harrier | avg(FT Gemma-2, gbt_tfidf) | Draftsim pre-release | ATA week 1 |
|---|---|---|---|---|---|---|---|
| TLA | 0.59 | 0.62 | 0.66 | 0.67 | 0.65 | 0.59 | 0.65 |
| ECL | 0.69 | 0.69 | 0.65 | 0.62 | 0.69 | 0.59 | 0.69 |
| TMT | 0.65 | 0.58 | 0.66 | 0.64 | 0.68 | 0.60 | 0.70 |
| SOS | 0.52 | 0.56 | 0.57 | 0.54 | 0.57 | 0.43 | 0.62 |
| MSH | 0.58 | 0.60 | 0.66 | 0.65 | 0.65 | 0.46 | 0.68 |
| HOB | 0.48 | 0.48 | 0.48 | 0.48 | 0.50 | 0.58 | 0.56 |
| FRA | 0.63 | 0.65 | 0.68 | 0.65 | 0.67 | 0.61 | 0.68 |

## Zero-shot SOTA encoders and contamination

Frozen embeddings + ridge, exactly as the Phase 1 `lm_*` models:

| encoder | released | data cutoff | Spearman | within rarity |
|---|---|---|---|---|
| all-mpnet-base-v2 (110M) | 2021-08 | – | 0.519 | 0.34 |
| bge-base-en-v1.5 (110M) | 2023-09 | – | 0.513 | 0.34 |
| Qwen3-Embedding-0.6B | 2025-06 | – | 0.508 | 0.31 |
| EmbeddingGemma-2 (740M total, 270M text) | 2026-09 | **Jan 2025** (model card) | 0.515 | 0.33 |
| Harrier-OSS-v1-0.6B (Microsoft, Qwen3-based) | 2026-03 | not stated | 0.526 | 0.35 |
| Nemotron-3.5-Embed-1B (NVIDIA; text tower from Nemotron-3-Embed-1B, Jul 2026) | 2026-10 | not stated | 0.530 | 0.35 |

**Contamination check.** The test sets were released between Nov 2025 and Oct 2026, so the two newest encoders might have read spoilers, set reviews or tier lists for some of them. Similar zero-shot scores are not proof on their own: a genuinely better encoder and a contaminated one would both score higher. So the check also compares each newer encoder's gain over the mean of the three older ones on the sets whose spoiler season (4 weeks before Arena release) began before the encoder's date with its gain on the sets after it:

| encoder | gain on sets it could have seen | gain on sets it could not have seen |
|---|---|---|
| EmbeddingGemma-2 (cutoff Jan 2025) | – | +0.002 (7 sets) |
| Harrier-0.6B | +0.015 (4 sets) | +0.008 (3 sets) |
| Nemotron-3.5-1B | +0.025 (5 sets) | −0.003 (2 sets) |

The differences point the way contamination would, but they are about 0.01–0.03 Spearman. Per-set noise is about ±0.03, and the comparison rests on only 2–3 sets on one side. This is weak evidence either way. After fine-tuning, Harrier is no better than EmbeddingGemma-2 on the 4 sets it could have seen: it is 0.01 better on TLA and 0.02–0.03 worse on ECL, TMT and SOS. So there's no sign that it carries memorized ratings. The reported fine-tuned model, EmbeddingGemma-2, has a documented data cutoff 10 months before the first test set, so it is the clean choice. Nemotron wasn't fine-tuned. It had the weakest contamination profile, and at 1.5B parameters it was the most expensive to train on CPU.

A frozen embedding is a fairly insensitive probe for contamination, since a model can know a card's reputation without that showing in a general-purpose embedding. A stronger probe would ask a generative model of the same family to rate the cards. That isn't feasible on this CPU budget.

## Human baselines (design section 5.4)

- **Draftsim pre-release ratings** (0–5 scale, used by Draftsim's draft bots). Draftsim revises them after release, so the current files leak game results. For example, TLA's Aang's Journey moved from 2.5 to 2.8. The ratings used here are Wayback Machine snapshots: for each set, the latest one before the first day of PremierDraft on Arena. TMT's only snapshot is from the morning of launch day. FRA's ratings changed a lot between 22 and 25 September; 25 September is still 4 days before Arena release, so it is used. With the 22 September snapshot, FRA scores 0.49 instead of 0.60. Files and URLs are in `scripts/human_baselines.py` and `data/raw/human/`.
- **First-week ATA.** Average pick position in the first 7 days of PremierDraft, computed from the 17lands public draft dumps (1.0–1.5M picks per set). This is not a pre-release grade: by then the community has played a week of drafts and seen early 17lands data. It is the "beat ATA from the first week" bar from the design.
- **Coverage.** All 1,748 test cards have both ratings, so every number above uses the same cards.

Combining the model with the humans (z-scores averaged within set):

| | Spearman |
|---|---|
| FT EmbeddingGemma-2 alone | 0.62 |
| Draftsim pre-release alone | 0.55 |
| model + Draftsim pre-release | 0.65 |
| first-week ATA alone | 0.65 |
| model + first-week ATA | **0.69** |

The model and the pre-release ratings correlate only 0.62, and each adds information to the other. This fits the design's view of the card model as a pre-release prior that is then updated with play data. Details are in `human_combination.csv`.

## Caveats

- **Seeds and noise.** The fine-tuned models are 2–3-seed averages. Differences of about 0.01 between them are within run-to-run noise.
- **Not tuned.** Learning rates, the number of fine-tuned layers (k) and the loss weights weren't searched. Harrier used k = 2 rather than 4 to fit the CPU budget, which may understate it.
- **Top of the ranking.** The fine-tuned model ranks the whole set better but finds bombs less well than the GBT. The NLL and pairwise loss treat all pairs equally; a top-weighted ranking loss, or the ensemble, addresses this.
- **Not done in this phase.** No monotonicity regularizer, no environment encoder beyond the set-context features, and no behavioural probes. Model weights aren't saved; only predictions are.

## Files and how to rerun

New files:

- `scripts/phase2.py`: `prep`, `train`, `zeroshot` and `evaluate` commands.
- `mtgstrength/partial.py`: top-k fine-tuning on cached states.
- `scripts/human_baselines.py`: human baselines.
- Results in `out/phase2/`:
  - `metrics_summary.csv`, `metrics_per_set.csv`
  - `test_predictions.csv` (every model and human rating, per test card)
  - `contamination_per_set.csv`, `calibration.csv`, `human_combination.csv`, `human_ratings.csv`
  - `runs/<model>/`: training logs and per-seed predictions
- New embeddings in `out/phase1/emb/`: `embeddinggemma2`, `harrier_0_6b`, `nemotron35_1b`.

To rerun:

```
python scripts/phase2.py prep
python scripts/embed.py --model google/embeddinggemma-2 --name embeddinggemma2
python scripts/phase2.py zeroshot --embeddings out/phase1/emb/*.parquet
python scripts/phase2.py train --encoder google/embeddinggemma-2 --top-k 4 --name ft_embeddinggemma2_k4 --max-epochs 6 --seeds 0 1 2
python scripts/human_baselines.py --draft-dir <cache dir for the 17lands draft dumps>
python scripts/phase2.py evaluate
```

On 4 CPU cores:

- Embedding all cards with a 0.6–1.5B encoder takes about 1 hour.
- Caching the frozen layers takes about the same.
- Fine-tuning EmbeddingGemma-2 (top 4 layers) takes about 8 minutes per epoch, or about 2.5 hours with validation and 3 seeds.

Packages added: `torch` (CPU build), `transformers`, `sentence-transformers[image]`, plus `torchvision` (CPU build), which EmbeddingGemma-2's processor needs.

## Next steps

1. **Top-weighted ranking loss, or the ensemble as the default.** The fine-tuned model's main weakness is at the top of the ranking.
2. **Behavioural probes** (design 5.5): save the fine-tuned model, then check that predicted strength rises when a card gets bigger or cheaper.
3. **A real environment encoder** in place of the four set-context features. HOB, which is small and different, is still the hardest set for every model and both human baselines.
4. **Bayesian update with early play data.** The model + first-week ATA result (0.69) suggests the most useful product is the model as a prior, updated as 17lands data comes in.
