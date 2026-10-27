# Phase 1: card-only strength baselines

2026-10-26

## Summary

All card-feature models beat the rarity-only baseline on all 7 held-out sets. So there is real signal beyond rarity.

- The best model ranks cards within a set at **Spearman 0.59–0.60** (vs 0.42 for rarity alone). It recovers about **0.47 Spearman inside each rarity**, the part rarity can't explain.
- For finding the best cards, **GBT on engineered features + TF-IDF text** is best: NDCG@20 0.83 and 56% of the true top 20 found. Rarity alone gets 0.68 and 39%.
- Absolute error is **1.75 win-rate points** per card vs 2.04 for rarity alone. The gap is modest because most cards sit near replacement level.
- **Frozen sentence embeddings + ridge did worse than plain TF-IDF + ridge** (Spearman about 0.51 vs 0.60), and the larger encoder (Qwen3-Embedding-0.6B) was no better than the 110M ones. Generic sentence embeddings lose the numbers and exact wording that decide card power (costs, P/T, "draw two" vs "draw a card").
- The labels are not the bottleneck. Their noise ceiling is a correlation of about 0.96–0.98 per test set, so most of the remaining gap belongs to the model.

## Data and split

- **Labels**: the Phase 0 debiased GIH label `theta_vs_repl` (logit scale, relative to a replacement-level common). It was fitted per set on 17lands PremierDraft data for **31 sets**, from KHM (Feb 2021) to FRA (Oct 2026). Sets with more than 800k games were subsampled to 800k. The batch runs skipped the opening-hand and archetype labels to save time. ECL keeps its original full run.
- **Rows**: one per (set, oracle ID). Basic lands and bonus-sheet cards are dropped. 8,281 rows in total.
- **Temporal holdout**:
  - Train: 24 sets, KHM to EOE (Aug 2025), 6,533 rows.
  - Validation for hyperparameters: TDM, FIN and EOE. Each model is then refitted on all 24 training sets.
  - Test: 7 sets, 1,748 rows: **TLA, ECL, TMT, SOS, MSH, HOB, FRA** (Nov 2025 to Oct 2026).
- **Contamination**: every text encoder used was released before TLA, so no test set can have been in its training data. TF-IDF and GBT have no pretraining at all.
- **Weights**: 1 / (SD² + 0.02), so noisy labels count less but very precise ones don't dominate.

## Models

| model | input | learner |
|---|---|---|
| `rarity_only` | rarity | precision-weighted mean label per rarity |
| `gbt` | 167 engineered features | LightGBM, rounds early-stopped on validation |
| `tfidf` | TF-IDF of rules text (own name → CARDNAME), type line, mana cost + rarity | ridge |
| `gbt_tfidf` | engineered features + 32 SVD components of the TF-IDF | LightGBM |
| `lm_mpnet_base` | all-mpnet-base-v2 embedding (2021, 110M) + rarity | ridge |
| `lm_bge_base` | bge-base-en-v1.5 embedding (2023, 110M) + rarity | ridge |
| `lm_qwen3_emb_0_6b` | Qwen3-Embedding-0.6B embedding (2025, 600M) + rarity | ridge |

The engineered features cover:

- Mana value, pips per colour, generic cost, X and hybrid costs, and colours.
- Card types and subtypes (Aura, Equipment, Vehicle, Saga), layout (MDFC, adventure), power, toughness and loyalty.
- **Stat efficiency**: P+T minus a vanilla line, P+T = 1.49·MV + 0.61, fitted on training creatures.
- Scryfall keywords that appear on at least 15 training cards.
- 34 rules-text patterns: removal, card draw, tokens, ETB triggers, drawbacks and so on.
- Rarity.
- Four **set-context** features, a crude stand-in for the environment encoder: the commons' average stat efficiency, removal rate, creature rate and mana value. Each card also gets its stat efficiency relative to its set.

## Results

Each metric is computed within each test set, then averaged over the 7 sets. The number after ± is the SD across sets.

**All test cards** (about 250 per set)

| model | Spearman | Spearman within rarity | NDCG@20 | top-20 overlap | MAE (win-rate pts) |
|---|---|---|---|---|---|
| rarity_only | 0.42 ± 0.08 | 0 | 0.68 ± 0.06 | 0.39 ± 0.09 | 2.04 ± 0.15 |
| gbt | 0.58 ± 0.06 | 0.42 | 0.81 ± 0.05 | 0.53 ± 0.13 | 1.80 ± 0.14 |
| tfidf | **0.60 ± 0.07** | **0.47** | 0.80 ± 0.06 | 0.51 ± 0.15 | 1.80 ± 0.13 |
| gbt_tfidf | 0.59 ± 0.08 | **0.47** | **0.83 ± 0.04** | **0.56 ± 0.15** | **1.75 ± 0.16** |
| lm_mpnet_base | 0.52 ± 0.07 | 0.34 | 0.78 ± 0.05 | 0.48 ± 0.16 | 1.98 ± 0.12 |
| lm_bge_base | 0.51 ± 0.07 | 0.34 | 0.78 ± 0.04 | 0.49 ± 0.14 | 1.96 ± 0.10 |
| lm_qwen3_emb_0_6b | 0.51 ± 0.06 | 0.31 | 0.78 ± 0.06 | 0.44 ± 0.13 | 1.97 ± 0.11 |

**Well-measured cards only** (at least 2,000 games in hand). Labels of rarely played cards are pulled toward their rarity × colour mean, which flatters `rarity_only`. On this subset, rarity_only scores 0.51 Spearman, gbt 0.61, tfidf 0.63 and gbt_tfidf 0.62, so the ordering doesn't change.

**New cards only** (oracle ID never in a training set) give the same numbers to within 0.01. Reprints aren't leaking.

**Per-set Spearman, all cards**

| set | rarity_only | gbt | tfidf | gbt_tfidf | best LM |
|---|---|---|---|---|---|
| TLA | 0.43 | 0.58 | 0.62 | 0.59 | 0.51 |
| ECL | 0.51 | 0.65 | 0.69 | 0.69 | 0.62 |
| TMT | 0.46 | 0.64 | 0.58 | 0.65 | 0.55 |
| SOS | 0.44 | 0.51 | 0.56 | 0.52 | 0.47 |
| MSH | 0.45 | 0.58 | 0.60 | 0.58 | 0.58 |
| HOB | 0.25 | 0.48 | 0.48 | 0.48 | 0.43 |
| FRA | 0.41 | 0.59 | 0.65 | 0.63 | 0.54 |

## Findings and caveats

- **There is signal beyond rarity, and it holds every time.** Every non-trivial model beats rarity_only on all 7 test sets. About 0.47 Spearman inside a rarity is the share a card's text and stats explain.
- **Text matters as much as hand-built features.** TF-IDF + ridge matches or beats the GBT on 5 of 7 sets. Combining the two helps most at the top of the ranking (NDCG@20 and top-20 overlap).
- **Off-the-shelf sentence embeddings are a weak frozen encoder for this task.** A larger encoder doesn't help, which fits the design's view that the text encoder needs fine-tuning. The size-vs-contamination ablation in section 6 needs fine-tuned encoders to say anything.
- **The most important GBT features** are rarity, toughness, token-making, total pips, number of abilities and mana value. Set context comes next: set mean MV and stat efficiency relative to the set.
- **Hard sets.** HOB (188 cards) and SOS are the hardest. Rarity alone is especially weak on HOB at 0.25.
- **Noise.** LightGBM results move by about ±0.005 Spearman between runs, so treat differences that small as ties.
- **MAE in win-rate points** uses one conversion factor, 22.1 points per logit, fitted on the training sets. The per-card label `ame_vs_repl_pp` is the target it is compared against.
- **Pipeline-coverage caveats:**
  - Oracle text comes from Scryfall's current wording, not the text as printed for that set.
  - Old exports (KHM to VOW) have no colour columns. For those, deck colours are inferred from the decklist: a colour is main at 4 or more non-land cards, a splash at 1–3.

## Pipeline changes, files and how to rerun

Changes to Phase 0, needed to run older and larger sets:

- `data.py`:
  - Streams the CSV in batches with optional `max_games` subsampling, so it no longer runs out of memory on 1M+ game sets.
  - Handles the 2021 exports: a tar inside the gzip, the old column names and rank format, no `tutored_` columns, commas stripped from names, and counts written as "0.0".
- `labels.py`: builds the sparse matrices without a dense float64 copy, which was about 2.5 GB per matrix.
- `scryfall.py`: also matches back-face names.
- `phase0.py`: has a new `--max-games` option.

New files:

- `mtgstrength/features.py`, `mtgstrength/evaluate.py`
- `scripts/phase1.py` runs every model and the evaluation, in about 1 minute on CPU.
- `scripts/embed.py` writes frozen embeddings. Each one takes 5–30 minutes on CPU.
- Results in `out/phase1/`:
  - `metrics_summary.csv`, `metrics_per_set.csv`
  - `test_predictions.csv`, one row per test card with every model's prediction
  - `gbt_feature_importance.csv`, `run_info.json`
  - `emb/*.parquet`

To rerun:

```
python scripts/phase0.py --set <SET> --skip-oh --min-arch-games 1000000000 --max-games 800000
python scripts/embed.py --model BAAI/bge-base-en-v1.5 --name bge_base
python scripts/phase1.py
```

## Next steps

- **Fine-tune a text encoder on θ** with the precision-weighted Gaussian NLL plus the within-set pairwise loss (design section 4). Frozen embeddings fall short, and TF-IDF's strength suggests the information is in the text.
- **Add the human-expert baseline** from section 5.4, such as pre-release set-review grades or first-week ATA for the test sets. That is the comparison that says whether 0.60 is useful.
- **Backfill the opening-hand and archetype labels** for the 30 batch-run sets. This takes about 10 minutes per set.
