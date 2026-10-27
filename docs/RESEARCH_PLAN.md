# Research plan: one ordered queue from here to the final card model

2026-10-27. Replaces the separate queues of the Oct 27 threads. It builds on
`docs/GPU_RERUN_PLAN.md` (the run catalogue R0–R9, the cost model and the
resolution analysis) and puts every remaining item in **one dependency order**:
a step runs only after every step whose result could change it.

**Scope:** encoders and embedding models only, no LLMs (Alessandro, Oct 27).
**Budget:** Kaggle, 2× T4, 30 GPU h/week (resets Saturday 00:00 UTC); 0.48 h
used this week before this plan. GPU h below are T4 hours, recalibrated from R0
(EmbeddingGemma-2 top-4: ~50 s/epoch, 3 seeds × 11 epochs ≈ 23 min).

## The dependency chain

```
A protocol + tooling ──► C contamination ──► D recipe on Gemma-2 ──► E encoder choice
                                                                         │
     B in-flight work (screening only) ─────────────────────────────────┤
                                                                         ▼
                              F architecture + heads on the chosen encoder
                                                                         ▼
                              G final model: seeds, backtest, ONE test look, probes
                                                                         ▼
                              H downstream: app precompute + swap, early update,
                                deck-model distillation, embeddings
```

Why this order:

- **C before E.** Whether Harrier, Nemotron or a 2025 Qwen model may be scored
  on all test sets, or only on sets after its release, decides which encoders
  can compete at all.
- **D before E.** Comparing a new encoder against an untuned EmbeddingGemma-2
  would only show that the baseline was weak. The input format (reminder text)
  and the training recipe (full FT vs top-k, lr, loss weights) are fixed first,
  on the reference encoder, then reused for every candidate encoder.
- **E before F.** The env encoder, no-ctx, LambdaRank and aux-head effects were
  ties on the structured model and may change with a stronger text tower, so
  they are decided on the final encoder, not before.
- **F before G, G before H.** The app's 33k-card precompute, the early-update
  prior and any distillation all consume the final model's predictions.
  Nothing downstream is computed until the model is frozen.

## Rules for every step (frontier-lab rigor)

1. **Selection on validation only.** Three forward-chaining folds
   (V1: train KHM…LCI, validate MKM OTJ MH3; V2: …MH3 → BLB DSK FDN;
   V3: …DFT → TDM FIN EOE). The selection metric is mean within-set Spearman
   over the 9 validation sets; NDCG@20 is secondary.
2. **Test-set access log** in `out/TEST_LOG.md`: one line per run scored on the
   7 test sets (date, run, reason). Every earlier test-scored run is backfilled.
   Steps C–F do not look at test. The final model is scored on test once, in G.
3. **Paired statistics.** Every comparison is Δ vs a named reference: mean ± SE
   across sets, sets won, and a card-level bootstrap within set. A difference is
   real only if Δ > 2 SE and it holds on ≥ 2/3 of sets.
4. **Seeds.** 3 seeds for comparisons (seed SD of a 3-seed mean ≈ 0.002), 5 for
   the shipped model; seed spread reported with every result.
5. **Pre-registration.** Each GPU job's commit message states the hypothesis,
   metric and go/no-go threshold before the job runs. The thresholds are below.
6. **GPU baselines.** GPU runs are compared with GPU runs only.

## The steps

| step | what | depends on | GPU h | decision it makes (pass / fail) |
|---|---|---|---|---|
| **A** | Protocol and code, CPU only: `out/TEST_LOG.md` (backfilled); `phase2.py --val-fold V1/V2/V3` with saved val predictions; `compare_runs.py` scoring val by default, `--test` opt-in, `--ref` paired stats + bootstrap; `--top-k all` with layer-wise lr decay, `--amp fp16`, `--peft lora`; `scripts/contamination_probe.py`. Smoke-tested on CPU with a tiny encoder. | — | 0 | none; gates every GPU step |
| **B** | Close in-flight work. Kaggle v7 (env, no-ctx, LambdaRank, matched 4-epoch baseline on untuned Gemma-2 top-4, scored on test, logged as such). CPU early update (Bayesian prior + first D days) for the last 4 sets. | — | ~1 (running) | **Screening only.** No architecture is adopted from it; F re-decides on validation. The early update's numbers are provisional until H reruns them with the final prior. |
| **C** | Contamination probe, inference only (R1): (a) name-only embeddings + ridge, seen vs unseen sets, for EmbeddingGemma-2, Qwen3-Embedding-0.6B/4B, Harrier-0.6B, Nemotron-3.5-1B; (b) paraphrase sensitivity of the fine-tuned models. | A | ~1 | An encoder is **flagged** if its seen − unseen gain over the clean encoders is ≥ 0.03 on (a), or its paraphrase drop is ≥ 0.02 larger on seen sets. Flagged encoders are scored on post-release sets only. |
| **D1** | Input format on Gemma-2 top-4, 3 seeds × 3 folds: current text vs keyword reminder-text expansion. | A | ~1 | Adopt if ≥ +0.01 on val. Fixed for all later steps. |
| **D2** | Full FT vs top-k (R2): k ∈ {4, 8, all}, 3 seeds × 3 folds. | D1 | ~2.5 | Adopt full FT if ≥ +0.01 over k = 4. |
| **D3** | Hyperparameter sweep (R3) on the D2 winner: ~20 configs × 1 seed × 3 folds, top 2 at 3 seeds. Includes the epoch count and the FT : gbt_tfidf ensemble weight. | D2 | ~5 | The tuned config becomes **the reference** for E and F. |
| **E1** | Encoder ladder (R4a) with the D recipe: Qwen3-Embedding-0.6B (full FT), 4B (LoRA r = 16, fp16), Harrier-0.6B; mini lr sweep on V3, then 3 seeds × 3 folds. Encoders flagged in C are scored on clean sets only. | C, D3 | ~6 | Go to E2 if 4B beats tuned Gemma-2 by ≥ 0.015, or the 0.6B → 4B slope is clearly positive. Otherwise the best of E1 is the backbone. |
| **E2** | Qwen3-Embedding-8B, QLoRA (R4b), 3 seeds. Week 2, only if E1 passes. | E1 | ~8 | Ship only if ≥ +0.01 over 4B at < 3× CPU inference cost. |
| **F1** | Architecture on the chosen backbone (R5): env encoder (`--env set`), `--no-ctx`, LambdaRank, aux deck-value head; 3 seeds × 3 folds each. | E | ~4 | Adopt each only if > 2 SE on val. |
| **F2** | Commander head with the final recipe (R7), 3 seeds. | E | ~1 | Replace the TF-IDF ridge in the app if ≥ +0.01 on val. |
| **G** | Final model: 5 seeds; rolling-origin backtest MKM–EOE for the final model and `gbt_tfidf` (R6, 11 extra held-out sets); **one** scored look at the 7 test sets; behavioural probes (bigger P/T, lower MV, reprints in new sets, uncertainty on new mechanics). | F | ~6 | Reported result. Probes report monotonicity violations; a high rate re-opens the monotonicity regularizer. |
| **H** | Downstream, only after G: app precompute of all 33k Scryfall cards (GPU, minutes) and model swap; early update rerun with the final prior (CPU); deck-model work if justified (R8 scale-up, R9 end-to-end with the final encoder). | G | ~0.5 (+3 / 8 if R8 / R9) | — |

**Calendar:** this week (to Oct 31, ~29 h left): B, C, D1–D3, E1 ≈ 16.5 h
plus ~4 h reserve. Week 2: E2 (if gated in), F1, F2, G, H ≈ 12–20 h. Week 3:
R8/R9 only if the case below holds. If Kaggle counts session hours rather than
GPU hours (checked from v7's quota use), two lanes per session halve the cost.

## Parked items, judged on value

| item | verdict | why |
|---|---|---|
| Size-vs-contamination ablation | **folded into C + E** | The Qwen3-Embedding ladder plus the probe answer it with one family and a clean release date. |
| Behavioural probes | **in G** | Inference only, cheap, and they evaluate the final model. |
| Monotonicity regularizer | **parked, re-opened by G** | Unlikely to move Spearman by ≥ 0.01; worth it only if the probes show many violations. |
| Reminder text | **in D1** | It is an input-format choice, so it must be fixed before the sweep and the encoder comparison. |
| Pick-order head | **parked** | An aux target like the deck head, which tied (+0.001 ± 0.004); needs 24 sets of draft dumps on CPU. Revisit after G if aux targets help in F1. |
| Bigger / uncapped deck model (R8) | **H, optional** | Improves synergy search, not card-strength metrics (the test-set gap is unseen mechanics). |
| End-to-end deck model (R9) | **H, week 3** | The one bet on the 0.75 → 0.54 generalization gap; needs the final encoder, so it goes last. |
| Archetype labels backfill | **parked** | CPU work with no consumer until an archetype head exists. |
| Forge AST encoder | **parked** | A new input modality with high build cost; reconsider only if unseen mechanics remain the main error after G. |
| MTGGoldfish history | **any time, CPU** | Independent of the card model; blocked on web.archive.org reachability. |

## Log of progress

Each finished step adds one line here and its result to CLAUDE.md.

- 2026-10-27: plan written; B running.
