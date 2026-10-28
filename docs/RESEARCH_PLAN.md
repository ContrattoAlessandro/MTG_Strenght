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

- 2026-10-27 14:20: plan written; B running.
- 14:50 **A done** (tooling, test log). **B done**: v7 screening on test, all
  ties (env +0.005 ± 0.003, no-ctx +0.003 ± 0.004, LambdaRank −0.006 ± 0.003);
  LambdaRank dropped from F. Kaggle bills session time, so 2 lanes = 1.
- 17:30 **C done, no encoder flagged.** Name-only probe, seen − unseen gain over
  mpnet/bge: Gemma-2 +0.002 ± 0.011, Qwen3-Emb-0.6B +0.007 ± 0.009, 4B
  −0.004 ± 0.011, Harrier +0.003 ± 0.008, Nemotron −0.004 ± 0.015 (threshold
  0.03). Names add only ~0.01 over rarity for every encoder. Paraphrase drops on
  Gemma-2 fold models are ≤ 0.012 and not larger on seen sets. All encoders may
  be scored on all sets; the probe's detection limit is ~0.02. Qwen3-Emb-8B is
  probed only if E2 runs.
- 17:30 **D1 done: keep the original text.** no_reminder +0.0046 ± 0.0052 (5/9),
  below the +0.01 bar.
- 17:30 **D2 partial**: val (9 sets, 3 seeds, best epoch) top-4 0.6246, top-8
  0.6334 (+0.0089 ± 0.0047, 8/9 sets, just under the bar). Full FT ran out of
  memory; rerun with gradient checkpointing plus top-12 in job v9. The 50/50
  blend with gbt_tfidf_folds adds ~+0.01 to every variant (top-8 blend 0.643).
- 21:50 **D2 done: top-12 is the reference.** Val (9 sets, best epoch, all
  seeds) vs top-4 0.6246: top-8 0.6334 (+0.009 ± 0.005, tie), **top-12 0.6444
  (+0.020 ± 0.006, 7/9, better; 3 seeds, seed SD 0.001)**, full FT lr 2e-5 0.6387
  (2 seeds, +0.014 ± 0.007), full FT lr 5e-5 0.6389 (1 seed). Seed-0-only:
  full FT lr 5e-5 0.639 vs top-12 0.635 (tie). By the pre-registered rule
  (highest val Spearman among variants passing the bar) top-12 wins; it is also
  2.7× cheaper than full FT (113 vs 310 s/epoch). Its curve still rises at
  epoch 8, so D3 allows 10 epochs.
- 21:50 **E1 lr screen** (session B, V3, 1 seed): Qwen3-Emb-0.6B best at 3e-5
  (V3 0.649), Harrier-0.6B at 1e-4 (0.652); ~320 s/epoch full FT fp16. Qwen3-4B
  LoRA screen running.
- 21:50 **D3 cut to 10 configs** (default + 9 drawn, 1 seed × 3 folds, ≤ 10
  epochs, ~4 h on 2 lanes): at ~57 min per config 20 would not fit this week's
  quota next to E–G.
- 18:30 **G prep**: gbt_tfidf rolling-origin backtest over MKM–EOE (10 sets) =
  0.616 Spearman, NDCG@20 0.819 (CPU).
- 00:55 **E1 stage 1 done** (1 seed × 3 folds, best epoch, vs Gemma-2 top-12
  seed 0 = 0.635): Qwen3-Emb-0.6B 0.649 (+0.014 ± 0.011, 6/9), Qwen3-Emb-4B LoRA
  0.647 (+0.012 ± 0.010, 6/9), Harrier-0.6B 0.622 (−0.013 ± 0.019, dropped: more
  than 0.01 behind the best). **E2 (8B) is not run**: 4B does not beat the best
  0.6B model by ≥ 0.01 (−0.002). Stage 2 (3 seeds × 3 folds): Qwen3-0.6B,
  Qwen3-4B and the D3 winner on Gemma-2.
- 04:00 **D3 done: the default recipe stays**, now with 10 epochs. 8 of 10
  configs ran (the two with batch 64 ran out of memory on a T4). Val, seed 0,
  vs the default (d3_c00, 0.640): best c01 (enc-lr 2e-5, head-lr 3e-4, batch 16)
  +0.003 (4/9), under the +0.005 bar; three configs worse (high enc-lr 1e-4 with
  head-lr 3e-3, or enc-lr 1e-5). The default at epoch 8 reproduces D2's top-12
  seed 0 (0.636 vs 0.635); epochs 9-10 add +0.004. The 3-seed follow-up is cut
  to the default only (seeds 1-2 in job v11, with Qwen3-0.6B seeds 1-2).
- 07:30 **E1 done: EmbeddingGemma-2 stays the backbone.** Val, 3 seeds × 3
  folds, best epoch, vs Gemma-2 top-12 10 epochs (d3_c00, 0.6487): Qwen3-Emb-0.6B
  0.6598 (+0.011 ± 0.008, 6/9, tie), Qwen3-Emb-4B LoRA 0.6605 (+0.012 ± 0.007,
  7/9, tie); 4B vs 0.6B +0.001 ± 0.005. Neither passes Δ > 2 SE, so by the
  pre-registered rule the cheaper reference stays: Gemma-2 (300M, top-12) also
  keeps CPU inference cheap for the app. Blended 50/50 with gbt_tfidf_folds the
  gap shrinks to +0.005 (Gemma 0.653 vs Qwen 0.658). 10 epochs beat 8 (d2 −0.004
  ± 0.002, 7/9). The consistent +0.011 is noted: if F/G leave budget, a 5-seed
  Qwen3-0.6B rerun could settle it.
- 07:30 **F1 started** (job v12, seed-0 screen, env encoder and no-ctx);
  seeds 1-2 after the Oct 31 quota reset (~27 of 30 h used this week).
- 09:05 **F1 screen** (seed 0, val, vs d3_c00 seed 0 = 0.640): env encoder
  0.645 (+0.006 ± 0.003, 7/9), no-ctx 0.644 (+0.005 ± 0.006, 5/9). Both ties
  so far; both get seeds 1-2 after the quota reset (Sat 00:00 UTC), together
  with the aux deck-value head (3 seeds) and the Commander head (F2).
- 09:40 **F2 prep (CPU)**: `phase4.py commander-folds` writes first-printing
  Commander labels (`commander_labels_first.csv`, so reprints never carry a
  label across a fold) and fold baselines: TF-IDF ridge val 0.768, GBT+TF-IDF
  0.733. `compare_runs.py --label-csv` scores against it. F2 bar: the
  fine-tuned head replaces the ridge in the app only if ≥ +0.01 and > 2 SE.
  Caveat for the aux head: the deck model was trained on all 24 training sets,
  so its values on fold-training cards saw the validation sets' games; a gain
  from it would be optimistic, a tie is conclusive.
- Jobs v13 (env, no-ctx seeds 1-2) and b6 (aux head, Commander head, 3 seeds)
  are committed and go to Kaggle at the quota reset (Sat Oct 31 00:00 UTC).
- 10:55 **Job b6 (seed 0)**. Aux deck-value head: 0.6398 vs 0.6396
  (+0.000, 4/9, tie). Even with its optimistic bias (deck model trained on the
  val sets' games) it adds nothing, matching Phase 3 (+0.001 ± 0.004): **aux
  head dropped**, its seeds 1-2 are not run. **F2 Commander head** (first-
  printing labels, val): fine-tuned Gemma-2 top-12 0.793 vs TF-IDF ridge 0.768
  (+0.025 ± 0.007, 8/9); 50/50 blend with the ridge 0.798 (+0.030 ± 0.004,
  9/9). Passes the bar on one seed; seeds 1-2 at the reset confirm it before
  the app switches.
