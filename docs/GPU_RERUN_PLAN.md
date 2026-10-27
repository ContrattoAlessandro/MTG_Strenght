# Which experiments to rerun now that we have a GPU

2026-10-27 · review of Phases 0–4 and the Oct 27 work, written for the move from
4 CPU cores to Kaggle T4s (about 30 GPU hours per week, `docs/GPU.md` on the
`gpu-kaggle` branch).

**Scope (Alessandro, Oct 27): encoders and embedding models only, no LLMs**,
even with the GPU. Every run below uses an embedding/encoder backbone
(EmbeddingGemma-2, Harrier, Qwen3-Embedding, Nemotron-Embed). No generative
model is used as a rater, a feature or a probe. Qwen3-Embedding and Harrier
are built on a Qwen3 backbone but are released, trained and used here only as
embedding models (pooled vectors into our head, never text generation).

## Bottom line

1. **Most of our conclusions were shaped by what the CPU allowed, but only a few
   of them are actually at risk.** The ones at risk are: "a bigger or newer
   encoder doesn't help", "Harrier is no better than EmbeddingGemma-2", "the
   model isn't tuned", and the deck model's size and data. The rest either
   don't need a GPU, or a GPU wouldn't change their result.
2. **Our real limit is the number of test sets, not seeds.** The 7 test sets
   only let us detect differences of about 0.01 between close variants and about
   0.03 between model families (table below). So GPU hours should go to
   experiments that could move Spearman by 0.03 or more, plus the work that
   makes our comparisons sharper. Re-running ablations that moved things by 0.003
   isn't worth it.
3. **Highest value per GPU hour:** (a) a contamination probe that needs only
   encoders: embed each card's *name alone* and check whether a newer encoder
   can rank cards from the name on sets it could have seen. Plus a
   paraphrase-sensitivity test. (b) Full fine-tuning against top-k fine-tuning,
   and a real hyperparameter sweep on EmbeddingGemma-2 over several validation
   folds. (c) A size ladder of embedding models from one family
   (Qwen3-Embedding 0.6B → 4B, then 8B if the trend is positive). Qwen3-Embedding
   was released in June 2025, before the first test set's spoilers, so it is
   contamination-clean by release date.
4. **Before scaling, fix the protocol.** Use paired per-set statistics, select
   on several temporal validation folds instead of 3 sets, and keep a log of
   every look at the test sets. With a GPU we will run 10× more configurations,
   and without these changes the 7 test sets would quietly become a validation
   set.
5. **Not GPU work:** uncapped Phase 0 labels, the deck-game arrays, the early
   17lands update and Forge simulation. These are limited by RAM, downloads or
   Java CPU time. Forge self-play gains nothing from a GPU, so next step 13 in
   CLAUDE.md should drop "GPU" as its precondition.

Week 1 plan: about 20 GPU h of runs plus 4 h in reserve (section 4). The first
Kaggle job, already running (EmbeddingGemma-2 top-4, 3 seeds), is step R0. It
measures real T4 throughput and CPU↔GPU parity, and every cost below gets
recalibrated from it.

## 1. How much can our evaluation resolve?

Seed spread and set-to-set spread were measured from the saved per-seed test
predictions (`scripts/compare_runs.py`, mean within-set Spearman on the 7 test
sets):

| comparison | mean Δ | SE across the 7 sets (paired) | sets won |
|---|---|---|---|
| FT Gemma-2 k4 − gbt_tfidf (different model families) | +0.032 | 0.016 | 6/7 |
| FT Gemma-2 k4 − structured-only (`s_base`) | +0.036 | 0.010 | 6/7 |
| FT Gemma-2 k4 − FT Harrier k2 (same recipe, different encoder) | +0.015 | 0.007 | 5/7 |
| FT + aux deck head − FT (one loss change) | +0.001 | 0.004 | 4/7 |
| `struct_mlp` − `s_base` (rerun of the same model) | +0.000 | 0.002 | 5/7 |

The seed SD of a single run is 0.001–0.004. The SD of a 3-seed mean is therefore
about 0.002.

What this means:

- **The smallest effect we can detect (about 2 SE) is roughly 0.01 for close
  variants and 0.02–0.03 for different model families.** The CPU-era ablations
  (LambdaRank, aux head, no-ctx, env encoder on the structured model) all landed
  below this and are correctly reported as ties. A GPU doesn't change that: more
  seeds shrink only the seed part of the error, and that part is already about
  the same size as the set part for close variants and far smaller for
  different families.
- **3 seeds is enough for comparisons.** Use 5 only for the model we ship, where
  the seed average is itself the product. One exception: measure seed spread
  again for the LoRA-tuned 4B/8B encoders, which are usually noisier on 6.5k
  examples.
- **The FT Gemma vs Harrier gap (0.015 ± 0.007) is borderline significant, and
  it is confounded.** Harrier was trained with k = 2 to fit the CPU budget,
  Gemma with k = 4 (see E3).
- **More evaluation sets is the only way to sharpen claims across model
  families.** A rolling-origin backtest over MKM–EOE adds 11 held-out sets (R6).
  Every extra set costs one training run per model, which is cheap on a GPU and
  was not on 4 cores.

## 2. Each experiment the CPU shaped

"GPU h" is the estimated T4 cost of the GPU version (section 6 has the cost
model). "Verdict" says whether to rerun it.

| # | experiment | how the CPU shaped it | conclusion at risk | GPU version | GPU h | verdict |
|---|---|---|---|---|---|---|
| E1 | Frozen encoders zero-shot (Phase 1–2: mpnet, bge, Qwen3-Emb-0.6B, Gemma-2, Harrier, Nemotron) | only ≤1.5B embedding models, frozen | "bigger frozen encoders don't help" | stands as is; the bigger-encoder question moves to fine-tuning (R4) | 0 | **no rerun** |
| E2 | Contamination check (Phase 2) | probe limited to whole-text frozen embeddings and 2–3 sets per side | "no clear sign of contamination" rests on a weak probe | encoder-only probes: name-only embeddings and paraphrase sensitivity, seen vs unseen sets (R1) | 1 | **rerun as R1** |
| E3 | Fine-tuned encoders (Phase 2): Gemma-2 top-4, Harrier top-2, Nemotron not fine-tuned | top-k layers on cached states; k fixed by time (Harrier k = 2); 1.5B skipped | "fine-tuning helped, newer encoders didn't"; "Harrier ≈ Gemma" | full fine-tuning plus k ∈ {4, 8, all} on Gemma-2 (R2); same-family embedding-model ladder, Qwen3-Embedding 0.6B/4B/8B (R4) | 2 + 7–15 | **rerun as R2, R4** |
| E4 | Hyperparameters (Phase 2) | never searched: lr 5e-5 / 1e-3, batch 32, pair_w 1, epochs picked on 3 validation sets | every FT comparison is between untuned models | sweep of about 20 configs over 3 temporal validation folds (R3) | 5 | **rerun as R3** |
| E5 | Seeds (all phases) | 2–3 seeds, 1 for text-only and Commander | comparisons are noise-limited | keep 3; give the 1-seed runs (text-only, Commander) 3 seeds | <1 | only the 1-seed runs |
| E6 | `max_len` 256 | chosen for speed | none: the longest card text is 920 characters (≈250 tokens), mean 238 | no change | 0 | **no rerun** |
| E7 | Phase 3 deck model | d = 96, 2 layers, 2 epochs, 200k games per set (4.65M); frozen Gemma embeddings as card tokens; ≈2 h CPU | "distilling the deck model adds nothing" | (a) scale it: all games, d = 256, 4 layers (R8); (b) train it end to end with a trainable card encoder, so millions of games supervise the text model (R9) | 3 / 8+ | (a) medium, (b) the high-upside bet |
| E8 | Distillation ablations (Phase 3) | 2 seeds, 4 epochs, no validation | ties within 0.004 | redo only if R9 succeeds | 0 | **no rerun now** |
| E9 | Structured-model ablations (Oct 27: env, no-ctx, LambdaRank) | run on the structured model because it was fast on CPU; FT versions queued on CPU until about 21:30 | effects might grow with a stronger text tower | rerun the env encoder and LambdaRank on the backbone that wins R3/R4 (R5) | 3 | after R3/R4 |
| E10 | Commander head (Phase 4) | 1 seed, Limited hyperparameters; TF-IDF ridge 0.749 > FT 0.717 | "TF-IDF beats fine-tuning for Commander" | short sweep + 3 seeds (R7) | 1 | low |
| E11 | Interaction search (Phase 4) | top 40 cards per colour pair, from the 200k-game deck model | small, but limited by the deck model | follows R8; the game-length confound is a CPU analysis fix | in R8 | after R8 |
| E12 | Phase 0 labels capped at 800k games per set; opening-hand and archetype labels skipped on 30 sets | RAM and time of the sparse logistic fit, **not** compute a GPU speeds up | small: label noise ceiling is already 0.96–0.98 | run on a Kaggle **CPU** session (30 GB RAM, no GPU quota) or locally | 0 | not GPU work |
| E13 | Forge simulation labels (Phase 4, dropped) | Forge AI is single-threaded Java | none from a GPU: ~10^7 games per set needs a CPU cluster | drop; fix the wording in CLAUDE.md next step 13 | 0 | **drop** |
| E14 | Early 17lands update (Oct 27) | none: a Bayesian update with no deep training | — | — | 0 | not GPU work |

## 3. Protocol changes before spending GPU hours

These are code and process changes. They cost no GPU time and should land before
R3.

1. **Test-set discipline.** Model choice, epochs and hyperparameters come from
   validation only. Keep `out/TEST_LOG.md`: one line per run scored on the test
   sets (date, run, why). A final model is scored on test once, and every
   test-scored run is reported, not just the winners. `compare_runs.py` should
   score validation by default, with `--test` as an explicit opt-in.
2. **Temporal validation folds instead of 3 sets.** Forward chaining over the
   training period, with each fold trained only on earlier sets:

   | fold | train | validate |
   |---|---|---|
   | V1 | KHM … LCI | MKM, OTJ, MH3 |
   | V2 | KHM … MH3 | BLB, DSK, FDN |
   | V3 | KHM … DFT | TDM, FIN, EOE (the current validation) |

   This gives 9 validation sets instead of 3. Select on the mean over folds. A
   fold costs one training run, about 5–10 min for the 300M model on a T4.
3. **Paired statistics.** Report Δ against the reference model as mean ± SE
   across sets, the number of sets won, and a card-level bootstrap within each
   set (resample cards, keep sets fixed). Add a `--ref` option to
   `compare_runs.py`. Call a difference real only if it is more than 2 SE and
   holds on at least 5 of 7 sets.
4. **Pre-register each GPU experiment:** the hypothesis, the metric (mean
   Spearman over validation folds, NDCG@20 secondary), and the go/no-go
   threshold, written in the job's commit message before it runs. The
   thresholds for this plan are in section 4.
5. **GPU baselines.** GPU kernels are not bit-identical to CPU kernels. Compare
   GPU runs with GPU runs, including a GPU rerun of the reference
   (`ft_embeddinggemma2_k4`, which R0 does).
6. **Contamination by release date.** A model's public release date is a hard
   upper bound on its training cutoff. Any model released before TLA's spoiler
   season (early Oct 2025) is clean for all 7 test sets. Validation sets
   (TDM–EOE, Mar–Aug 2025) can be contaminated for 2025 models. That inflates
   validation and can bias selection toward models that memorised ratings, so
   report each model's validation-to-test drop next to its score.

## 4. Ranked reruns and weekly budget

The ranking is by expected value per GPU hour: the chance a run changes a
decision or the shipped model, times how much it would change, divided by its
cost. Probabilities are my judgment, not measurements.

| rank | run | GPU h | why it ranks here | go / no-go |
|---|---|---|---|---|
| R0 | **GPU parity + throughput** (running now: Gemma-2 top-4, 3 seeds) | ~1.5 | Calibrates every estimate here and validates the pipeline. | Must match CPU 0.623 ± 0.01. Record sec/epoch. |
| R1 | **Encoder-only contamination probe.** (a) *Name-only probe:* embed only each card's name with EmbeddingGemma-2 (cutoff Jan 2025), Qwen3-Embedding-0.6B (released Jun 2025), Harrier-0.6B (Mar 2026) and Nemotron-3.5-1B (Oct 2026). Fit ridge on training-set names, score within-set Spearman on the test sets. A name carries little strength signal beyond creature type, so a newer encoder that does better on sets it could have seen than on ones it couldn't (vs the clean encoders) has memorised reputations. (b) *Paraphrase sensitivity:* score the fine-tuned models on meaning-preserving rewrites of the rules text (ability order swapped, reminder text dropped or added). A memoriser loses more on seen cards than unseen ones. Inference only. | ~1 | Closes the rigor gap in E2 without a generative model, and decides whether Harrier and Nemotron can be used on all 7 test sets or only the ones after their release. | Flag an encoder if its seen-minus-unseen gain over the clean encoders is ≥0.03 on (a), or its paraphrase drop is ≥0.02 larger on seen sets in (b). Flagged encoders are reported on clean sets only. |
| R2 | **Full FT vs top-k on EmbeddingGemma-2**: k ∈ {4, 8, all}, layer-wise lr decay for "all", 3 seeds, fold V3. | ~2 | Clean answer to "full fine-tuning vs partial" at almost no cost. Full FT of 300M is cheap on a T4. Sets the backbone for R3. | Adopt full FT if ≥ +0.01 over k = 4 on validation. |
| R3 | **Hyperparameter sweep, 300M encoder**: ~20 configs (enc lr 1e-5–2e-4, head lr, pair_w 0.3–3, epochs, dropout, batch 16–64), 1 seed × 3 folds, then 3 seeds for the top 2. | ~5 | Every later comparison needs a tuned reference, or a "bigger model wins" result just means "the untuned baseline was weak". Expected gain +0.01–0.02. | The tuned config becomes the reference for R4–R7. |
| R4a | **Embedding-model size ladder, stage 1**: Qwen3-Embedding-0.6B (full FT, 8-bit Adam) and Qwen3-Embedding-4B (LoRA r=16 on all linear layers, fp16), last-token pooling as the model was trained, plus the same struct tower and head. lr from a 3-point mini sweep on V3, then 3 seeds. Optionally Harrier-0.6B full FT (same Qwen3 base) to settle "Harrier ≈ Gemma" on the sets R1 marks clean. | ~6 | The biggest possible gain. It answers next step 10 (size vs contamination) with one family and a clean release date. Prior: about 30% chance of ≥ +0.02 over the tuned EmbeddingGemma-2. | Go to R4b if 4B beats the tuned EmbeddingGemma-2 by ≥0.015 on the validation folds, or if the within-family slope 0.6B → 4B is clearly positive. |
| R4b | Stage 2 (week 2, gated): Qwen3-Embedding-8B, QLoRA (4-bit, fp16 compute), 3 seeds. | ~8 | Only if the trend says it pays. | Ship it only if it beats 4B by ≥0.01 at <3× inference cost on CPU (the web app runs on CPU). |
| R5 | **Re-check FT ablations on the winning backbone**: env encoder (Set Transformer) and LambdaRank top-weighting, 3 seeds. Text-only vs +struct only if the backbone changed. | ~3 | The structured-model ablations were ties. A stronger text tower may leave more for the environment to explain (HOB is still the hardest set). | Adopt if > 2 SE on the validation folds. |
| R6 | **Rolling-origin backtest of the final model and gbt_tfidf**: predict each set from MKM to EOE using only earlier sets (one run per fold, 3 seeds). | ~5 | Raises the held-out evaluation from 7 sets to 18 (11 backtest sets + 7 test), so the final "model vs GBT" and "small vs large encoder" claims become solid. Run once, at the end. | Reporting only. |
| R7 | **Commander head**: short lr/epoch sweep with the R3 recipe, 3 seeds. | ~1 | Small; tests whether fine-tuning really loses to TF-IDF (0.717 vs 0.749) or was just untuned with 1 seed. | Replace the TF-IDF ridge in the app if ≥ +0.01. |
| R8 | **Deck model scale-up**: all games (arrays built on CPU first), d = 256, 4 layers, more epochs until validation log loss stops improving. | ~3 | Improves the synergy search and the aux head. Unlikely to change card-strength metrics, because new-set values are limited by unseen mechanics (0.75 on training sets vs 0.54 on test), which more capacity doesn't fix. | Keep if validation log loss < 0.638 (now 0.642). |
| R9 | **End-to-end deck model with a trainable card encoder** (week 3): card tokens come from the fine-tuned text tower, and game outcomes train the encoder alongside θ. | 8+ | The one design that attacks the 0.75 → 0.54 generalization gap: it trains the text model on millions of games instead of 6.5k labels. High risk, needs new code. | Pilot on 6 sets; continue if new-set deck values reach ≥0.60 vs θ. |
| — | More seeds on existing CPU runs; Phase 0 or Forge on GPU | 0 | See sections 1 and 2. | Don't run. |

**Week 1 (≈24 of 30 h):** R0 1.5 · R1 1 · R2 2 · R3 5 · R4a 6 · reserve 4
(failed jobs, reruns), plus about 4 h to move future FT ablations off the CPU (a 3-seed FT run is
about 2.5 h on 4 cores and should take well under 1 h on a T4).

**Week 2:** R4b 8 (if gated in) · R5 3 · R7 1 · R8 3 · R6 5 · reserve 5.

**Week 3:** R9 pilot, then the final model scored once on test and written up.

Order of dependence: R0 → R2 → R3 → (R4a, R5) → R4b → R6. R1 and R7 are
independent and can share a session with anything else.

## 5. T4 engineering notes

- **No bf16 on T4.** EmbeddingGemma's model card says its activations don't
  support float16, so run EmbeddingGemma-2 in fp32 on a T4. That is still fast
  at 300M. Qwen3-Embedding and Harrier (Qwen3-based) are fine in fp16 autocast with a
  grad scaler, which is one reason the size ladder uses Qwen3-Embedding: there
  is no EmbeddingGemma larger than ~300M, and a fp32 4B model would be slow on
  a T4. Verify with a 50-step smoke test (no NaN, loss matches the
  fp32 run) before each new model family.
- **Use both T4s.** The Kaggle accelerator is 2× T4 and the job uses one. Run
  two seeds or two folds in parallel (`CUDA_VISIBLE_DEVICES=0` / `1`, separate
  cache dirs). If the quota counts session hours, not GPU hours (check
  `kaggle quota` before and after one 2-GPU job), this doubles throughput for
  free.
- **Memory on a 15 GB T4** (batch 32, ~100 tokens per card):

  | model | method | fits? |
  |---|---|---|
  | EmbeddingGemma-2 (~300M text) | full FT, fp32, AdamW (~5 GB + activations) | yes |
  | Qwen3-Embedding-0.6B / Harrier-0.6B | full FT, fp32 (~10 GB) | tight; use 8-bit Adam or LoRA |
  | Qwen3-Embedding-4B | LoRA, fp16 weights (8 GB) + gradient checkpointing | yes |
  | Qwen3-Embedding-8B | QLoRA 4-bit (~5.5 GB) + gradient checkpointing | yes, slow |

- **Keep cached states where they still help.** For top-k runs the cache is
  still the cheapest path. For full FT and LoRA, drop it and tokenize on the
  fly. Inputs are short: mean 238 characters, max 920.
- **Sessions stop at 12 h** and checkpoints on scratch are lost. Keep each job
  under about 8 h, and save each finished seed's predictions to
  `/kaggle/working` as it completes (the job already copies `out/`).
- **Inference for the web app stays on CPU.** A 4B or 8B card model would need
  its predictions precomputed for all ~30k Scryfall cards (one GPU pass of a few
  minutes), with live scoring of custom cards falling back to the 300M model.
  Factor this into R4b's go/no-go.
- **New code needed:** LoRA and full-FT paths in `phase2.py` (`--peft lora`,
  `--top-k all`), `--amp fp16`, multi-fold validation (`--val-fold`),
  `compare_runs.py --ref/--val`, and a `contamination_probe.py` for R1
  (name-only embeddings through `embed.py`, plus the paraphrase set). All can be
  written and smoke-tested on CPU with a tiny model before using quota.

## 6. Cost model (recalibrate after R0)

Tokens per epoch ≈ 6,533 training cards × ~65 tokens ≈ 0.42M. Training FLOPs ≈
6·N·tokens for full FT and ≈ 4·N·tokens for LoRA (the backward pass still runs
through the frozen weights). Assumed effective T4 throughput: ~5 TFLOP/s in
fp32, ~15 TFLOP/s in fp16 with LoRA, about half that with 4-bit dequantization.
Multiply by 2–3 for padding, validation passes and Python overhead.

| model | est. min / epoch | est. GPU h, 6 epochs × 3 seeds |
|---|---|---|
| EmbeddingGemma-2 top-4 (cached) | ~1 | ~0.5 (CPU: ~8 min/epoch) |
| EmbeddingGemma-2 full FT, fp32 | ~3–5 | ~1–1.5 |
| Qwen3-Embedding-0.6B full FT | ~5–8 | ~2 |
| Qwen3-Embedding-4B LoRA | ~12–20 | ~4 |
| Qwen3-Embedding-8B QLoRA | ~30–45 | ~8 (4 epochs) |

If R0 comes in more than 1.5× off these numbers, rescale section 4 before
launching R2.
