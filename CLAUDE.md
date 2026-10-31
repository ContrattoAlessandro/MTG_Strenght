# MTG card strength model: handoff for coding agents

This file is the full context of the project. Read it before touching the code.
It was written at the end of the first round of work (Oct 2026), done in a cloud
sandbox, so that work can continue on Alessandro's local machine.

- `README.md`: short user-facing overview, Phase 0 label model and output columns.
- `out/phase1/REPORT.md` … `out/phase4/REPORT.md`: detailed results per phase. Trust them
  over this summary if they disagree.
- `docs/DESIGN.md`: the original design, verbatim. It is the spec; the phases
  below follow its roadmap (section 8).

## Standing decisions (from Alessandro)

1. **Target.** Strength of card *c* in environment *E* = expected change in win
   probability when *c* replaces a replacement-level card in a typical deck that
   can play it. **Limited (17lands PremierDraft) is the primary supervised
   target**, because it is the only dense source with outcomes. Constructed play
   and Commander inclusion are auxiliary heads.
2. **Evaluation is temporal holdout only.** Never use random splits (reprints and
   near-duplicates leak). Group by Scryfall oracle ID.
   - Train: 24 sets, KHM (Feb 2021) to EOE (Aug 2025).
   - Validation (hyperparameters, epoch count): TDM, FIN, EOE; then refit on all 24.
   - Test: TLA, ECL, TMT, SOS, MSH, HOB, FRA (Nov 2025 to Oct 2026), 1,748 cards.
   - The set order is `SETS` in `scripts/phase1.py`; `FIRST_TEST = "TLA"`.
3. **Metrics** are computed within each test set and averaged over the 7:
   Spearman, Spearman within rarity, NDCG@20, top-20 overlap, MAE in win-rate
   points (22.14 points per logit), and calibration of the predicted SD.
4. **Contamination control.** Any pretrained text encoder must be checked:
   test sets should postdate its training cutoff, and results on sets it could
   have seen are reported separately. EmbeddingGemma-2 (documented cutoff Jan
   2025) is the clean reference encoder.
5. **SOTA encoders are first tested zero-shot** (frozen embeddings + ridge) and
   only then fine-tuned. Fine-tuning was what helped, not newer encoders.
6. **CPU only.** Alessandro uses the GPU for another model. Everything so far
   ran on 4 CPU cores; large encoders are trained by fine-tuning only the top
   k layers on cached lower-layer states (`mtgstrength/partial.py`). Ask before
   using a GPU locally.
   Free cloud GPUs (Kaggle, Colab) are fine: `phase2.py train` and
   `phase3.py train` take `--device cuda`, and `docs/GPU.md` has the Kaggle
   job (`gpu/kaggle/`, driven with the `kaggle` CLI) and a Colab notebook.
7. **Human baselines are the bar that matters**: Draftsim pre-release ratings
   (Wayback snapshots from before release) and first-week ATA from the 17lands
   draft dumps.
8. Work autonomously through the roadmap; Alessandro prefers results over
   check-ins.

## Repository layout

```
mtgstrength/        library code
  data.py           17lands download + streaming CSV reader (handles 2021 export quirks)
  labels.py         Phase 0 hierarchical logistic label model (EB partial pooling)
  scryfall.py       Scryfall bulk download, name -> oracle_id join (DFCs, back faces)
  features.py       167 engineered card features (+4 set-context features)
  evaluate.py       within-set metrics
  partial.py        top-k layer fine-tuning on cached hidden states
scripts/
  phase0.py         labels for one set  -> out/<SET>/
  phase1.py         baselines           -> out/phase1/
  embed.py          frozen embeddings   -> out/phase1/emb/<name>.parquet  (not in git)
  phase2.py         prep / zeroshot / train / evaluate -> out/phase2/
  human_baselines.py Draftsim + first-week ATA -> out/phase2/human_ratings.csv
  phase3_data.py    decklists+outcomes  -> <work-dir>/<SET>.npz          (not in git)
  phase3.py         deck model train / values / evaluate -> out/phase3/
  phase4_data.py    EDHREC + MTGGoldfish scraping -> out/phase4/
  phase4.py         Commander and Constructed heads -> out/phase4/
  phase4_interactions.py  card-pair synergy search -> out/phase4/interactions_*.csv
app/                web app (server.py, build.py, engine.py, static/), see "Web app"
data/raw/human/     Draftsim pre-release snapshots (small, in git)
out/<SET>/          Phase 0 labels per set (card_labels.csv, summary.json, ...)
out/phaseN/         results per phase
```

Scripts are run from the repo root (`python scripts/phaseN.py ...`); they
put the repo root and `scripts/` on `sys.path` themselves.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Network hosts the pipeline needs: `17lands-public.s3.amazonaws.com` (game and
draft dumps), `api.scryfall.com` and `data.scryfall.io` (oracle bulk data),
`huggingface.co` / `*.hf.co` (encoders), `json.edhrec.com`, `www.mtggoldfish.com`,
`web.archive.org` (Draftsim and MTGGoldfish snapshots). Some encoders on Hugging
Face (EmbeddingGemma) are gated: accept the licence and `huggingface-cli login`.

### Data not in git (regenerate)

| file | how to get it |
|---|---|
| `data/raw/scryfall_oracle_cards.jsonl.gz` (25 MB) | downloaded automatically by `mtgstrength.scryfall.download_oracle_cards("data/raw")`, which `phase0.py` calls. Note: oracle text is Scryfall's current wording. |
| 17lands `game_data_public.<SET>.PremierDraft.csv.gz` | downloaded automatically by `phase0.py` / `phase3_data.py` into `--data-dir` / `--raw-dir`. 0.5–2 GB each. |
| 17lands `draft_data_public.<SET>.PremierDraft.csv.gz` | downloaded by `human_baselines.py --draft-dir` (test sets only) |
| `out/phase1/emb/*.parquet` (35–95 MB each) | `scripts/embed.py`, see below. Phase 3 needs `embeddinggemma2.parquet`. |
| Phase 3 deck-game arrays (`<SET>.npz`) | `phase3_data.py`. The trained deck model **is** in git: `models/deck_model.pt` (36 MB); pass `--model-dir models` to `phase3.py values` / `phase4_interactions.py`, or retrain with `phase3.py train`. |
| fine-tuned encoder weights | `phase2.py train --save-dir`; the Oct 27 Gemma-2 k4 seeds are in `/mnt/project-files/mtg_strength/models/ft_card/` (too big for git). |

Defaults for `--work-dir`, `--model-dir`, `--cache-dir` point at
`/tmp/claude-0/...` (the sandbox). Pass local paths, e.g. `--work-dir work/p3
--model-dir models --cache-dir work/cache` (all git-ignored).

## How to rerun each phase

```
# Phase 0: labels (one set; ~10 min and ~4 GB RAM per set on 4 CPUs)
python scripts/phase0.py --set ECL --data-dir data --out-dir out
#   batch mode used for the other 30 sets:
python scripts/phase0.py --set <SET> --skip-oh --min-arch-games 1000000000 --max-games 800000

# Phase 1: baselines (~1 min) + frozen embeddings (5–60 min each)
python scripts/embed.py --model BAAI/bge-base-en-v1.5 --name bge_base
python scripts/embed.py --model google/embeddinggemma-2 --name embeddinggemma2
python scripts/phase1.py

# Phase 2: card encoder
python scripts/phase2.py prep
python scripts/phase2.py zeroshot --embeddings out/phase1/emb/*.parquet
python scripts/phase2.py train --encoder google/embeddinggemma-2 --top-k 4 \
    --name ft_embeddinggemma2_k4 --max-epochs 6 --seeds 0 1 2 --cache-dir work/cache
python scripts/human_baselines.py --draft-dir data/raw/draft
python scripts/phase2.py evaluate

# Phase 3: deck model
python scripts/phase3_data.py --raw-dir <17lands cache> --work-dir work/p3 --max-games 200000
python scripts/phase3.py train    --work-dir work/p3 --model-dir models   # ~2 h CPU
python scripts/phase3.py values   --work-dir work/p3 --model-dir models
python scripts/phase3.py evaluate
#   distillation into the card head (ablation of design section 6):
python scripts/phase2.py train --encoder google/embeddinggemma-2 --top-k 4 \
    --deck-values out/phase3/deck_values.csv --target deck --name ft_embeddinggemma2_k4_deckonly ...
python scripts/phase2.py train ... --deck-values out/phase3/deck_values.csv --aux-w 0.5 --epochs 4 --skip-val --seeds 0 1 \
    --name ft_embeddinggemma2_k4_aux
python scripts/phase2.py train ... --deck-values out/phase3/deck_values.csv --aux-w 0.5 --epochs 4 --skip-val --seeds 0 1 \
    --consistency-on-test --name ft_embeddinggemma2_k4_auxcons

# Phase 4: extensions
python scripts/phase4_data.py edhrec
python scripts/phase4_data.py goldfish
python scripts/phase4.py commander
python scripts/phase4.py constructed
python scripts/phase4_interactions.py --sets TLA ECL TMT SOS MSH HOB FRA \
    --work-dir work/p3 --model-dir models
#   fine-tuned Commander head:
python scripts/phase2.py train ... --target label --label-csv out/phase4/commander_labels.csv \
    --name ft_embeddinggemma2_k4_commander
```

Check exact flags with `--help`; the `...` stand for the same encoder/top-k/seed
flags as the Phase 2 run. The exact config of every run is the first line of
`out/phase2/runs/<name>/log.jsonl`.

## Results so far

All numbers are mean within-set Spearman on the 7 test sets unless noted.

### Phase 0: debiased labels (done)

One hierarchical logistic regression per set on 17lands games: nuisance terms
(skill bucket, rank, colours, splash, opponent colours, play/draw, mulligans),
θ_c for "in hand", θ'_c for deck copies, η_{c,arch} archetype deviations;
partial pooling by rarity × colour with empirical-Bayes τ²; SDs from the
inverse Hessian. Label = `theta_vs_repl` (vs median regularly played common),
weight = 1/(SD² + 0.02). 31 sets labelled (KHM–FRA), 8,281 (set, oracle_id) rows.
Only ECL has the opening-hand and archetype labels; the batch runs skipped them.
Label noise ceiling ≈ 0.96–0.98 correlation.

### Phase 1: card-only baselines (done, `out/phase1/`)

| model | Spearman | NDCG@20 | MAE pts |
|---|---|---|---|
| rarity only | 0.42 | 0.68 | 2.04 |
| GBT, 167 engineered features | 0.58 | 0.81 | 1.80 |
| TF-IDF + ridge | 0.60 | 0.80 | 1.80 |
| GBT + TF-IDF SVD (`gbt_tfidf`) | 0.59 | **0.83** | **1.75** |
| frozen mpnet / bge / Qwen3-0.6B + ridge | 0.51–0.52 | 0.78 | ~1.97 |

Frozen sentence embeddings lose numbers and exact wording; bigger frozen
encoders don't help.

### Phase 2: fine-tuned card encoder (done, `out/phase2/`)

- Fine-tuned EmbeddingGemma-2 (top 4 layers + structured MLP, Gaussian NLL with
  label variance + within-set Bradley–Terry loss): **0.62** (within rarity 0.51),
  NDCG@20 0.78. Text-only 0.62, structured-only 0.59, Harrier-0.6B (k=2) 0.61.
- Average of FT Gemma-2 and `gbt_tfidf`: **0.63**, NDCG@20 0.81: best all-round.
- Zero-shot 2025–26 encoders (EmbeddingGemma-2, Harrier, Nemotron-3.5-1B):
  0.515–0.530, no better than older ones. Contamination check: weak, not
  significant differences (see report).
- Humans: Draftsim pre-release 0.55 (model beats it on 6/7 sets); first-week
  ATA 0.65 (model does not beat it). Model + ATA = **0.69**, model + Draftsim = 0.65.
- Uncertainty roughly calibrated: z-score SD 1.12, 88% coverage of 90% interval.

### Phase 3: deck model (done, `out/phase3/REPORT.md`)

- Deck model: card tokens = MLP(engineered features + frozen EmbeddingGemma-2
  embedding) + per-card ID embedding (dropped at random, p=0.5) + seen/copies
  embeddings; environment z_E from the set pool; 2-layer transformer, attention
  pooling; nuisance terms linear. Trained on 4.65M games (200k per training set),
  ~2 h CPU. Weights: `models/deck_model.pt`.
- Val log-loss 0.642 (0.643 without IDs) vs 0.666 nuisance-only, 0.688 constant.
- Counterfactual values vs θ: training sets 0.86 (real decks), 0.80 (no ID),
  0.75 (synthetic); **test sets from synthetic decks only: 0.54**.
- Distillation (2 seeds, `distill_per_set_spearman.csv`): θ only 0.621;
  **θ + aux deck-value head (aux 0.5) 0.624** (tie, within noise);
  θ + aux + consistency on test cards 0.614; deck values only 0.601;
  undistilled deck values 0.521.
- Conclusion: the deck model works but distilling it adds nothing measurable.
  Best test predictor is still the Phase 2 ensemble (0.627–0.629), behind
  first-week ATA (0.652), ahead of Draftsim (0.551).

### Phase 4: extensions (done, `out/phase4/REPORT.md`)

- **Commander** (logit of EDHREC inclusion, 8,046 cards with data): TF-IDF ridge
  **0.749**, fine-tuned EmbeddingGemma-2 0.717 (1 seed, Limited hyperparameters),
  GBT+TF-IDF 0.684, GBT 0.672, rarity 0.465, Limited θ 0.222.
- **Constructed** (censored: MTGGoldfish top-50 staples, Standard/Pioneer/
  Modern/Legacy/Pauper): **only the 20 live lists** were used because the
  Wayback history was unreachable (connection resets). 333 staples (73 test).
  AUC within set: GBT 0.81, censored GBT 0.81, TF-IDF logistic 0.79 (best avg.
  precision 0.27), Limited θ 0.65, rarity 0.65. Spearman among staples ≤ 0.2.
- **Interaction search**: deck-model 2×2 synergy over the top 40 cards per colour
  pair, validated against empirical b_ab in held-out test-set games. Weighted
  correlation mean **0.22** (0.13–0.29), p = 0.002 in all 7 sets; top-decile
  pairs positive, bottom-decile negative everywhere. Caveat: "both seen"
  correlates with game length; condition on turns played for a cleaner test.
- **Simulation labels (Forge/XMage self-play): dropped.** Forge hosts returned
  502 from the sandbox, and ±1 pp per card needs ~10^4 games per card (~10^7
  per set), infeasible on CPU.
- Takeaway: Limited, Commander and Constructed strength are close to unrelated,
  so use separate format heads, not one shared strength score.

## Web app (`app/`)

Local FastAPI app plus a no-build vanilla JS frontend, CPU only.

```
pip install -r requirements.txt -r app/requirements.txt
python app/server.py [--port 8000] [--ft-dir models/ft_card] [--no-ft] [--rebuild]
```

- `app/build.py` (~1 min, run automatically when `app/data/` is missing;
  `app/data/` is git-ignored) writes `bundle.joblib`, `set_cards.parquet`,
  `cards.parquet` (every Scryfall card), `synergy.json`, `meta.json`.
- **Limited model for live scoring**: `gbt_tfidf` (167 features + 32 TF-IDF SVD
  comps, 305 rounds) refitted on all 31 labelled sets; a second small GBT on
  |out-of-fold residual| gives a card-specific SD, rescaled so OOF z-scores have
  SD 1 (80% interval coverage 0.80).
- **Estimates shown for labelled cards are out-of-sample**: test sets use the
  Phase 2 ensemble (0.5 FT Gemma-2 + 0.5 gbt_tfidf, from
  `out/phase2/test_predictions.csv`); training sets use 6-fold CV over whole sets.
- **Grades** are within-set percentiles (A+ ≥ 99, A ≥ 95, A- ≥ 90, B+ ≥ 80 …
  D- ≥ 4, F below). Observed grades rank θ; model grades rank model estimates,
  and custom cards are placed among the model estimates of the chosen set (or
  the pooled newest 8 sets, "typical").
- **Commander**: TF-IDF ridge on logit EDHREC inclusion; **Constructed**: GBT
  binary on MTGGoldfish staple status. Observed values are shown when they
  exist. Both are rough for cards outside the Limited sets.
- **Synergy**: top positive pairs from `out/phase4/interactions_*.csv` (test sets only).
- **Fine-tuned encoder**: `engine.FineTuned` loads `seed*.pt` files saved by
  `phase2.py train --save-dir` (first run found under `--ft-dir`, default
  `models/ft_card/`), runs the frozen lower layers live and averages seeds;
  custom cards, and cards without 17lands data when their page is opened
  (~0.5-1 s, cached), are then scored by 0.5 FT + 0.5 GBT. The FT features use
  the curve and set context fitted on the 24 training sets (`ft_curve`,
  `ft_ctx` in the bundle), as in `phase2.py prep`. Runs with `env="set"` are
  not supported. With `ft_embeddinggemma2_k4_saved` (3 seeds, copy from
  `/mnt/project-files/mtg_strength/models/ft_card/` to `models/ft_card/`,
  git-ignored) the app pipeline (oracle data -> features -> model) reproduces
  the saved test predictions to 8e-5 logits. Loading takes ~50 s and ~3 GB RAM.
  The search dropdown grades and set pages stay precomputed (GBT for cards
  without data; scoring all 33k Scryfall cards with Gemma would take ~3 h on
  4 CPUs). Training-set cards show the CV GBT estimate (FT is in-sample there).
- **All-cards precompute (done, final model)**: `out/app/ft_scores.parquet`
  (Limited, `g_final`, SD already rescaled) and `out/app/ft_cmd_scores.parquet`
  (Commander, `g_commander`) hold every Scryfall card, computed on a Kaggle T4
  with `python app/precompute_ft.py --device cuda --ft-dir models/ft_card/g_final`
  (and `--ft-dir models/ft_cmd/g_commander --out out/app/ft_cmd_scores.parquet`).
  `app/build.py` uses them: Limited = 0.5 Gemma + 0.5 GBT, Commander = 0.5
  fine-tuned + 0.5 ridge. Test-set cards show the out-of-sample `g_final` +
  gbt_tfidf ensemble. `engine.find_ft_weights` prefers `models/ft_card/g_final`.
  With 5 top-12 seeds the live model peaks at ~10 GB RAM; `server.py
  --ft-seeds N` loads fewer seeds (the precomputed scores are unaffected).

## Research queue (from Oct 27): one ordered plan

**All remaining research follows `docs/RESEARCH_PLAN.md`** (steps A–H, each
run only after every step that could change it: protocol → contamination →
recipe on EmbeddingGemma-2 → encoder choice → architecture/heads → final model
→ app, early update, distillation). Work happens on branch `research-queue`.
Its "Log of progress" section is the status; results land here when a step closes.

Protocol rules (binding for every new run):
- **Selection on validation only**: `phase2.py train --val-fold V1 V2 V3`
  (forward-chaining folds V1 = MKM OTJ MH3, V2 = BLB DSK FDN, V3 = TDM FIN EOE;
  each trained on the sets before it; per-epoch val predictions in
  `runs/<name>/val_pred_<fold>_seed<k>.csv`; test is never predicted).
- **Test looks are logged**: `compare_runs.py --test --why "..."` appends to
  `out/TEST_LOG.md` (earlier looks backfilled there). The final model is scored
  on test once (step G).
- **Paired stats**: `compare_runs.py RUN... --ref REF` prints Δ ± SE across
  sets, sets won, a card-level bootstrap and a verdict (real only if Δ > 2 SE
  and won on ≥ 2/3 of sets). `--blend gbt_tfidf_folds --blend-w ...` scores
  ensembles with the fold version of gbt_tfidf (`scripts/baseline_folds.py`,
  val 0.593).
- **Pre-registration**: each Kaggle job's commit message states hypotheses and
  go/no-go thresholds before it runs.

New `phase2.py` options: `--top-k -1` (full fine-tuning, no cache),
`--peft lora --lora-r R --enc-fp16`, `--amp fp16`, `--grad-ckpt`, `--llrd`,
`--optim adamw8bit`, `--weight-decay`, `--warmup`, `--dropout`,
`--text-variant {no_reminder,reorder,this_card}`; `--save-dir` also works in
fold mode (`<fold>_seed<k>.pt`). `scripts/contamination_probe.py` has the
name-only and paraphrase probes. Kaggle: `gpu/kaggle/job.py` runs one lane per
T4 in parallel and bills session time, so two lanes cost the same as one;
lane logs are kept in `gpu/kaggle/logs/`. Uploads over ~16 MB from the cloud
sandbox are cut off, so weights are produced inside the Kaggle job.

Done so far (Oct 27):
- **Weights saved**: `--save-dir`, `phase2.py predict`; run
  `ft_embeddinggemma2_k4_saved` (3 seeds, CPU) 0.626 test Spearman, weights in
  `/mnt/project-files/mtg_strength/models/ft_card/`.
- **R0 GPU parity**: GPU 0.6226 vs CPU 0.6231 (Δ −0.0005 ± 0.0033), ~50 s/epoch on a T4.
- **Screening on test (step B, untuned Gemma-2 top-4, 4 epochs, 3 seeds, vs
  matched baseline 0.626)**: env encoder +0.005 ± 0.003 (6/7), no-ctx
  +0.003 ± 0.004, LambdaRank −0.006 ± 0.003 (1/7). All ties; nothing adopted.
  Env and no-ctx are re-decided on validation in step F; LambdaRank is dropped
  (no gain on the structured or the fine-tuned model).
- **Structured-model screening** (5 seeds, `runs/s_*`): base 0.587 / NDCG
  0.800; no-ctx 0.589 / 0.811; env 0.592 / 0.810; LambdaRank 0.582–0.585.
- **Early update** (`scripts/early_update.py`, `out/early_update/REPORT.md`,
  all 7 test sets): prior = Phase 2 ensemble; data = first D days of games;
  target = labels from games after day 7. Day 1: prior 0.621, data alone
  0.633, model_prior 0.732 (NDCG@20 0.905), ATA 0.603. Days 2–7: the naive
  precision update is best (0.822, 0.851, 0.875, 0.886); model_prior falls
  ~0.02 behind by day 7 (its residual τ² over-shrinks); adding ATA hurts once
  game data exists. Product rule: ensemble pre-release, model_prior on day 1,
  naive update from day 2. **Rerun with the final prior** (`out/early_update_final/`):
  prior 0.640 (was 0.621); day 1 model_prior 0.727, day 2 naive 0.818, day 7 0.884,
  all within ±0.005 of the old run, so the product rule stands.

Decisions on validation (Oct 27-28, details in the plan's log):
- **C contamination**: no encoder flagged (name-only and paraphrase probes).
- **D recipe on EmbeddingGemma-2**: original text (no reminder expansion);
  top-12 layers (val 0.644 vs top-4 0.625; full FT no better, 2.7x slower);
  sweep of 10 configs kept the defaults; 10 epochs. Reference run `d3_c00`
  (`--top-k 12 --max-epochs 10`), val 0.649 (3 seeds × V1-V3).
- **E encoder**: Qwen3-Embedding-0.6B (full FT, lr 3e-5) 0.660 and Qwen3-4B
  (LoRA r16, lr 1e-4) 0.661 are ties vs Gemma-2 (+0.011/+0.012 < 2 SE);
  Harrier-0.6B 0.622; 8B not run. **Gemma-2 top-12 stays the backbone.**
- **F**: env encoder (+0.002 ± 0.003), no-ctx (+0.002 ± 0.006) and the
  deck-value aux head (+0.000) are ties: nothing adopted. Commander head
  fine-tuned with the same recipe adopted (val 0.795 vs ridge 0.768).
- **G final model** (`out/final/REPORT.md`): `g_final` = Gemma-2 top-12,
  10 epochs, 5 seeds on all 24 training sets. **Test (single look) 0.652**
  (old model 0.626, 7/7 sets; first-week ATA 0.652; Draftsim 0.551; model +
  ATA 0.702), NDCG@20 0.814, MAE 1.71 pp; shipped 50/50 with gbt_tfidf (0.650,
  MAE 1.66). Rolling backtest MKM-EOE 0.649 vs gbt_tfidf 0.616. Predicted SD is
  rescaled by 7.8 (fitted on validation, `scripts/calibrate_sd.py`,
  `runs/g_final/sd_scale.json`, `sd_scale` in the weights): test 90% coverage
  0.90. Commander: 50/50 fine-tuned + ridge 0.764 vs ridge 0.749 on test.
  Probes (`scripts/behaviour_probes.py`, `out/probes/`): a cheaper cost lowers
  the score for 17% of cards, so the monotonicity regularizer is the obvious
  next experiment. Weights: `/mnt/project-files/mtg_strength/models/ft_card/g_final/`
  and `.../ft_cmd/g_commander/` (130 MB per seed; copy to `models/`).

## What is not done (next steps, in rough priority)

Superseded by the ordered plan and its "parked items" table in
`docs/RESEARCH_PLAN.md`; kept for the descriptions.

1. **Constructed history**: rerun `python scripts/phase4_data.py goldfish`
   from a network that reaches web.archive.org to add the monthly MTGGoldfish
   snapshots, then `phase4.py constructed` again (only 73 test staples now).
2. **Top-of-ranking weakness**: FT model NDCG@20 0.78 vs GBT 0.83. Try a
   top-weighted ranking loss (LambdaRank-style) or make the ensemble the default.
3. **Bayesian update with early play data**: model as prior + first-week
   17lands data (model + ATA already gives 0.69). Likely the most useful product.
4. **Real environment encoder** (Set Transformer over the set pool) for the card
   head, replacing the 4 set-context features; ablate with/without (design 6).
   HOB is the hardest set for every model and both human baselines.
5. **Monotonicity regularizer** (design 4) with augmented counterfactual pairs
   (more P/T, lower MV), and **behavioural probes** (design 5.5): bigger-card
   edits, reprints in new environments, uncertainty on new mechanics. Requires
   saving the fine-tuned model weights (currently not saved).
6. **Forge script AST encoder** (design 3, "symbolic parse") and the
   text / structured / AST / all ablation. Not started: needs the Forge
   card scripts (github.com/Card-Forge/forge, `res/cardsfolder`).
7. **Reminder-text expansion and LLM keyword normalization** for unseen mechanics
   (design 7). Not started.
8. **Backfill opening-hand and archetype labels** for the 30 batch-run sets
   (~10 min/set), then an archetype-conditional head.
9. **Draft pick-order head** (design 4 multi-task) from the 17lands draft dumps.
10. **Size-vs-contamination ablation** with fine-tuned encoders of several sizes.
11. Time-varying environment embedding for Constructed metagame drift (design 7).
12. Interaction search conditioned on turns played (remove the game-length
    confound); Commander fine-tune with tuned hyperparameters and more seeds.
13. Simulation labels (Forge self-play) only if a GPU/cluster becomes available.

## Known caveats

- `user_game_win_rate_bucket` (skill proxy) includes the modelled game; 17lands
  has no player ID, so skill is a bucket, not a random effect.
- GIH labels keep some survivorship bias; compare with the opening-hand label.
- KHM–VOW exports lack colour columns: colours are inferred from decklists.
- Sets over 800k games were subsampled to 800k for labels; 200k for Phase 3.
- Differences of ~0.01 Spearman are within seed/run noise.
- Draftsim FRA ratings changed a lot between 22 and 25 Sep 2026; the 25 Sep
  snapshot is used (22 Sep kept in `data/raw/human/` for reference).
