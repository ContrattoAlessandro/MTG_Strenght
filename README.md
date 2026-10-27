# MTG card strength

A model of Magic: The Gathering card strength, trained on 17lands Limited data
and evaluated on a temporal holdout of the 7 newest sets. Design: `docs/DESIGN.md`.
**Full project context, results per phase, how to rerun and what is left: `CLAUDE.md`.**
Setup: `pip install -r requirements.txt` (CPU builds of torch are fine).

Large files are not in git (Scryfall bulk data, 17lands dumps, frozen embeddings,
fine-tuned encoder weights); `CLAUDE.md` says how to regenerate each one.

## Web app: "is this card good?"

A local web app (CPU only) on top of the models: search any card, see its
Limited grade within its set (measured on 17lands and predicted by the model,
with an 80% range), Commander and Constructed signals, synergy partners,
set tier lists, and a Card Lab that grades a card you type in.

```
pip install -r requirements.txt -r app/requirements.txt
python app/server.py            # open http://127.0.0.1:8000
```

The first start builds `app/data/` (about a minute; it downloads the Scryfall
oracle file if missing). `python app/server.py --rebuild` rebuilds it after
new labels or model outputs. To use fine-tuned EmbeddingGemma-2 weights for
custom cards, put the `seed*.pt` files from `scripts/phase2.py train --save-dir`
under `models/ft_card/<run>/` (or pass `--ft-dir`); this needs torch,
transformers and access to `google/embeddinggemma-2` on Hugging Face. Without
them the app uses the gradient-boosting model. Details: `CLAUDE.md`, "Web app".

## Phase 0: data pipeline

Per-set debiased card-strength labels from the public 17lands game data,
following section 2 of the design.

```
pip install numpy pandas scipy pyarrow
python scripts/phase0.py --set ECL --data-dir data --out-dir out
```

The script downloads `game_data_public.<SET>.PremierDraft.csv.gz` and
`cards.csv` from the 17lands public S3 bucket, fits the label model, tries the
Scryfall oracle-ID join, and writes to `out/<SET>/`. A run on ECL (536k games)
takes about 11 minutes on 4 CPUs and peaks near 4 GB of RAM.

## Model

One hierarchical logistic regression per set, one row per game:

```
logit P(win) = nuisance + Σ_c θ_c·in_hand_c + Σ_c θ'_c·deck_copies_c
               + Σ_{c,a} η_{c,a}·in_hand_c·[archetype = a]
```

- **Nuisance terms**: player skill (logit of the 17lands win-rate bucket, with
  a separate slope per games-played bucket), rank, main colours, splash,
  opponent colours, play or draw, own and opponent mulligans.
- **Partial pooling**: θ_c = μ_{rarity×colour} + δ_c with δ_c ~ N(0, τ²). Same
  for θ'. η is pooled towards zero. Each τ² is set by empirical Bayes (EM on
  the Laplace approximation).
- **Variances** come from the inverse Hessian at the MAP, so a card's SD
  includes the uncertainty in its group mean.
- **Replacement level** is the median θ of regularly played commons
  (≥ 2,000 games in hand). `theta_vs_repl` is the label to train on.
- **Opening-hand variant**: the same model with "in opening hand" instead of
  "in hand". It is free of the long-game survivorship bias but noisier.

## Outputs (`out/<SET>/`)

`card_labels.csv`, one row per card:

| column | meaning |
|---|---|
| `theta_vs_repl`, `theta_hand_sd` | main label (logit scale) and posterior SD |
| `precision_weight` | 1 / SD², the per-sample training weight |
| `ame_vs_repl_pp` | the same effect in win-rate points, averaged over games where the card was in hand |
| `theta_deck` | deck-inclusion effect θ' (selection signal) |
| `theta_oh_vs_repl`, `theta_oh_sd` | opening-hand robustness label |
| `raw_wr_in_hand`, `raw_iwd` | raw 17lands-style GIH WR and IWD, for comparison |
| `oracle_id` | Scryfall oracle ID (empty when Scryfall was unreachable) |
| `match` | how the name joined: `full` name or `front` face (DFCs) |

`card_archetype_labels.csv`: η (deviation) and θ+η per card and two-colour
archetype, for pairs with ≥ 500 games in hand. `summary.json`: fitted prior
variances, nuisance coefficients, convergence history.

## Known limitations

- `user_game_win_rate_bucket` includes the game being modelled. The leak is
  small for players with many games but inflates the skill slope for the
  low-games buckets.
- The GIH label still carries some survivorship bias (cards seen in long
  games). Compare against the opening-hand label.
- 17lands data has no player ID, so skill is a bucketed proxy, not a
  per-player random effect.
- `eta_sd` in the archetype table ignores the covariance with θ_c.

## Phase 1 (card-only baselines)

Labels now exist for 31 sets (KHM to FRA). `scripts/phase1.py` trains the
rarity-only, GBT, TF-IDF and frozen-embedding baselines on a temporal
holdout (test: TLA onward). Results and details: `out/phase1/REPORT.md`.

## Phase 2 (fine-tuned card encoder)

`scripts/phase2.py` fine-tunes a text encoder together with the structured
features, using a precision-aware Gaussian NLL plus a within-set pairwise
loss. It also runs frozen-embedding baselines for newer encoders.
`mtgstrength/partial.py` fine-tunes only the top layers of a large encoder,
which makes it trainable on CPU. `scripts/human_baselines.py` builds the
human baselines: Draftsim pre-release ratings and first-week ATA.

Best result: EmbeddingGemma-2 fine-tuned, with 0.62 test Spearman. Results,
contamination check and human comparison: `out/phase2/REPORT.md`.

## Phase 3 (deck model, counterfactual values, distillation)

`scripts/phase3_data.py` turns the 17lands game dumps into per-set deck
arrays (200k games per set). `scripts/phase3.py` trains a set-transformer
win model on decks from the 24 training sets, computes counterfactual card
values on real and synthetic decks, and compares them to θ. Distillation into
the card head uses `scripts/phase2.py train --deck-values out/phase3/deck_values.csv`
with `--target deck`, `--aux-w` or `--consistency-on-test`. Trained deck model:
`models/deck_model.pt`.

Result: deck values agree with θ (0.86 on training sets, 0.54 on new sets
from synthetic decks), but distillation ties plain fine-tuning (0.624 vs
0.621). Details: `out/phase3/REPORT.md`.

## Phase 4 (Commander, Constructed, interactions)

`scripts/phase4_data.py` fetches EDHREC inclusion and MTGGoldfish staples.
`scripts/phase4.py commander|constructed` fits and scores the format heads.
`scripts/phase4_interactions.py` scores card-pair synergy with the deck model
and validates it on held-out games.

Result: Limited strength barely transfers to other formats; text predicts
Commander play (0.75) and Constructed staples (AUC 0.81) when trained on
their own labels; pair synergy correlates 0.22 with held-out empirical
interactions. Simulation labels were not feasible here. Details:
`out/phase4/REPORT.md`.
