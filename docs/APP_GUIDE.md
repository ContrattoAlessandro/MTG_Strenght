# Card Strength Oracle: install and use the web app

The app runs locally, on CPU, and contains everything the research adopted:

- **Limited**: the final model `g_final` (EmbeddingGemma-2, top 12 layers fine-tuned,
  5 seeds) averaged 50/50 with gradient boosting (`gbt_tfidf`). Test Spearman 0.65,
  the same as the drafting crowd after one week (first-week ATA), better than
  Draftsim pre-release ratings on 6 of 7 sets. Every Scryfall card (33k) is
  pre-scored with it, so search and set pages are instant.
- **Uncertainty**: an 80% range for every estimate, calibrated on validation
  (90% intervals cover 90% of test cards).
- **Commander**: fine-tuned head + TF-IDF ridge, 50/50 (test 0.76 vs 0.75 ridge alone).
- **Constructed**: probability of being a MTGGoldfish staple (rough).
- **Synergy**: best partner cards from the deck model (new sets only).
- **Card Lab**: grade a card you type in, live, with the same models.

Not adopted (tested on validation, no measurable gain): environment encoder,
dropping set context, deck-value distillation, LambdaRank, bigger encoders
(Qwen3 0.6B/4B tie). Full results: `out/final/REPORT.md`.

## 1. Install (once)

In PowerShell, from the repo folder (`C:\Users\alexa\Desktop\MTG_Strenght\repo`):

```powershell
git fetch origin
git checkout research-queue
git pull origin research-queue
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r app/requirements.txt
```

That is enough to run the app with the pre-computed scores (step 3). Custom
cards in the Card Lab then use gradient boosting alone.

## 2. Optional: the fine-tuned model for custom cards

Needed only to score **cards you type in** with the full model (everything
else is already pre-scored).

1. Get the weights (about 130 MB per seed). Either download them from the
   project Library (`mtg_strength/models/ft_card/g_final/` and
   `mtg_strength/models/ft_cmd/g_commander/`), or from Kaggle:

   ```powershell
   kaggle kernels output alessandrocontratto/mtg-strength-gpu -p kaggle_out
   ```

   then copy `results/models/ft_card/g_final` and `results/models/ft_cmd/g_commander`.
   Note: the Kaggle copies of `g_final` lack the SD calibration stamp; run
   `python scripts/calibrate_sd.py --val-run d3_c00 --epoch 10 --test-run g_final --weights models/ft_card/g_final`
   once on them, or use the Library copies, which have it.

2. Put them here (the folder names matter):

   ```
   models/ft_card/g_final/seed0.pt ... seed4.pt
   models/ft_cmd/g_commander/seed0.pt ... seed2.pt
   ```

3. EmbeddingGemma-2 is gated on Hugging Face: accept the licence on its model
   page, then `hf auth login` (once). After the first download you can start
   offline with `$env:HF_HUB_OFFLINE=1`.

The app needs the CPU build of torch (already in your venv if you followed
CLAUDE.md). It does not use your GPU.

## 3. Run

```powershell
python app/server.py
```

Open http://127.0.0.1:8000. The first start builds `app/data/` (about a
minute; downloads the Scryfall card file if missing). Then the models load:

| setup | start time | RAM |
|---|---|---|
| no weights, or `--no-ft` | a few seconds | 1-2 GB |
| `--ft-seeds 1` | ~20 s | ~6 GB |
| all 5 seeds (default) | ~50 s | ~10 GB |

On a 16 GB PC use `python app/server.py --ft-seeds 2`. Fewer seeds only
affect custom cards: the pre-computed scores always use all 5.

Other flags: `--port 8001`, `--rebuild` (rebuild `app/data/` after you pull
new outputs), `--threads N`. The status line at the bottom of each page says
which model is active ("fine-tuned encoder + GBT" or "gradient boosting").

## 4. What each page does

**Home / search.** Type any card name (all Scryfall cards, not only Limited
sets). The dropdown shows the card's Limited grade. "🎲 Random card" opens one.

**Card page.**
- *Grade*: the card's percentile in its set, A+ (top 1%) to F. If the card has
  17lands data you see two grades: **measured** (from games) and **model**
  (predicted without the card's games: pre-release prediction for new sets,
  held-out cross-validation for older ones). Cards without data get the model
  grade only.
- *Tiles*: win-rate points vs a replacement-level common, the model estimate
  with its 80% range, raw GIH win rate and number of games, and for new sets
  what humans said (Draftsim rating, first-week pick position).
- A chart of the set's distribution with the measured value and the model range.
- *Set tabs*: if the card was in several Limited sets, one tab per set.
- *Best partners*: cards it wins more with, from the deck model (new sets).
- *Around it in the ranking*: the cards just above and below it in its set.
- *Other formats*: Commander inclusion (observed EDHREC rate when known, else
  the model) and Constructed staple probability.
- *Test in another set*: copies the card into the Card Lab, where you can
  edit it or change the environment.

**Sets.** All 31 Limited sets; "new" marks the 7 sets released after the
training data. A set page has:
- *Tier list*, *Table* and *Model vs games* (scatter of predicted vs measured).
- A switch to grade by **17lands data** or by **Model**.
- Filters by colour, rarity and name/type.
- In the table, for new sets: Draftsim pre-release rating and first-week ATA,
  so you can compare the model with the humans.

**Card Lab.** Type a card (name, cost such as `{2}{W}{W}`, rarity, type line,
rules text, P/T or loyalty) or start from an existing card, choose the Limited
environment (a specific set, or "typical recent set"), and press *Grade this
card* (Ctrl+Enter). You get the grade and range in that environment, where it
lands among that set's cards, real cards the model rates the same, and the
Commander and Constructed estimates. Useful for "what if" edits: change the
cost or P/T and compare.

**How it works.** A one-page summary of the labels, the model, grades and limits.

## 5. Limits worth knowing

- The model predicts the measured win-rate effect; it is as good as the
  crowd's first week, not as good as the measured data. Once a set has a few
  days of 17lands games, trust the measured grade.
- It is less reliable on very unusual cards and new mechanics (its range is
  only slightly wider there). A cheaper version of a card sometimes scores
  lower (16% of cards in the probes), so treat small cost edits in the Card
  Lab with care.
- Commander and Constructed estimates are rough, especially for cards outside
  the Limited sets. Limited strength says almost nothing about either format.

## 6. Early update (not in the app)

For a new set, the best ranking over time is: model before release; on day 1
the model as a prior combined with day-1 17lands data; from day 2 the plain
data-weighted update. `scripts/early_update.py` implements it (see
`out/early_update_final/REPORT.md`). Adding it to the app is a possible next
step.
With the final model: before release 0.64, day 1 0.73, day 2 0.82, day 7 0.88
(Spearman vs the final labels).
