# Phase 3 report: deck model, counterfactual card values, distillation

All work ran on CPU. Same temporal split as Phases 1 and 2: games and labels
from the 24 training sets (KHM..EOE) only; the 7 test sets (TLA, ECL, TMT,
SOS, MSH, HOB, FRA) are scored against their debiased GIH label θ by
within-set Spearman. Code: `scripts/phase3_data.py`, `scripts/phase3.py`,
distillation through `scripts/phase2.py train --deck-values ...`.

## 1. Deck-level win model

P(win) of a 17lands Premier Draft game from its maindeck (`scripts/phase3.py`
docstring has the full spec). Each card is a token built from its features
(Phase 1 engineered features + frozen EmbeddingGemma-2 text embedding), a
per-card ID embedding dropped 50% of the time in training, and "seen" and
copy-count embeddings. A set environment vector (mean-pooled set pool) is
added. Two transformer layers, attention pooling, MLP; skill, games played,
play/draw and mulligans enter as linear nuisance terms. 200k games per set,
2 epochs, about 2 hours on CPU. Saved weights: `models/deck_model.pt` (36 MB).

| held-out games (3% of training-set games) | log loss |
|---|---|
| constant | 0.688 |
| nuisance terms only | 0.666 |
| deck model, no ID embeddings | 0.643 |
| deck model, with ID embeddings | 0.642 |

The deck carries real signal beyond player skill, and almost all of it goes
through the card features rather than the card IDs, which is what lets the
model score cards from sets it never saw a game of.

## 2. Counterfactual card values

Value of a card = mean ΔP(win) from swapping it in, seen, for a random
non-land card, over decks that can cast it. Agreement with θ (mean within-set
Spearman, `deck_values_vs_theta.csv`):

| deck source | training sets | test sets |
|---|---|---|
| real decks, with ID | 0.861 | n/a |
| real decks, no ID | 0.804 | n/a |
| synthetic decks from the pool, no ID | 0.751 | **0.541** |

On the test sets the synthetic-deck value (`deck_synth`, no game data from
those sets) scores 0.521 on the full Phase 2 metric set, against 0.515 for
zero-shot EmbeddingGemma-2 and 0.421 for rarity alone. The drop from 0.75 on
training sets to 0.54 on new sets is the cost of new mechanics the features
have not seen played.

## 3. Distillation into the card head

The Phase 2 card model (EmbeddingGemma-2, top 4 layers tuned, plus
structured features) was fine-tuned again with the deck values as targets.
Two seeds each, 4 epochs. Phase 2 FT is shown with the same two seeds for a
fair comparison (`distill_per_set_spearman.csv`).

| variant | mean Spearman (7 test sets) |
|---|---|
| Phase 2 FT, θ only (seeds 0,1) | 0.621 |
| **θ + auxiliary deck-value head (aux 0.5)** | **0.624** |
| θ + aux + consistency on unlabeled test cards | 0.614 |
| deck values only, no θ at all | 0.601 |
| deck model synthetic values, not distilled | 0.521 |

- The auxiliary deck-value head is a tie with plain FT: +0.003 on average,
  better in 4 of 7 sets, inside seed noise (seeds differ by about 0.005). It
  does not hurt, and it gives a second head that estimates ΔP(win).
- Consistency on the test cards (fitting the head to their synthetic deck
  values) hurts slightly, as the synthetic values on new sets are noisier
  than the model's own prediction.
- Distilling deck values alone, with no θ, reaches 0.601, close to the full
  model. Note this is not a label-free result: the training-set deck values
  come from the same games as θ, so it shows the deck model's counterfactual
  is a good substitute label, not that game data is unnecessary.
- Distillation lifts the deck model's own test score from 0.521 to 0.601:
  the text encoder generalizes the deck model's judgments better than the
  deck model does on new cards.

## 4. Conclusion

The deck model works and its counterfactual values agree with θ, but on new
sets it does not beat a card model trained on θ directly, and distilling it
adds nothing measurable. The best test-set predictor remains the Phase 2
ensemble (FT + GBT/TF-IDF, 0.627 to 0.629), still behind first-week ATA
(0.652) but ahead of Draftsim pre-release grades (0.551). The deck model is
used in Phase 4 to search for card interactions.
