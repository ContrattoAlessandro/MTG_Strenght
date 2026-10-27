# Phase 4 report: Commander and Constructed heads, interaction search, simulation

All on CPU, same temporal split (train on cards first printed in the 24
training sets, test on the 7 newest). Code: `scripts/phase4_data.py`,
`scripts/phase4.py`, `scripts/phase4_interactions.py`.

## 1. Commander head

Label: logit of EDHREC inclusion (decks playing the card / decks whose colour
identity allows it), fetched for 8,128 cards, 8,046 with data, 1,695 in the
test sets. Mean within-set Spearman over the 7 test sets
(`commander_metrics_per_set.csv`):

| model | Spearman |
|---|---|
| rarity only | 0.465 |
| Limited θ (true label) | 0.222 |
| Phase 2 Limited model prediction | 0.219 |
| GBT on engineered features | 0.672 |
| GBT + TF-IDF | 0.684 |
| EmbeddingGemma-2 fine-tuned on Commander label (1 seed) | 0.717 |
| **ridge on TF-IDF of the rules text** | **0.749** |

Limited strength barely transfers to Commander (0.22, below rarity). Rules
text predicts Commander play well, and here simple TF-IDF beats the
fine-tuned encoder, likely because Commander play is driven by keywords and
tribal or archetype words that TF-IDF captures directly. The fine-tune used
the Limited hyperparameters and one seed (a second seed was cut for time).

Caveat: EDHREC counts are a current snapshot, so recent cards (test sets)
have had less time to be adopted than older ones. Within-set ranking is
unaffected by a shared offset, but adoption speed may differ by card type.

## 2. Constructed head

Label: MTGGoldfish format staples (top 50 per format and category, for
Standard, Pioneer, Modern, Legacy, Pauper). The archived history (Wayback
Machine) was unreachable from this environment (connection resets), so only
the live pages were used: 20 lists, 333 staples in our card pool (260 train,
73 test). Unlisted cards are censored below each list's cut-off (median 6%
of decks). Scores (`constructed_metrics.csv`):

| model | AUC within set | avg. precision | Spearman among staples |
|---|---|---|---|
| rarity only | 0.651 | 0.071 | -0.21 |
| Limited θ | 0.653 | 0.103 | -0.19 |
| Phase 2 Limited model | 0.588 | 0.071 | -0.04 |
| GBT, binary | **0.810** | 0.227 | 0.17 |
| GBT, censored | 0.808 | 0.233 | **0.20** |
| logistic on TF-IDF | 0.786 | **0.274** | 0.18 |

Limited strength says nothing about Constructed beyond rarity. Features
trained on the Constructed label do identify staples (AUC 0.81), but with 73
test positives and a single current snapshot this is a rough result; the
ordering among staples is weak. Running `phase4_data.py goldfish` from a
network that reaches web.archive.org adds the monthly history.

## 3. Interaction search

The Phase 3 deck model (never trained on a test-set game) scores pair
synergy as the 2×2 counterfactual S(a,b) = P(D+a+b) − P(D+a) − P(D+b) + P(D)
over 64 synthetic decks per colour pair, for all pairs of the 40 best cards
of each colour pair. Each candidate was checked against the test sets' real
games: a logistic interaction b_ab (both cards seen) fitted on the games
where both are in the deck (at least 1,500 games). 54,600 scored pairs,
9,554 unique pairs with an estimate (`interactions_summary_dedup.csv`):

| set | unique pairs | weighted corr(S, b_ab) | b_ab top decile | b_ab bottom decile |
|---|---|---|---|---|
| ECL | 1200 | 0.196 | +0.037 | -0.070 |
| FRA | 1345 | 0.151 | +0.061 | -0.027 |
| HOB | 1445 | 0.284 | +0.030 | -0.089 |
| MSH | 1241 | 0.134 | +0.038 | -0.047 |
| SOS | 1567 | 0.193 | +0.057 | -0.065 |
| TLA | 1229 | 0.273 | +0.031 | -0.103 |
| TMT | 1527 | 0.291 | +0.069 | -0.096 |
| mean | | **0.217** | +0.046 | -0.071 |

Every set's correlation is significant (permutation p = 0.002, the minimum
for 500 permutations), and in every set the pairs the model likes most show
a positive interaction in held-out play while its least liked pairs show a
negative one. Examples of high-synergy pairs with a strong empirical
interaction: Together as One + Studious First-Year (SOS, b_ab +0.31),
Old Fat Spider + Troop of Ponies (HOB, +0.44), Garruk, Curse Breaker +
Arcane Amphisbaena (FRA, +0.40). Per-pair details are in
`interactions_<SET>.csv`.

Caveat: "both seen" correlates with longer games, and the deck model was
trained on the same seen flags, so part of the agreement may be a shared
game-length effect (for example two expensive cards) rather than synergy.
The average b_ab over all pairs is about zero (−0.008), so it is not a
uniform bias, but a version that conditions on turns played would be
cleaner.

## 4. Simulation labels

Not done. The candidate engine (Forge) needs its distribution hosts, which
returned 502 from this environment, and on CPU the cost is prohibitive: a
±1 pp estimate of a card's win-rate effect needs on the order of 10^4
simulated games per card, about 10^7 games per set. This is left for a
machine with the engine installed.

## 5. Summary

- Limited, Commander and Constructed strength are close to unrelated:
  Limited θ correlates 0.22 with Commander play and gives no lift over rarity
  for Constructed staples.
- Rules text predicts each format well when trained on that format's label
  (Commander 0.75, Constructed staples AUC 0.81), so the multi-task design
  should use separate heads, not a shared strength score.
- The deck model finds real interactions in held-out sets (correlation 0.22
  with the empirical interaction, positive top decile in all 7 sets).
