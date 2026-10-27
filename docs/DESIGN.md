# Designing a model of MTG card strength

*Original design by Alessandro (2026-10-26). This is the project's spec; see `CLAUDE.md` for what has been built.*

The hard part isn't the architecture. It's that "strength" isn't a property of a card alone. Llanowar Elves is weak in Commander, strong in Standard, and different again in a slow Limited format. So the design starts by pinning down the target. Then it builds a model that separates what is intrinsic to the card from what depends on its context.

## 1. Defining the target

I'd define strength operationally:

> **Strength of card *c* in environment *E*** = the expected change in win probability when *c* replaces a replacement-level card in a typical deck that can play it, in environment *E*.

This is measurable, it's in the right units (win rate), and it handles synergy naturally because it is averaged over realistic decks.

Possible data sources, ranked by how useful they are as labels:

| Source | Signal | Problem |
|---|---|---|
| 17lands Limited game data | Dense: about 250–300 cards per set, millions of games, decklists plus outcomes | Confounded by player skill, colour pair, and when the card was drawn |
| Constructed tournament results (decklists, play rates) | The true "outlier power" signal | Very sparse. About 95% of cards are never played, so the labels are censored |
| EDHREC inclusion rates | Large coverage | Measures popularity and synergy, not power |
| Prices | Easy to get | Driven by rarity, collectability and reprints |

**Recommendation:** use Limited as the primary supervised target, because it's the only dense source with outcomes. Treat constructed play rate and Commander inclusion as auxiliary tasks on a shared encoder.

## 2. Label engineering: debiasing win rates

Raw games-in-hand win rate (GIH WR) is a biased label. Strong players draft certain cards, some colour pairs are overpowered, and cards seen only in long games pick up survivorship effects. I'd fit a hierarchical logistic model per set:

```
logit P(win_g) = α_skill(player) + β_archetype(deck colours) + γ·on_play
                 + Σ_c θ_c · drawn_{c,g} + Σ_c θ'_c · in_deck_{c,g}
```

- θ_c is the "drawn" effect, which is the main label. It has a posterior variance, and that variance becomes a per-sample weight during training.
- Partial pooling across cards with the same rarity and colour keeps rare cards with few observations stable.
- Archetype-conditional effects θ_{c,arch} make a secondary label, which captures "great in UB, unplayable in RW".

## 3. Model architecture

There are two levels. A **deck model** learns from games. Card strength is then read out by intervening on decks. A **card model** distills that readout so it can score unseen cards, such as spoiler-season cards before any games exist.

```
 card text ──► Text encoder (LM, fine-tuned) ─┐
 structured ─► MLP (MV, pips, P/T, type,     ├─► card embedding  e_c
              rarity, keywords)               │
 Forge script AST ─► tree/graph encoder ──────┘
                                                 │
 all cards in format pool ─► Set Transformer ──► environment embedding  z_E
                                                 │
            ┌────────────────────────────────────┴───────────────┐
  Deck model: Set Transformer over {e_c} of a 40-card deck,     Card head: f(e_c, z_E) → θ̂_c, σ̂_c
  conditioned on z_E → P(win)                                    (+ archetype, constructed, EDH heads)
```

**Card encoder**
- **Rules text.** Replace the card's name with `CARDNAME` and expand reminder text. Use a pretrained language model, because it carries useful prior knowledge, such as what "ward 2" means.
- **Structured features.** Include engineered efficiency features, for example how a creature's stats compare with the vanilla stats-per-mana curve for that format and colour.
- **Symbolic parse.** Forge's card-script DSL covers almost every printed card. Encoding its abstract syntax tree gives a compositional view of effects ("deal N damage to target creature") that generalizes better than surface text.

**Environment encoder.** This module attends over every card in the format, so it learns things like removal density, average creature size, and the speed of the format. It's what lets the model learn that a 2/2 for two is fine in one set and filler in another. Each set is one "environment", which gives about 100 environments since 2019.

**Deck model.** This is trained on hundreds of millions of decklist→outcome pairs. Card strength is then estimated by counterfactual swaps: Δ P(win) from inserting *c* instead of a replacement-level card, averaged over sampled realistic decks that can cast it. This captures synergy, which a card-only model cannot see.

**Output.** A mean plus heteroscedastic uncertainty. A new mechanic should produce wide error bars, not confident nonsense.

## 4. Training objectives

- Gaussian negative log-likelihood on θ_c, weighted by the label's posterior precision.
- **Within-set pairwise ranking loss** (Bradley–Terry). Ranking inside a set is what drafters need, and it is robust to shifts in the absolute win-rate scale between sets.
- **Consistency loss.** The card head's prediction should match the deck model's counterfactual estimate. Applying this to unlabeled cards adds extra supervision.
- **Monotonicity regularizers.** Raising power or toughness, or lowering mana value, should not lower predicted strength, all else equal. Enforce this with augmented counterfactual pairs.
- **Multi-task heads** for constructed play (a censored or ordinal model: "below threshold" is information, not a missing label), EDHREC inclusion, and draft pick order.

## 5. Evaluation protocol

Evaluation is where most published attempts at this go wrong.

1. **Temporal set holdout only.** Train on sets before *T* and test on the newest sets. Random splits leak through reprints (group by oracle ID) and near-functional duplicates.
2. **LLM contamination control.** If the text encoder is a pretrained LLM, it has probably read tier lists and 17lands discussions for older sets. Final test sets must be released after the base model's training cutoff. Report results on pre-cutoff sets separately and treat them as optimistic.
3. **Metrics:**
   - Within-set Spearman correlation.
   - Ranking quality on the top cards (NDCG@20), because identifying bombs matters most.
   - Mean absolute error in win-rate percentage points.
   - Calibration of the predicted uncertainty.
4. **The baseline that actually matters is human experts.** Compare against professional pre-release set reviews and community tier grades made before any game data existed. Beating ATA (average taken at) from the first week of drafting would be a strong result.
5. **Behavioural probes:**
   - Counterfactual edits: does strength go up when you make a card bigger?
   - Reprints in new environments: does the context encoder move the prediction the right way?
   - Brand-new mechanics: is the model's uncertainty high?

## 6. Ablations to run

- Text-only vs. structured-only vs. Forge AST vs. all three.
- With vs. without the environment encoder. I'd expect large gains for commons and small ones for bombs.
- Card head trained directly on θ_c vs. distilled from the deck model.
- Size of the language model behind the encoder vs. the contamination-safe test score. This shows whether larger models actually understand card power or just remember it.

## 7. Risks and open problems

- **Unseen mechanics.** Mitigate with reminder-text expansion and by having an LLM rewrite new keywords into known primitives as an input-normalization step.
- **Constructed outliers.** Broken cards come from rare interactions, such as combos and mana-cost cheats, that Limited data barely contains. A card-level model will systematically under-predict these. This is probably a separate phase that searches for interactions over pairs of cards.
- **Metagame non-stationarity.** Constructed strength drifts. Model it as a time-varying environment embedding.
- **Optional, expensive phase 3.** Self-play simulation in Forge or XMage to label hypothetical cards. This is useful for design-space exploration, but the labels inherit the weaknesses of the simulated players.

## 8. Roadmap

| Phase | Weeks | Deliverable |
|---|---|---|
| 0. Data | 1–3 | Joined Scryfall/MTGJSON cards, Forge scripts and 17lands games; per-set debiased θ_c with variances |
| 1. Baselines | 3–4 | Gradient-boosted trees on engineered features; frozen LM embeddings + ridge regression. Sets the bar |
| 2. Card model | 4–8 | Full card encoder + environment encoder, temporal evaluation, comparison against human reviews |
| 3. Deck model | 8–14 | Deck-level win model, counterfactual card values, distillation into the card head |
| 4. Extensions | 14+ | Constructed and Commander heads, interaction search, simulation labels |

The single most informative early experiment is Phase 1 against pre-release human grades on a post-cutoff set. It tells you quickly whether there's a real signal or whether you're mostly measuring rarity.
