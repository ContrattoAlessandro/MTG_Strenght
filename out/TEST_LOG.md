# Test-set access log

One line per look at the 7 test sets (TLA ECL TMT SOS MSH HOB FRA), as required by
docs/RESEARCH_PLAN.md rule 2. From 2026-10-27 lines are appended by
`scripts/compare_runs.py --test --why ...`; earlier looks are backfilled below.

## Backfill (looks before the protocol, Oct 2026)

Every run below was scored on the test sets at least once, mostly through
`phase2.py evaluate` / `compare_runs.py` before validation folds existed.
Several design choices were made after seeing these numbers, so the 7 test
sets are **not pristine**: the final claim in step G is backed by the
rolling-origin backtest (11 sets never used for any choice) as well.

| phase | runs scored on test | choices informed by them |
|---|---|---|
| 1 | rarity_only, gbt, tfidf, gbt_tfidf, lm_mpnet_base, lm_bge_base, lm_qwen3_emb_0_6b | gbt_tfidf as baseline and ensemble partner |
| 2 | zs_* (mpnet, bge, Qwen3-0.6B, EmbeddingGemma-2, Harrier-0.6B, Nemotron-3.5-1B), ft_embeddinggemma2_k4, _text, struct_mlp, ft_harrier_0_6b_k2, contamination table | EmbeddingGemma-2 as reference; "fine-tune, don't scale frozen" |
| 2 | human baselines (Draftsim pre-release, ATA week 1) | none (reference only) |
| 3 | deck_synth, ft_embeddinggemma2_k4_deckonly, _aux, _auxcons, deck values | "distillation adds nothing" |
| 4 | ft_embeddinggemma2_k4_commander, Commander/Constructed heads, interaction search | TF-IDF ridge as Commander head |
| Oct 27 | ft_embeddinggemma2_k4_saved; s_base, s_noctx, s_env, s_env_ctx, s_env_emb, s_fixvar, s_fixvar_noctx_env, s_lam05, s_lam09_k20, s_lam1, s_lam1_k20, s_lam1_pw3 | none adopted (all ties) |
| Oct 27 | ft_embeddinggemma2_k4_gpu (R0, GPU parity) | none (pipeline check) |
| Oct 27 | early_update (TLA, ECL, TMT so far) | none yet |

## Looks from 2026-10-27 on

| date (UTC) | runs | why |
|---|---|---|
| 2026-10-27 14:33 | ft_embeddinggemma2_k4, ft_embeddinggemma2_k4_gpu, ft_embeddinggemma2_k4_saved | R0 GPU parity check (also tests the new compare_runs.py) |
