"""Phase 4: interaction search (design section 7, "search for interactions
over pairs of cards").

    python scripts/phase4_interactions.py --sets TLA ECL FRA

For each set, the Phase 3 deck model (trained on the 24 training sets only,
so it has never seen a game of a test set) scores card pairs by their
synergy, the interaction term of a 2x2 counterfactual:

    S(a, b) = E_D[ P(D+a+b) - P(D+a) - P(D+b) + P(D) ]

over synthetic two-colour decks built from the set pool (as in Phase 3),
with both cards seen. Candidates are, for each colour pair, all pairs of
the 40 cards with the highest single-card deck value.

Validation on held-out game data (the test sets' real games): for each
candidate pair that co-occurs in enough decks, an empirical interaction is
fitted on the games where both cards are in the deck,

    logit P(win) = skill terms + b_a seen_a + b_b seen_b + b_ab seen_a seen_b

and the model's synergy is correlated with b_ab, weighted by its precision.
The pairs the model ranks highest should show positive empirical b_ab.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import phase3 as p3  # noqa: E402

RES = "phase4"


@torch.no_grad()
def predict(model, D, idx, env):
    return torch.sigmoid(model(*p3.batch_tensors(D, idx), use_ids=False, env=env)).numpy()


def with_cards(D: dict, extra: list[int]) -> dict:
    """Copy of the decks with each card in ``extra`` added (seen), replacing
    one random non-land card per added card (the same slots every time, so
    the four arms of the 2x2 differ only in the inserted cards)."""
    E = {k: v.copy() for k, v in D.items()}
    for j, r in enumerate(extra):
        slot = D["_slots"][:, j]
        E["tok"][np.arange(len(slot)), slot] = r
        E["cnt"][np.arange(len(slot)), slot] = 1
        E["seen"][np.arange(len(slot)), slot] = 1
    return E


def synergy_scores(model, cards, s, env, n_decks, top_n, rng):
    vals = pd.read_csv(os.path.join(args.out_dir, "phase3", "deck_values.csv"))
    vals = vals[vals["set"] == s].set_index("oracle_id")["value_synth"]
    pool = cards[(cards["set"] == s) & ~cards["is_land"]].copy()
    pool["v"] = pool["oracle_id"].map(vals)
    rows = []
    for pair in p3.PAIRS:
        ok = pool[pool["colors"].map(lambda c: set(c) <= set(pair))].sort_values("v", ascending=False)
        cand = ok.index[:top_n].to_numpy()
        D = p3.synth_decks(cards, s, n_decks, rng)
        # Make every deck this colour pair.
        sub = pool[pool["colors"].map(lambda c: set(c) <= set(pair))].index.to_numpy()
        for i in range(n_decks):
            pick = rng.choice(sub, size=min(23, len(sub)), replace=False)
            D["tok"][i] = -1
            D["tok"][i, :len(pick)] = pick
            D["cnt"][i, :len(pick)] = 1
        # Two distinct replacement slots per deck.
        D["_slots"] = np.stack([rng.permutation(23)[:2] for _ in range(n_decks)])
        idx = np.arange(n_decks)
        base = predict(model, D, idx, env)
        single = {}
        for a in cand:
            single[a] = predict(model, with_cards(D, [a]), idx, env)
        for a, b in itertools.combinations(cand, 2):
            both = predict(model, with_cards(D, [a, b]), idx, env)
            # P(D+a) with a in slot 0; P(D+b) needs b in slot 1 to match the 2x2.
            Db = {k: v.copy() for k, v in D.items()}
            Db["_slots"] = D["_slots"][:, 1:2]
            pb = predict(model, with_cards(Db, [b]), idx, env)
            S = float(np.mean(both - single[a] - pb + base))
            rows.append({"set": s, "pair": pair, "a": a, "b": b, "synergy": S,
                         "name_a": cards.at[a, "name"], "name_b": cards.at[b, "name"]})
    return pd.DataFrame(rows)


def empirical(G: dict, pairs: pd.DataFrame, min_games: int = 1500) -> pd.DataFrame:
    """Interaction b_ab from games where both cards are in the deck."""
    import statsmodels.api as sm
    tok, seen = G["tok"], G["seen"]
    out = []
    for r in pairs.itertuples():
        ia = (tok == r.a)
        ib = (tok == r.b)
        both = ia.any(1) & ib.any(1)
        if both.sum() < min_games:
            out.append((np.nan, np.nan, int(both.sum())))
            continue
        sa = (ia & (seen > 0)).any(1)[both].astype(float)
        sb = (ib & (seen > 0)).any(1)[both].astype(float)
        X = np.column_stack([np.ones(both.sum()), G["skill"][both], G["on_play"][both], sa, sb, sa * sb])
        try:
            m = sm.Logit(G["won"][both].astype(float), X).fit(disp=0)
            out.append((m.params[-1], m.bse[-1], int(both.sum())))
        except Exception:
            out.append((np.nan, np.nan, int(both.sum())))
    e = pd.DataFrame(out, columns=["b_ab", "b_ab_se", "n_both"], index=pairs.index)
    return pairs.join(e)


def main() -> None:
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--work-dir", default="/tmp/claude-0/p3")
    ap.add_argument("--model-dir", default="/tmp/claude-0/p3model")
    ap.add_argument("--emb", default="out/phase1/emb/embeddinggemma2.parquet")
    ap.add_argument("--sets", nargs="+", default=["TLA", "ECL", "FRA"])
    ap.add_argument("--n-decks", type=int, default=64)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    res = os.path.join(args.out_dir, RES)
    os.makedirs(res, exist_ok=True)
    cards, feats = p3.card_table(args.out_dir, args.emb)
    model = p3.DeckModel(feats, cards["set"].map(p3.SETS.index).to_numpy())
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "deck_model.pt")))
    model.eval()
    env = model.env_embeddings()
    rng = np.random.default_rng(0)
    frames = []
    for s in args.sets:
        path = os.path.join(res, f"interactions_{s}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
            continue
        t0 = time.time()
        sc = synergy_scores(model, cards, s, env, args.n_decks, args.top_n, rng)
        G = p3.load_games(args.work_dir, [s])
        sc = empirical(G, sc)
        sc.to_csv(path, index=False)
        frames.append(sc)
        print(f"{s}: {len(sc)} pairs, {time.time() - t0:.0f}s", flush=True)
    allp = pd.concat(frames, ignore_index=True)
    rows = []
    for s, g in allp.groupby("set"):
        g = g.dropna(subset=["b_ab"])
        w = 1 / g["b_ab_se"] ** 2
        # Precision-weighted correlation between predicted synergy and b_ab.
        x, y = g["synergy"], g["b_ab"]
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        r = np.average((x - mx) * (y - my), weights=w) / np.sqrt(
            np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))
        top = g.nlargest(max(10, len(g) // 10), "synergy")
        bot = g.nsmallest(max(10, len(g) // 10), "synergy")
        wm = lambda d: np.average(d["b_ab"], weights=1 / d["b_ab_se"] ** 2)
        rows.append({"set": s, "n_pairs_tested": len(g), "weighted_corr": r,
                     "b_ab_top_decile": wm(top), "b_ab_bottom_decile": wm(bot), "b_ab_all": wm(g)})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(res, "interactions_summary.csv"), index=False)
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
