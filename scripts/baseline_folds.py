"""gbt_tfidf on the validation folds (docs/RESEARCH_PLAN.md), so ensembles with
it can be chosen on validation instead of test.

    python scripts/baseline_folds.py

Same model as Phase 1 gbt_tfidf (LightGBM on the 167+4 structured features
plus 32 SVD components of a TF-IDF of type line, mana cost and rules text),
fitted per fold on the sets before the fold: rounds by early stopping on the
last 3 of those sets, then refitted on all of them. Rules text uses the card's
own name -> CARDNAME from cards.parquet, a close stand-in for Phase 1's
Scryfall-name replacement. Writes runs/gbt_tfidf_folds/val_pred_<fold>_seed0.csv
(epoch = 1) in the format of phase2.py --val-fold. --rolling instead predicts
each set MKM..EOE from all earlier sets (fold R, step G backtest).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from phase1 import SETS, fit_gbt, weights  # noqa: E402
from phase2 import FOLDS  # noqa: E402


def rules_text(cards: pd.DataFrame) -> pd.Series:
    out = []
    for ot, name in zip(cards["oracle_text"].fillna(""), cards["name"]):
        for n in {name, *str(name).split(" // ")}:
            ot = ot.replace(n, "CARDNAME")
            if "," in n:
                ot = ot.replace(n.split(",")[0], "CARDNAME")
        out.append(ot)
    return pd.Series(out, index=cards.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--rolling", action="store_true",
                    help="rolling-origin backtest MKM..EOE -> runs/gbt_tfidf_rolling (fold R)")
    args = ap.parse_args()
    res = os.path.join(args.out_dir, "phase2")
    cards = pd.read_parquet(os.path.join(res, "cards.parquet"))
    X = pd.read_parquet(os.path.join(res, "struct.parquet")).to_numpy(float)
    y = cards["theta_vs_repl"].to_numpy()
    w = weights(cards)
    text = (cards["type_line"].fillna("") + " | " + cards["mana_cost"].fillna("") + " | "
            + rules_text(cards)).str.replace("\n", " ")
    order = {s: i for i, s in enumerate(SETS)}
    run = os.path.join(res, "runs", "gbt_tfidf_folds")
    os.makedirs(run, exist_ok=True)
    folds = FOLDS
    if args.rolling:
        # Rolling origin (step G): each set from MKM to EOE predicted from all earlier sets.
        run = os.path.join(res, "runs", "gbt_tfidf_rolling")
        os.makedirs(run, exist_ok=True)
        folds = {s: [s] for s in SETS[SETS.index("MKM"):SETS.index("EOE") + 1]}
    parts = []
    for fold, sets in folds.items():
        first = min(order[s] for s in sets)
        fit = cards["is_train"].to_numpy() & (cards["set_idx"] < first).to_numpy()
        fit_sets = list(dict.fromkeys(cards.loc[fit, "set"]))
        inner = cards["set"].isin(fit_sets[-3:]).to_numpy()
        va = cards["set"].isin(sets).to_numpy()
        tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                              token_pattern=r"[A-Za-z]+|\{[^}]+\}|[+-]?\d+/[+-]?\d+|\d+")
        T = tfv.fit(text[fit]).transform(text)
        S = TruncatedSVD(32, random_state=0).fit(T[fit]).transform(T)
        XS = np.hstack([X, S])
        a, b = fit & ~inner, inner
        p, n, _ = fit_gbt(XS[a], y[a], w[a], XS[b], y[b], w[b], XS[fit], y[fit], w[fit], XS[va])
        out = cards.loc[va, ["set", "oracle_id"]].assign(epoch=1, pred=p, pred_sd=np.nan)
        if args.rolling:
            parts.append(out)
        else:
            out.to_csv(os.path.join(run, f"val_pred_{fold}_seed0.csv"), index=False)
        print(f"{fold}: {n} rounds, {va.sum()} val cards", flush=True)
    if args.rolling:
        pd.concat(parts).to_csv(os.path.join(run, "val_pred_R_seed0.csv"), index=False)


if __name__ == "__main__":
    main()
