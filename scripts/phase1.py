"""Phase 1: card-only baselines on a temporal set holdout.

    python scripts/phase1.py --out-dir out --oracle data/raw/scryfall_oracle_cards.jsonl.gz

Models (all predict theta_vs_repl, the Phase 0 GIH label, on the logit scale):

- rarity_only: precision-weighted mean label per rarity, from the training sets
- gbt:       LightGBM on engineered features (+ set-level context features)
- tfidf:     ridge on TF-IDF of rules text, type line and mana cost, + rarity
- lm_<name>: ridge on frozen sentence embeddings, + rarity (when the
             embedding file from scripts/embed.py exists)
- gbt_tfidf: LightGBM on engineered features + 32 SVD components of TF-IDF

Hyperparameters are chosen on the last three training sets, then each model
is refitted on all training sets and scored on the test sets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mtgstrength import evaluate, features, scryfall  # noqa: E402

# Release order. Test sets are everything released after the newest
# training set; all of them are after the training cutoff of the text
# encoders used here.
SETS = ["KHM", "STX", "AFR", "MID", "VOW", "NEO", "SNC", "DMU", "BRO", "ONE", "MOM",
        "LTR", "WOE", "LCI", "MKM", "OTJ", "MH3", "BLB", "DSK", "FDN", "DFT", "TDM",
        "FIN", "EOE", "TLA", "ECL", "TMT", "SOS", "MSH", "HOB", "FRA"]
FIRST_TEST = "TLA"
N_VAL = 3
LABEL_NOISE_FLOOR = 0.02   # residual variance added to each label's SD^2 in weights
MIN_GAMES_WELL_MEASURED = 2000


def weights(df: pd.DataFrame) -> np.ndarray:
    return 1.0 / (df["theta_hand_sd"].to_numpy() ** 2 + LABEL_NOISE_FLOOR)


def fit_rarity(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    w = weights(tr)
    means = (tr.assign(w=w, wy=w * tr["theta_vs_repl"])
               .groupby("rarity")[["w", "wy"]].sum())
    means = means["wy"] / means["w"]
    return te["rarity"].map(means).fillna(means.mean()).to_numpy()


GBT_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15,
                  min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=2)


def fit_gbt(Xtr, ytr, wtr, Xva, yva, wva, Xfull, yfull, wfull, Xte) -> tuple[np.ndarray, int]:
    dtr = lgb.Dataset(Xtr, ytr, weight=wtr)
    dva = lgb.Dataset(Xva, yva, weight=wva)
    b = lgb.train(GBT_PARAMS, dtr, 3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    n = max(b.best_iteration, 50)
    b = lgb.train(GBT_PARAMS, lgb.Dataset(Xfull, yfull, weight=wfull), n)
    return b.predict(Xte), n, b


def fit_ridge(Xtr, ytr, wtr, Xva, yva, wva, Xfull, yfull, wfull, Xte,
              alphas=(0.1, 0.3, 1, 3, 10, 30, 100, 300)) -> tuple[np.ndarray, float]:
    best = None
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr, ytr, sample_weight=wtr)
        err = np.average((m.predict(Xva) - yva) ** 2, weights=wva)
        if best is None or err < best[0]:
            best = (err, a)
    m = Ridge(alpha=best[1]).fit(Xfull, yfull, sample_weight=wfull)
    return m.predict(Xte), best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--oracle", default="data/raw/scryfall_oracle_cards.jsonl.gz")
    ap.add_argument("--embeddings", nargs="*", default=None,
                    help="parquet files from scripts/embed.py (default: all in out/phase1/emb)")
    ap.add_argument("--first-test", default=FIRST_TEST)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    args = ap.parse_args()
    res_dir = os.path.join(args.out_dir, "phase1")
    os.makedirs(res_dir, exist_ok=True)

    labels = features.load_labels(args.out_dir, SETS)
    oracle = scryfall.load_oracle_cards(args.oracle)
    df = features.attach_oracle(labels, oracle)
    sets_present = [s for s in SETS if s in set(df["set"])]
    order = {s: i for i, s in enumerate(SETS)}
    df["set_idx"] = df["set"].map(order)
    df = df.sort_values(["set_idx", "theta_vs_repl"], ascending=[True, False]).reset_index(drop=True)
    train_sets = [s for s in sets_present if order[s] < order[args.first_test]]
    test_sets = [s for s in sets_present if order[s] >= order[args.first_test]]
    val_sets = train_sets[-args.n_val:]
    df["is_train"] = df["set"].isin(train_sets)
    df["is_test"] = df["set"].isin(test_sets)
    df["is_val"] = df["set"].isin(val_sets)
    print(f"train {train_sets}\nval {val_sets}\ntest {test_sets}\n{len(df)} card-set rows", flush=True)

    # A test card is "new" if its oracle ID never appears in a training set.
    seen = set(df.loc[df["is_train"], "oracle_id"])
    df["reprint_of_train"] = df["oracle_id"].isin(seen) & df["is_test"]

    # Engineered features.
    f = features.engineered(df)
    f = features.add_keywords(f, df, features.keyword_vocab(df[df["is_train"]]))
    f, curve = features.add_curve_and_context(f, df)
    X = f.astype(float)

    tr = df["is_train"].to_numpy()
    fit_m = tr & ~df["is_val"].to_numpy()
    va = df["is_val"].to_numpy()
    te = df["is_test"].to_numpy()
    y = df["theta_vs_repl"].to_numpy()
    w = weights(df)

    def split(M):
        return (M[fit_m], y[fit_m], w[fit_m], M[va], y[va], w[va], M[tr], y[tr], w[tr], M[te])

    preds = pd.DataFrame(index=df.index[te])
    info = {}

    preds["rarity_only"] = fit_rarity(df[tr], df[te])

    p, n, booster = fit_gbt(*split(X.to_numpy()))
    preds["gbt"] = p
    info["gbt_rounds"] = n
    imp = pd.Series(booster.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(res_dir, "gbt_feature_importance.csv"), header=["gain"])

    # Text: rules text with CARDNAME, plus type line and mana cost.
    text = (df["type_line"].fillna("") + " | " + df["mana_cost"].fillna("") + " | "
            + df.apply(features.card_text, axis=1)).str.replace("\n", " ")
    tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                          token_pattern=r"[A-Za-z]+|\{[^}]+\}|[+-]?\d+/[+-]?\d+|\d+")
    tfv.fit(text[tr])
    T = tfv.transform(text)
    R = sparse.csr_matrix(X[[f"rarity_{r}" for r in features.RARITIES]].to_numpy())
    TR = sparse.hstack([T, R * 3.0], format="csr")
    p, a = fit_ridge(*split(TR))
    preds["tfidf"] = p
    info["tfidf_alpha"] = a

    svd = TruncatedSVD(32, random_state=0).fit(T[tr])
    S = svd.transform(T)
    XS = np.hstack([X.to_numpy(), S])
    p, n, _ = fit_gbt(*split(XS))
    preds["gbt_tfidf"] = p
    info["gbt_tfidf_rounds"] = n

    # Frozen LM embeddings, when available.
    emb_dir = os.path.join(res_dir, "emb")
    emb_files = args.embeddings
    if emb_files is None:
        emb_files = sorted(os.path.join(emb_dir, x) for x in os.listdir(emb_dir)) \
            if os.path.isdir(emb_dir) else []
    for path in emb_files:
        name = os.path.basename(path).removesuffix(".parquet")
        E = pd.read_parquet(path).set_index("oracle_id")
        missing = ~df["oracle_id"].isin(E.index)
        if missing.any():
            print(f"{name}: {missing.sum()} cards without embeddings, skipped model", flush=True)
            continue
        M = np.hstack([E.loc[df["oracle_id"]].to_numpy(), X[[f"rarity_{r}" for r in features.RARITIES]].to_numpy()])
        p, a = fit_ridge(*split(M), alphas=(1, 3, 10, 30, 100, 300, 1000, 3000))
        preds[f"lm_{name}"] = p
        info[f"lm_{name}_alpha"] = a

    models = list(preds.columns)
    # Win-rate points: ame_vs_repl_pp is close to linear in theta; use the
    # slope fitted on the training sets to put predictions on that scale.
    m = tr & df["ame_vs_repl_pp"].notna().to_numpy()
    k = float(np.sum(df.loc[m, "ame_vs_repl_pp"] * y[m]) / np.sum(y[m] ** 2))
    info["pp_per_logit"] = k
    out = df.loc[te, ["set", "name", "oracle_id", "rarity", "theta_vs_repl", "theta_hand_sd",
                      "ame_vs_repl_pp", "n_games_in_hand", "reprint_of_train"]].join(preds)
    for c in models:
        out[c + "_pp"] = k * out[c]
    out.to_csv(os.path.join(res_dir, "test_predictions.csv"), index=False)

    subsets = {
        "all": out,
        f"well_measured (>= {MIN_GAMES_WELL_MEASURED} games in hand)":
            out[out["n_games_in_hand"] >= MIN_GAMES_WELL_MEASURED],
        "new_cards (oracle ID not in training sets)": out[~out["reprint_of_train"]],
    }
    per_set = pd.concat([evaluate.evaluate(d, models, name) for name, d in subsets.items()])
    per_set.to_csv(os.path.join(res_dir, "metrics_per_set.csv"), index=False)
    summary = (per_set.groupby(["subset", "model"], sort=False)
               [["spearman", "spearman_within_rarity", "ndcg@20", "top20_overlap", "mae_pp", "mae_logit"]]
               .agg(["mean", "std"]).round(3))
    summary.to_csv(os.path.join(res_dir, "metrics_summary.csv"))
    info.update({"train_sets": train_sets, "val_sets": val_sets, "test_sets": test_sets,
                 "n_train_rows": int(tr.sum()), "n_test_rows": int(te.sum()),
                 "vanilla_curve_pt_vs_mv": [float(c) for c in curve]})
    with open(os.path.join(res_dir, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    pd.set_option("display.width", 200)
    print(summary)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
