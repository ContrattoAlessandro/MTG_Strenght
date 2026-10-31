"""Phase 4: Commander and Constructed heads (design section 4, multi-task heads).

    python scripts/phase4.py commander
    python scripts/phase4.py constructed
    python scripts/phase4.py commander-folds   # validation folds, for step F2

Both use the Phase 2 card table and the same temporal split (train on cards
from the 24 training sets, test on the 7 newest), and score each model by
within-set Spearman on the test sets.

Commander target: logit of the EDHREC inclusion rate (decks that play the
card / decks whose colour identity allows it).

Constructed target: censored and ordinal. A card's value is its highest share
of decks in any archived MTGGoldfish top-50 staples list (Standard, Pioneer,
Modern, Legacy, Pauper). Cards never listed are only known to sit below the
lists' cut-off. Models are scored on the binary "ever a staple" (AUC within
the test sets pooled, and average precision) and on Spearman among staples.

Predictors compared, all fitted on the training sets only:
- rarity only;
- the Limited label θ itself and the Phase 2 model's predicted θ (does
  Limited strength transfer?);
- GBT on the engineered features; ridge on TF-IDF of the rules text;
- for Constructed, a censored (Tobit-style) GBT: the staples' share of decks
  as a regression target, with never-listed cards set to the cut-off and
  down-weighted.
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
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mtgstrength import features  # noqa: E402
from phase1 import GBT_PARAMS  # noqa: E402

RES = "phase4"


def load(out_dir: str):
    cards = pd.read_parquet(os.path.join(out_dir, "phase2", "cards.parquet"))
    f = pd.read_parquet(os.path.join(out_dir, "phase2", "struct.parquet"))
    # One row per oracle card: its first set in our data decides train/test,
    # so a reprint never crosses the split.
    first = cards.groupby("oracle_id")["set_idx"].transform("min")
    keep = (cards["set_idx"] == first).to_numpy()
    cards, f = cards[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    p2 = pd.read_csv(os.path.join(out_dir, "phase2", "test_predictions.csv"))
    return cards, f, p2


def text_matrix(cards: pd.DataFrame, tr: np.ndarray):
    tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                          token_pattern=r"[A-Za-z]+|\{[^}]+\}|[+-]?\d+/[+-]?\d+|\d+")
    tfv.fit(cards.loc[tr, "text"])
    return tfv.transform(cards["text"])


def fit_gbt(X, y, w, tr, va, objective="regression"):
    params = dict(GBT_PARAMS, objective=objective)
    fit = tr & ~va
    b = lgb.train(params, lgb.Dataset(X[fit], y[fit], weight=w[fit]), 3000,
                  valid_sets=[lgb.Dataset(X[va], y[va], weight=w[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    n = max(b.best_iteration, 50)
    b = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w[tr]), n)
    return b


def fit_ridge(M, y, w, tr, va):
    fit = tr & ~va
    best = None
    for a in (1, 3, 10, 30, 100, 300):
        m = Ridge(alpha=a).fit(M[fit], y[fit], sample_weight=w[fit])
        e = np.average((m.predict(M[va]) - y[va]) ** 2, weights=w[va])
        if best is None or e < best[0]:
            best = (e, a)
    return Ridge(alpha=best[1]).fit(M[tr], y[tr], sample_weight=w[tr])


def per_set_spearman(df: pd.DataFrame, target: str, models: list[str]) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("set", sort=False):
        for m in models:
            ok = g[m].notna() & g[target].notna()
            if ok.sum() > 10:
                rows.append({"set": s, "model": m, "n": int(ok.sum()),
                             "spearman": spearmanr(g.loc[ok, m], g.loc[ok, target]).statistic})
    return pd.DataFrame(rows)


def commander(args) -> None:
    res = os.path.join(args.out_dir, RES)
    cards, f, p2 = load(args.out_dir)
    e = pd.read_csv(os.path.join(res, "edhrec.csv")).drop_duplicates("oracle_id")
    e = e[(e["potential_decks"] > 1000)]
    e["incl"] = (e["num_decks"] + 1) / (e["potential_decks"] + 2)
    cards = cards.merge(e[["oracle_id", "incl", "num_decks", "potential_decks"]], on="oracle_id", how="left")
    ok = cards["incl"].notna().to_numpy()
    cards, f = cards[ok].reset_index(drop=True), f[ok].reset_index(drop=True)
    cards["y"] = np.log(cards["incl"] / (1 - cards["incl"]))
    tr = cards["is_train"].to_numpy()
    va = cards["is_val"].to_numpy()
    te = cards["is_test"].to_numpy()
    y = cards["y"].to_numpy()
    w = np.ones(len(y))
    X = f.astype(float).to_numpy()
    out = cards.loc[te, ["set", "name", "oracle_id", "rarity", "y", "theta_vs_repl", "num_decks"]].copy()

    # Rarity only.
    means = cards[tr].groupby("rarity")["y"].mean()
    out["rarity_only"] = out["rarity"].map(means)
    # Limited strength as a predictor: the true label, and the Phase 2 model's prediction.
    out["limited_theta"] = out["theta_vs_repl"]
    p = p2.set_index(["set", "oracle_id"])["ft_embeddinggemma2_k4"]
    out["limited_ft_gemma2"] = [p.get((s, o), np.nan) for s, o in zip(out["set"], out["oracle_id"])]
    b = fit_gbt(X, y, w, tr, va)
    out["gbt"] = b.predict(X[te])
    T = text_matrix(cards, tr)
    R = sparse.csr_matrix(f[[f"rarity_{r}" for r in features.RARITIES]].to_numpy() * 3.0)
    TR = sparse.hstack([T, R], format="csr")
    out["tfidf"] = fit_ridge(TR, y, w, tr, va).predict(TR[te])
    from sklearn.decomposition import TruncatedSVD
    S = TruncatedSVD(32, random_state=0).fit(T[tr]).transform(T)
    XS = np.hstack([X, S])
    out["gbt_tfidf"] = fit_gbt(XS, y, w, tr, va).predict(XS[te])
    # Fine-tuned card encoder with the Commander target, when it was run
    # (phase2.py train --label-csv out/phase4/commander_labels.csv).
    ft = os.path.join(args.out_dir, "phase2", "runs", "ft_embeddinggemma2_k4_commander")
    if os.path.isdir(ft):
        fs = [pd.read_csv(os.path.join(ft, x)) for x in os.listdir(ft) if x.startswith("test_pred")]
        if fs:
            m = pd.concat(fs).groupby(["set", "oracle_id"])["pred"].mean()
            out["ft_gemma2_commander"] = [m.get((s, o), np.nan) for s, o in zip(out["set"], out["oracle_id"])]
    models = [c for c in out.columns if c not in ("set", "name", "oracle_id", "rarity", "y",
                                                     "theta_vs_repl", "num_decks")]
    per = per_set_spearman(out, "y", models)
    per.to_csv(os.path.join(res, "commander_metrics_per_set.csv"), index=False)
    out.to_csv(os.path.join(res, "commander_test_predictions.csv"), index=False)
    # Labels for fine-tuning: one per (set, oracle_id) row of the Phase 2 table.
    allc = pd.read_parquet(os.path.join(args.out_dir, "phase2", "cards.parquet"))
    lab = allc[["set", "oracle_id"]].merge(cards[["oracle_id", "y"]].drop_duplicates("oracle_id"),
                                           on="oracle_id", how="left")
    lab.rename(columns={"y": "label"}).to_csv(os.path.join(res, "commander_labels.csv"), index=False)
    piv = per.pivot(index="model", columns="set", values="spearman")
    piv["mean"] = piv.mean(axis=1)
    pd.set_option("display.width", 200)
    print(piv.round(3).sort_values("mean"))
    print(f"{len(cards)} cards with EDHREC data, {te.sum()} in test sets")


def constructed(args) -> None:
    res = os.path.join(args.out_dir, RES)
    cards, f, p2 = load(args.out_dir)
    st = pd.read_csv(os.path.join(res, "goldfish_staples.csv"))
    thr = pd.read_csv(os.path.join(res, "goldfish_thresholds.csv"))
    key = lambda n: n.split(" // ")[0].lower().strip()
    best = st.assign(k=st["name"].map(key)).groupby("k")["pct_decks"].max()
    fmt = st.assign(k=st["name"].map(key)).groupby("k")["format"].agg(lambda x: ",".join(sorted(set(x))))
    cards["k"] = cards["name"].map(key)
    cards["staple_pct"] = cards["k"].map(best)
    cards["staple_formats"] = cards["k"].map(fmt)
    cards["is_staple"] = cards["staple_pct"].notna().astype(int)
    cut = float(thr["min_pct"].median())
    tr = cards["is_train"].to_numpy()
    va = cards["is_val"].to_numpy()
    te = cards["is_test"].to_numpy()
    X = f.astype(float).to_numpy()
    yb = cards["is_staple"].to_numpy()
    # Censored target: log share for staples, the cut-off for the rest.
    yc = np.log(np.where(yb == 1, cards["staple_pct"].fillna(cut), cut / 2))
    w = np.where(yb == 1, 1.0, 0.3)
    out = cards.loc[te, ["set", "name", "oracle_id", "rarity", "theta_vs_repl", "is_staple",
                         "staple_pct", "staple_formats"]].copy()
    rates = cards[tr].groupby("rarity")["is_staple"].mean()
    out["rarity_only"] = out["rarity"].map(rates)
    out["limited_theta"] = out["theta_vs_repl"]
    p = p2.set_index(["set", "oracle_id"])["ft_embeddinggemma2_k4"]
    out["limited_ft_gemma2"] = [p.get((s, o), np.nan) for s, o in zip(out["set"], out["oracle_id"])]
    out["gbt_binary"] = fit_gbt(X, yb.astype(float), np.ones(len(yb)), tr, va, "binary").predict(X[te])
    out["gbt_censored"] = fit_gbt(X, yc, w, tr, va).predict(X[te])
    T = text_matrix(cards, tr)
    R = sparse.csr_matrix(f[[f"rarity_{r}" for r in features.RARITIES]].to_numpy() * 3.0)
    TR = sparse.hstack([T, R], format="csr")
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(TR[tr], yb[tr])
    out["tfidf_logistic"] = lr.predict_proba(TR[te])[:, 1]
    models = [c for c in out.columns if c not in ("set", "name", "oracle_id", "rarity", "theta_vs_repl",
                                                     "is_staple", "staple_pct", "staple_formats")]
    rows = []
    for m in models:
        ok = out[m].notna()
        g = out[ok]
        # AUC within each set, averaged over sets that have at least 2 staples.
        aucs = [roc_auc_score(x["is_staple"], x[m]) for _, x in g.groupby("set")
                if 2 <= x["is_staple"].sum() < len(x)]
        st_only = g[g["is_staple"] == 1]
        rows.append({"model": m, "auc_within_set": np.mean(aucs) if aucs else np.nan,
                     "n_sets": len(aucs),
                     "avg_precision_pooled": average_precision_score(g["is_staple"], g[m]),
                     "spearman_among_staples": spearmanr(st_only[m], st_only["staple_pct"]).statistic
                     if len(st_only) > 5 else np.nan})
    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(res, "constructed_metrics.csv"), index=False)
    out.to_csv(os.path.join(res, "constructed_test_predictions.csv"), index=False)
    info = {"n_cards": len(cards), "n_staples_train": int(yb[tr].sum()), "n_staples_test": int(yb[te].sum()),
            "staples_per_test_set": out.groupby("set")["is_staple"].sum().to_dict(),
            "median_cutoff_pct": cut, "n_snapshots": len(thr)}
    json.dump(info, open(os.path.join(res, "constructed_info.json"), "w"), indent=2)
    print(met.round(3).to_string())
    print(json.dumps(info, indent=2))


def commander_folds(args) -> None:
    """Commander baselines on the validation folds (docs/RESEARCH_PLAN.md, F2).

    Writes out/phase4/commander_labels_first.csv (the label on each card's
    first printing only, so a reprint never carries its label across a fold
    boundary; use it for phase2.py --label-csv in fold mode and for
    compare_runs.py --label-csv) and runs/commander_tfidf_folds/ and
    runs/commander_gbt_tfidf_folds/ val predictions in the phase2.py format.
    """
    from phase2 import FOLDS
    from sklearn.decomposition import TruncatedSVD
    res = os.path.join(args.out_dir, RES)
    lab = pd.read_csv(os.path.join(res, "commander_labels.csv"))
    allc = pd.read_parquet(os.path.join(args.out_dir, "phase2", "cards.parquet"))
    lab = allc[["set", "oracle_id", "set_idx"]].merge(lab, on=["set", "oracle_id"], how="left")
    first = lab.groupby("oracle_id")["set_idx"].transform("min")
    lab.loc[lab["set_idx"] != first, "label"] = np.nan
    lab[["set", "oracle_id", "label"]].to_csv(os.path.join(res, "commander_labels_first.csv"), index=False)
    cards, f, _ = load(args.out_dir)
    cards = cards.merge(lab[["set", "oracle_id", "label"]], on=["set", "oracle_id"], how="left")
    ok = cards["label"].notna().to_numpy()
    cards, f = cards[ok].reset_index(drop=True), f[ok].reset_index(drop=True)
    y = cards["label"].to_numpy()
    w = np.ones(len(y))
    X = f.astype(float).to_numpy()
    runs = os.path.join(args.out_dir, "phase2", "runs")
    for name in ("commander_tfidf_folds", "commander_gbt_tfidf_folds"):
        os.makedirs(os.path.join(runs, name), exist_ok=True)
    for fold, sets in FOLDS.items():
        first_idx = cards.loc[cards["set"].isin(sets), "set_idx"].min()
        tr = cards["is_train"].to_numpy() & (cards["set_idx"] < first_idx).to_numpy()
        fit_sets = list(dict.fromkeys(cards.loc[tr, "set"]))
        inner = cards["set"].isin(fit_sets[-3:]).to_numpy()   # inner validation for alpha / rounds
        va = cards["set"].isin(sets).to_numpy()
        T = text_matrix(cards, tr)
        R = sparse.csr_matrix(f[[f"rarity_{r}" for r in features.RARITIES]].to_numpy() * 3.0)
        TR = sparse.hstack([T, R], format="csr")
        p_ridge = fit_ridge(TR, y, w, tr, inner).predict(TR[va])
        S = TruncatedSVD(32, random_state=0).fit(T[tr]).transform(T)
        XS = np.hstack([X, S])
        p_gbt = fit_gbt(XS, y, w, tr, inner).predict(XS[va])
        for name, p in (("commander_tfidf_folds", p_ridge), ("commander_gbt_tfidf_folds", p_gbt)):
            cards.loc[va, ["set", "oracle_id"]].assign(epoch=1, pred=p, pred_sd=np.nan).to_csv(
                os.path.join(runs, name, f"val_pred_{fold}_seed0.csv"), index=False)
        print(f"{fold}: {tr.sum()} fit cards, {va.sum()} val cards", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["commander", "constructed", "commander-folds"])
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()
    {"commander": commander, "constructed": constructed,
     "commander-folds": commander_folds}[args.cmd](args)


if __name__ == "__main__":
    main()
