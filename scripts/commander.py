"""Commander strength models on the full Commander-legal card pool (step I,
docs/COMMANDER_PLAN.md).

    python scripts/commander.py prep              # table + features (~2 min)
    python scripts/commander.py val               # every model on folds A, B
    python scripts/commander.py compare RUN... --ref REF [--split val|test]
    python scripts/commander.py test --models ... --why "..."   # logged look

prep needs out/commander/edhrec_all.csv and first_print.parquet
(scripts/cmd_data.py). Predictions: out/commander/runs/<run>/<split>_seed<k>.csv
with columns oracle_id, group, y, pred.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "app")]
from mtgstrength import features  # noqa: E402
from phase1 import GBT_PARAMS, SETS  # noqa: E402

OUT = os.path.join(ROOT, "out", "commander")
RUNS = os.path.join(OUT, "runs")
ORACLE = os.path.join(ROOT, "data", "raw", "scryfall_oracle_cards.jsonl.gz")
FETCH_DATE = "2026-11-01"
REF_AGE_DAYS = 730          # strength is reported at 2 years after first printing
MIN_GROUP = 20
SPLITS = {   # name: (train before, eval from, eval before)
    "A": ("2024-08-01", "2024-08-01", "2025-02-01"),
    "B": ("2025-02-01", "2025-02-01", "2025-11-01"),
    "test": ("2025-11-01", "2025-11-01", "2026-11-01"),
}
VAL = ["A", "B"]
LIMITED_TEST = [s.lower() for s in SETS[SETS.index("TLA"):]]
TOKEN = r"[A-Za-z]+|\{[^}]+\}|[+-]?\d+/[+-]?\d+|\d+"
SKIP_SET_TYPES = {"funny", "memorabilia", "token", "minigame", "vanguard", "alchemy"}
SET_TYPES = ["expansion", "core", "commander", "masters", "draft_innovation", "starter",
             "duel_deck", "from_the_vault", "premium_deck", "box", "eternal", "planechase",
             "arsenal", "spellbook"]


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- prep

def age_basis(age_days: np.ndarray) -> np.ndarray:
    """Additive age curve: log age plus two hinges (adoption is fast in the
    first months, then slow)."""
    a = np.log1p(np.clip(age_days, 0, None) / 30.0)
    return np.stack([a, np.clip(a - np.log1p(3), 0, None), np.clip(a - np.log1p(24), 0, None)], 1)


def commander_extras(df: pd.DataFrame) -> pd.DataFrame:
    tl = df["type_line"].fillna("")
    txt = df["oracle_text"].fillna("").str.lower()
    ci = df["color_identity"].fillna("")
    x = pd.DataFrame(index=df.index)
    for c in "WUBRG":
        x[f"ci_{c}"] = ci.str.contains(c).astype(int)
    x["ci_count"] = ci.str.len()
    x["commander_capable"] = ((tl.str.contains("Legendary") & tl.str.contains("Creature"))
                              | txt.str.contains("can be your commander")).astype(int)
    x["nonbasic_land"] = tl.str.contains("Land").astype(int)
    x["tribal_payoff"] = txt.str.contains(r"creature type|of the chosen type|share a creature type").astype(int)
    x["each_opponent"] = txt.str.contains("each opponent").astype(int)
    x["whenever_another"] = txt.str.contains("whenever another|whenever a creature|whenever one or more").astype(int)
    x["log_text_len"] = np.log1p(txt.str.len())
    for s in SET_TYPES:
        x[f"settype_{s}"] = (df["first_set_type"] == s).astype(int)
    return x


def prep(args) -> None:
    import engine
    e = pd.read_csv(os.path.join(OUT, "edhrec_all.csv")).drop_duplicates("oracle_id")
    fp = pd.read_parquet(os.path.join(OUT, "first_print.parquet"))
    rows = []
    with gzip.open(ORACLE, "rt", encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            if c.get("legalities", {}).get("commander") != "legal":
                continue
            if c.get("set_type") in SKIP_SET_TYPES or "Basic Land" in (c.get("type_line") or ""):
                continue
            faces = c.get("card_faces") or []
            get = lambda k: c.get(k) if c.get(k) is not None else (  # noqa: E731
                " // ".join(str(f.get(k)) for f in faces if f.get(k) is not None) or None)
            rows.append({"oracle_id": c["oracle_id"], "name": c["name"], "mana_cost": get("mana_cost") or "",
                         "cmc": c.get("cmc"), "type_line": get("type_line") or "",
                         "oracle_text": c.get("oracle_text") if c.get("oracle_text") is not None
                         else "\n".join(f.get("oracle_text", "") for f in faces),
                         "power": get("power"), "toughness": get("toughness"), "loyalty": get("loyalty"),
                         "color_identity": "".join(c.get("color_identity") or []),
                         "keywords": c.get("keywords") or [], "layout": c.get("layout"),
                         "rarity": c.get("rarity")})
    raw = pd.DataFrame(rows).drop_duplicates("oracle_id")
    raw = raw.merge(fp, on="oracle_id", how="inner")
    raw = raw[~raw["first_set_type"].isin(SKIP_SET_TYPES)]
    raw = raw.merge(e[["oracle_id", "num_decks", "potential_decks", "salt", "precon"]], on="oracle_id", how="left")
    raw = raw.reset_index(drop=True)
    vocab = features.keyword_vocab(raw, min_count=30)
    df = engine.normalise(raw.to_dict("records"), vocab)
    df["oracle_id"] = raw["oracle_id"].to_numpy()
    df["scryfall_name"] = raw["name"].to_numpy()
    df["front_name"] = raw["name"].str.split(" // ").str[0].to_numpy()
    for c in ["first_release", "first_set", "first_set_type", "game_changer", "num_decks",
              "potential_decks", "salt", "precon"]:
        df[c] = raw[c].to_numpy()
    df["age_days"] = (pd.Timestamp(FETCH_DATE) - pd.to_datetime(df["first_release"])).dt.days
    ok = df["potential_decks"].fillna(0) > 1000
    incl = (df["num_decks"] + 1) / (df["potential_decks"] + 2)
    df["y"] = np.where(ok, np.log(incl / (1 - incl)), np.nan)
    lim = pd.read_parquet(os.path.join(ROOT, "out", "phase2", "cards.parquet"))
    df["limited_pool"] = df["oracle_id"].isin(set(lim["oracle_id"]))
    df["text"] = engine.input_text(df)
    df["tfidf_text"] = engine.tfidf_text(df)
    f = features.engineered(df)
    f = features.add_keywords(f, df, vocab)
    crea = (f["type_creature"] == 1) & f["pt_sum"].notna() & (f["pt_star"] == 0)
    curve = np.polyfit(f.loc[crea, "mana_value"], f.loc[crea, "pt_sum"], 1)
    f["stat_eff"] = np.where(crea, f["pt_sum"] - np.polyval(curve, f["mana_value"]), np.nan)
    f = pd.concat([f, commander_extras(df)], axis=1)
    keep = ["oracle_id", "name", "rarity", "type_line", "color_identity", "first_release", "first_set",
            "first_set_type", "game_changer", "num_decks", "potential_decks", "salt", "precon",
            "age_days", "y", "limited_pool", "text", "tfidf_text", "mana_cost", "oracle_text"]
    df[keep].to_parquet(os.path.join(OUT, "table.parquet"), index=False)
    f.astype(float).to_parquet(os.path.join(OUT, "features.parquet"), index=False)
    json.dump({"vocab": vocab, "curve": list(curve)}, open(os.path.join(OUT, "prep_info.json"), "w"))
    lab = df["y"].notna()
    log(f"{len(df)} Commander-legal cards, {lab.sum()} labelled, {df['limited_pool'].sum()} in the "
        f"Limited pool; {f.shape[1]} features")
    for s, (tb, a, b) in SPLITS.items():
        tr = lab & (df["first_release"] < tb)
        ev = lab & (df["first_release"] >= a) & (df["first_release"] < b)
        log(f"{s}: train {tr.sum()} (Limited pool {(tr & df['limited_pool']).sum()}), eval {ev.sum()}")


# --------------------------------------------------------------------------- models

def load():
    df = pd.read_parquet(os.path.join(OUT, "table.parquet"))
    f = pd.read_parquet(os.path.join(OUT, "features.parquet"))
    return df, f


def masks(df: pd.DataFrame, split: str):
    tb, a, b = SPLITS[split]
    lab = df["y"].notna().to_numpy()
    fr = df["first_release"].to_numpy()
    tr = lab & (fr < tb)
    ev = lab & (fr >= a) & (fr < b)
    # inner validation for alpha / boosting rounds: the last 6 months of train
    inner_from = (pd.Timestamp(tb) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    inner = tr & (fr >= inner_from)
    return tr, ev, inner


def tfidf(df, tr, col="text", min_df=3):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True, token_pattern=TOKEN)
    v.fit(df.loc[tr, col])
    return v.transform(df[col])


def dense_cols(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c.startswith(("ci_", "settype_", "type_", "rarity_"))
            or c in ("legendary", "commander_capable", "nonbasic_land", "tribal_payoff",
                     "each_opponent", "whenever_another", "log_text_len", "mana_value", "two_faced")]


def dense_block(f: pd.DataFrame, df: pd.DataFrame, tr: np.ndarray, age: bool) -> np.ndarray:
    cols = dense_cols(f)
    X = f[cols].fillna(0).to_numpy(float)
    if age:
        X = np.hstack([X, age_basis(df["age_days"].to_numpy())])
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    return (X - mu) / sd


def ridge_fit_predict(M, y, fit, inner, ev, alphas=(0.1, 0.3, 1, 3, 10, 30, 100), w=None):
    w = np.ones(len(y)) if w is None else w
    best = None
    for a in alphas:
        m = Ridge(alpha=a).fit(M[fit & ~inner], y[fit & ~inner], sample_weight=w[fit & ~inner])
        e = np.mean((m.predict(M[inner & fit]) - y[inner & fit]) ** 2)
        if best is None or e < best[0]:
            best = (e, a)
    m = Ridge(alpha=best[1]).fit(M[fit], y[fit], sample_weight=w[fit])
    return m.predict(M[ev]), best[1], m


def gbt_fit_predict(X, y, fit, inner, ev, seed):
    p = dict(GBT_PARAMS, seed=seed, num_threads=4)
    b = lgb.train(p, lgb.Dataset(X[fit & ~inner], y[fit & ~inner]), 4000,
                  valid_sets=[lgb.Dataset(X[inner & fit], y[inner & fit])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    n = max(b.best_iteration, 50)
    b = lgb.train(p, lgb.Dataset(X[fit], y[fit]), n)
    return b.predict(X[ev]), n, b


def run_models(df, f, split, models, seeds=(0, 1, 2)):
    tr, ev, inner = masks(df, split)
    y = df["y"].to_numpy()
    lim = df["limited_pool"].to_numpy()
    R = sparse.csr_matrix(f[[f"rarity_{r}" for r in features.RARITIES]].to_numpy() * 3.0)
    T_lim = None
    out = {}

    def save(name, seed, pred, info=""):
        d = os.path.join(RUNS, name)
        os.makedirs(d, exist_ok=True)
        df.loc[ev, ["oracle_id", "first_set"]].rename(columns={"first_set": "group"}).assign(
            y=y[ev], pred=pred).to_csv(os.path.join(d, f"{split}_seed{seed}.csv"), index=False)
        out[(name, seed)] = pred
        log(f"{split} {name} seed {seed} {info}")

    if "ft_cmd" in models:         # shipped fine-tuned head (g_commander, trained on the 24
        fs = pd.read_parquet(os.path.join(ROOT, "out", "app", "ft_cmd_scores.parquet"))  # Limited train sets)
        m = fs.drop_duplicates("oracle_id").set_index("oracle_id")["ft_pred"]
        save("ft_cmd", 0, df.loc[ev, "oracle_id"].map(m).to_numpy(), "precomputed g_commander")
    T = tfidf(df, tr)
    TR = sparse.hstack([T, R], format="csr")
    if "ridge_lim" in models:      # the shipped recipe: Limited-pool cards only
        fit = tr & lim
        T_lim = tfidf(df, fit)
        M = sparse.hstack([T_lim, R], format="csr")
        p, a, _ = ridge_fit_predict(M, y, fit, inner, ev)
        save("ridge_lim", 0, p, f"alpha {a}, {fit.sum()} cards")
    if "ridge_all" in models:
        p, a, _ = ridge_fit_predict(TR, y, tr, inner, ev)
        save("ridge_all", 0, p, f"alpha {a}, {tr.sum()} cards")
    if "ridge_sub" in models:      # full pool, subsampled to the Limited pool's size
        n = int((tr & lim).sum())
        for s in seeds:
            rng = np.random.default_rng(s)
            idx = rng.choice(np.flatnonzero(tr), n, replace=False)
            fit = np.zeros_like(tr)
            fit[idx] = True
            M = sparse.hstack([tfidf(df, fit), R], format="csr")
            p, a, _ = ridge_fit_predict(M, y, fit, inner, ev)
            save("ridge_sub", s, p, f"alpha {a}, {fit.sum()} cards")
    for name, age, alphas in (("ridge_struct", True, None), ("ridge_struct_noage", False, None),
                              ("ridge_struct_a1", True, (1,)), ("ridge_struct_a3", True, (3,))):
        # ridge_struct picks alpha on the inner split by MSE; _a1 / _a3 fix it
        # at the value chosen by within-group Spearman on folds A and B.
        if name in models:
            D = sparse.csr_matrix(dense_block(f, df, tr, age))
            M = sparse.hstack([T, R, D], format="csr")
            p, a, _ = ridge_fit_predict(M, y, tr, inner, ev, **({"alphas": alphas} if alphas else {}))
            save(name, 0, p, f"alpha {a}")
    # Which training cards: the Limited pool only, recent cards only, or
    # recency weights (half-life in years), all with the ridge_struct features.
    fr = pd.to_datetime(df["first_release"])
    yrs = (pd.Timestamp(SPLITS[split][0]) - fr).dt.days.to_numpy() / 365.25
    variants = {"ridge_struct_lim": (tr & lim, None),
                "ridge_struct_since2015": (tr & (fr >= "2015-01-01").to_numpy(), None),
                "ridge_struct_since2020": (tr & (fr >= "2020-01-01").to_numpy(), None),
                "ridge_struct_hl3": (tr, 0.5 ** (yrs / 3)),
                "ridge_struct_hl6": (tr, 0.5 ** (yrs / 6))}
    for name, (fit, w) in variants.items():
        if name in models:
            Tf = tfidf(df, fit)
            D = sparse.csr_matrix(dense_block(f, df, fit, True))
            M = sparse.hstack([Tf, R, D], format="csr")
            p, a, _ = ridge_fit_predict(M, y, fit, inner, ev, w=w)
            save(name, 0, p, f"alpha {a}, {fit.sum()} cards")
    if any(m.startswith("gbt") for m in models):
        S = TruncatedSVD(64, random_state=0).fit(T[tr]).transform(T)
        base = f.to_numpy(float)
        A = age_basis(df["age_days"].to_numpy())[:, :1]
        for name, X in (("gbt", np.hstack([base, S, A])), ("gbt_noage", np.hstack([base, S])),
                        ("gbt_nosvd", np.hstack([base, A]))):
            if name in models:
                for s in seeds:
                    p, n, _ = gbt_fit_predict(X, y, tr, inner, ev, s)
                    save(name, s, p, f"{n} rounds")
    return out


# --------------------------------------------------------------------------- evaluation

def read_run(name: str, split: str) -> pd.DataFrame:
    d = os.path.join(RUNS, name)
    fs = [pd.read_csv(os.path.join(d, x)) for x in sorted(os.listdir(d)) if x.startswith(f"{split}_seed")]
    if not fs:
        raise FileNotFoundError(f"{name}: no {split} predictions")
    g = pd.concat(fs).groupby(["oracle_id", "group", "y"], as_index=False)["pred"].mean()
    g["n_seeds"] = len(fs)
    return g


def blend(runs: dict[str, float], split: str) -> pd.DataFrame:
    parts = [read_run(r, split).set_index(["oracle_id", "group", "y"])["pred"] * w for r, w in runs.items()]
    return sum(parts).reset_index()


def ndcg(y, p, k=20):
    order = np.argsort(-p)[:k]
    ideal = np.sort(y)[::-1][:k]
    gain = lambda v: v - y.min()  # noqa: E731
    disc = 1 / np.log2(np.arange(2, k + 2))
    return float(np.sum(gain(y[order]) * disc[:len(order)]) / np.sum(gain(ideal) * disc[:len(ideal)]))


def per_group(g: pd.DataFrame, groups: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for k, x in g.groupby("group"):
        if groups is not None and k not in groups:
            continue
        if len(x) >= MIN_GROUP:
            rows.append({"group": k, "n": len(x), "spearman": spearmanr(x["pred"], x["y"]).statistic,
                         "ndcg20": ndcg(x["y"].to_numpy(), x["pred"].to_numpy())})
    return pd.DataFrame(rows)


def get_pred(name: str, split: str) -> pd.DataFrame:
    if "+" in name:   # blend "a*0.5+b*0.5"
        ws = {}
        for part in name.split("+"):
            r, w = part.split("*")
            ws[r] = float(w)
        return blend(ws, split)
    return read_run(name, split)


def paired(a: pd.DataFrame, b: pd.DataFrame, n_boot=1000, seed=0):
    m = a.merge(b, on="group", suffixes=("", "_ref"))
    d = m["spearman"] - m["spearman_ref"]
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d)


def card_bootstrap(pa: pd.DataFrame, pb: pd.DataFrame, groups, n_boot=500, seed=0):
    m = pa.merge(pb[["oracle_id", "pred"]], on="oracle_id", suffixes=("", "_ref"))
    m = m[m["group"].isin(groups)]
    rng = np.random.default_rng(seed)
    gs = [x for _, x in m.groupby("group")]
    out = []
    for _ in range(n_boot):
        ds = []
        for x in gs:
            i = rng.integers(0, len(x), len(x))
            xx = x.iloc[i]
            ds.append(spearmanr(xx["pred"], xx["y"]).statistic - spearmanr(xx["pred_ref"], xx["y"]).statistic)
        out.append(np.nanmean(ds))
    return np.percentile(out, [2.5, 97.5])


def _limited_groups() -> set:
    df = pd.read_parquet(os.path.join(OUT, "table.parquet"), columns=["first_set", "limited_pool", "y"])
    return set(df.loc[df["limited_pool"] & df["y"].notna(), "first_set"])


def compare(names: list[str], ref: str, splits: list[str], groups_filter: str = "all") -> pd.DataFrame:
    rows = []
    for name in [ref] + [n for n in names if n != ref]:
        pg, pooled, preds = [], [], []
        for s in splits:
            p = get_pred(name, s)
            groups = None
            if groups_filter == "limited":
                groups = [x for x in p["group"].unique() if x.upper() in SETS]
            elif groups_filter == "nonlimited":   # no card of the group is in the Limited pool,
                groups = [x for x in p["group"].unique() if x not in _limited_groups()]  # so the
                # shipped fine-tuned head (trained on Limited-pool cards) is out-of-sample there
            pg.append(per_group(p, groups).assign(split=s))
            q = p if groups is None else p[p["group"].isin(groups)]
            pooled.append(spearmanr(q["pred"], q["y"]).statistic)
            preds.append(p)
        pg = pd.concat(pg)
        rows.append({"run": name, "groups": len(pg), "spearman": pg["spearman"].mean(),
                     "ndcg20": pg["ndcg20"].mean(), "pooled": np.mean(pooled), "_pg": pg,
                     "_pred": pd.concat(preds)})
    ref_pg, ref_pred = rows[0]["_pg"], rows[0]["_pred"]
    for r in rows:
        d, se, won, n = paired(r["_pg"], ref_pg)
        r.update(delta=d, se=se, won=f"{won}/{n}")
        if r["run"] != ref:
            lo, hi = card_bootstrap(r["_pred"], ref_pred, list(r["_pg"]["group"]), n_boot=200)
            r["boot95"] = f"[{lo:+.3f}, {hi:+.3f}]"
            r["verdict"] = "REAL" if (d > 2 * se and won >= 2 * n / 3) else ("worse" if d < -2 * se else "tie")
    return pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])


# --------------------------------------------------------------------------- final model

FINAL_ALPHA = 1.0     # chosen by within-group Spearman on folds A and B (VAL.md)
FT_WEIGHT = 0.5       # weight of the shipped fine-tuned head (g_commander) in the blend


def _design(df, f, fit, vec=None, dense_stats=None, ref_age=None):
    """TF-IDF + rarity + standardized structured block, as ridge_struct."""
    if vec is None:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, token_pattern=TOKEN)
        vec.fit(df.loc[fit, "text"])
    T = vec.transform(df["text"])
    R = sparse.csr_matrix(f[[f"rarity_{r}" for r in features.RARITIES]].to_numpy() * 3.0)
    cols = dense_cols(f)
    # At the reference age: cards younger than it are scored as if they were
    # that old; older cards keep their age (their era effect is real).
    age = df["age_days"].to_numpy() if ref_age is None else np.maximum(df["age_days"].to_numpy(), ref_age)
    X = np.hstack([f[cols].fillna(0).to_numpy(float), age_basis(age)])
    if dense_stats is None:
        dense_stats = (X[fit].mean(0), X[fit].std(0) + 1e-6)
    D = (X - dense_stats[0]) / dense_stats[1]
    return sparse.hstack([T, R, sparse.csr_matrix(D)], format="csr"), vec, dense_stats


def ft_age_curve(age: np.ndarray, resid: np.ndarray) -> np.ndarray:
    A = np.hstack([np.ones((len(age), 1)), age_basis(age)])
    return np.linalg.lstsq(A, resid, rcond=None)[0]


def ft_age_shift(age: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(age), 1)), age_basis(age)]) @ w


def final(args) -> None:
    """Fit the chosen ridge on every labelled card and score every card.

    Writes out/commander/scores.parquet (one row per Commander-legal card):
    cmd_ridge (ridge at max(age, reference age)), cmd_ridge_oof (same, 6-fold CV
    over first-printing sets, for labelled cards), cmd_ft (g_commander),
    cmd_pred (blend; OOF ridge for labelled cards), cmd_obs (logit EDHREC
    inclusion) and cmd_obs_adj (observed, lifted to the reference age for
    cards younger than it). And out/commander/model.joblib for live scoring.
    """
    import joblib
    df, f = load()
    y = df["y"].to_numpy()
    lab = ~np.isnan(y)
    M, vec, stats = _design(df, f, lab)
    Mref, _, _ = _design(df, f, lab, vec, stats, ref_age=REF_AGE_DAYS)
    m = Ridge(alpha=FINAL_ALPHA).fit(M[lab], y[lab])
    df["cmd_ridge"] = m.predict(Mref)
    # Out-of-fold estimates for labelled cards (folds = whole first-printing sets).
    fold = df["first_set"].map(lambda s: int.from_bytes(s.encode(), "little") % 6).to_numpy()
    oof = np.full(len(df), np.nan)
    for k in range(6):
        fit = lab & (fold != k)
        Mk, vk, sk = _design(df, f, fit)
        Mk_ref, _, _ = _design(df, f, fit, vk, sk, ref_age=REF_AGE_DAYS)
        mk = Ridge(alpha=FINAL_ALPHA).fit(Mk[fit], y[fit])
        te = lab & (fold == k)
        oof[te] = mk.predict(Mk_ref[te])
        log(f"fold {k}: {te.sum()} cards")
    df["cmd_ridge_oof"] = oof
    fs = pd.read_parquet(os.path.join(ROOT, "out", "app", "ft_cmd_scores.parquet")).drop_duplicates("oracle_id")
    df["cmd_ft"] = df["oracle_id"].map(fs.set_index("oracle_id")["ft_pred"])
    ridge = np.where(lab, df["cmd_ridge_oof"], df["cmd_ridge"])
    # The fine-tuned head knows neither age nor era (it was trained on 2021-25
    # Limited cards, which are played more than older cards), so its offset is
    # corrected with an age curve fitted on the labelled cards, at the reference age.
    ok = lab & df["cmd_ft"].notna().to_numpy()
    ft_w = ft_age_curve(df["age_days"].to_numpy()[ok], (y - df["cmd_ft"].to_numpy())[ok])
    ft_adj = df["cmd_ft"].to_numpy() + ft_age_shift(np.maximum(df["age_days"].to_numpy(), REF_AGE_DAYS), ft_w)
    df["cmd_pred"] = np.where(df["cmd_ft"].notna(), (1 - FT_WEIGHT) * ridge + FT_WEIGHT * ft_adj, ridge)
    log(f"fine-tuned head offset at the reference age {ft_age_shift(np.array([REF_AGE_DAYS]), ft_w)[0]:+.2f}")
    # Age adjustment of the observed label: the ridge's additive age curve.
    k = len(dense_cols(f))
    w_age = m.coef_[-3:] / stats[1][k:]
    curve = lambda a: age_basis(np.atleast_1d(np.asarray(a, float))) @ w_age  # noqa: E731
    age = df["age_days"].to_numpy()
    lift = np.where(age < REF_AGE_DAYS, curve(REF_AGE_DAYS) - curve(np.clip(age, 0, None)), 0.0)
    df["cmd_obs"] = y
    df["cmd_obs_adj"] = y + lift
    # Predictive spread: residual quantiles of the validation-fold predictions
    # (ridge_struct_a1, out of sample), so the range is calibrated by construction.
    res = pd.concat([read_run("ridge_struct_a1", s) for s in VAL])
    q = np.percentile(res["y"] - res["pred"], [10, 90])
    keep = ["oracle_id", "name", "first_release", "first_set", "first_set_type", "age_days",
            "num_decks", "potential_decks", "game_changer", "salt", "cmd_ridge", "cmd_ridge_oof",
            "cmd_ft", "cmd_pred", "cmd_obs", "cmd_obs_adj"]
    df[keep].to_parquet(os.path.join(OUT, "scores.parquet"), index=False)
    joblib.dump({"vec": vec, "ridge": m, "dense_cols": dense_cols(f), "dense_stats": stats,
                 "ref_age": REF_AGE_DAYS, "ft_weight": FT_WEIGHT, "q80": q, "ft_age_w": ft_w,
                 "vocab": json.load(open(os.path.join(OUT, "prep_info.json")))["vocab"],
                 "curve": json.load(open(os.path.join(OUT, "prep_info.json")))["curve"],
                 "age_curve": [float(curve(a)[0]) for a in (30, 90, 182, 365, 730, 1461, 3652)]},
                os.path.join(OUT, "model.joblib"), compress=3)
    log(f"age curve (logit vs 1 month): " + ", ".join(
        f"{a}d {(curve(a) - curve(30))[0]:+.2f}" for a in (90, 182, 365, 730, 1461, 3652)))
    log(f"80% residual range {q.round(2)}; {lab.sum()} labelled, {len(df)} scored")
    # External check: Commander Game Changers (WotC's list of the strongest cards).
    from sklearn.metrics import roc_auc_score
    ok = lab & df["game_changer"].notna().to_numpy()
    gc = df.loc[ok, "game_changer"].astype(bool).to_numpy()
    rows = []
    for name, v in (("observed inclusion (age-adjusted)", df["cmd_obs_adj"]),
                    ("model, out-of-fold ridge", df["cmd_ridge_oof"]),
                    ("model, shipped blend", df["cmd_pred"]), ("EDHREC salt", df["salt"])):
        vv = v.to_numpy()[ok]
        good = ~np.isnan(vv)
        rows.append({"score": name, "auc": roc_auc_score(gc[good], vv[good]), "n": int(good.sum()),
                     "n_game_changers": int(gc[good].sum())})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "game_changers_auc.csv"), index=False)
    print(t.round(3).to_string(index=False))


def report(args) -> None:
    """Tables for out/commander/REPORT.md (VAL.md, TEST.md) and metrics.json for the app."""
    import joblib
    val_runs = ["ridge_all", "ridge_sub", "ridge_struct_lim", "ridge_struct", "ridge_struct_noage",
                "ridge_struct_a1", "ridge_struct_since2015", "ridge_struct_hl6", "gbt", "gbt_noage",
                "gbt_nosvd", "ridge_struct*0.7+gbt*0.3", "ft_cmd", "ridge_lim*0.5+ft_cmd*0.5",
                "ridge_struct*0.5+ft_cmd*0.5", "ridge_struct_a1*0.5+ft_cmd*0.5"]
    test_runs = ["ridge_lim", "ft_cmd", "ridge_struct", "ridge_struct_noage", "ridge_struct*0.5+ft_cmd*0.5",
                 "ridge_struct_a1", "ridge_struct_a1*0.5+ft_cmd*0.5"]
    cols = ["run", "groups", "spearman", "ndcg20", "pooled", "delta", "se", "won", "boot95", "verdict"]
    for name, runs, split, ref in (("VAL.md", val_runs, "val", "ridge_lim"),
                                   ("TEST.md", test_runs, "test", "ridge_lim*0.5+ft_cmd*0.5")):
        lines = [f"# Commander models, {split} (mean within-set Spearman; Δ vs `{ref}`)\n",
                 "Written by `python scripts/commander.py report`. Groups = first-printing sets with "
                 f"≥ {MIN_GROUP} labelled cards; 'nonlimited' = groups with no card in the Limited pool.\n"]
        for gf in ("all", "limited", "nonlimited"):
            t = compare(runs, ref, VAL if split == "val" else ["test"], gf)
            lines.append(f"\n## Groups: {gf}\n")
            lines.append(t[[c for c in cols if c in t]].round(4).to_markdown(index=False))
        open(os.path.join(OUT, name), "w").write("\n".join(lines) + "\n")
        log(f"wrote {name}")
    cmd = joblib.load(os.path.join(OUT, "model.joblib"))
    # 80% range coverage on test, with the fine-tuned offset curve fitted on
    # cards first printed before the test window only.
    df = pd.read_parquet(os.path.join(OUT, "table.parquet"), columns=["oracle_id", "y", "age_days", "first_release"])
    fs = pd.read_parquet(os.path.join(ROOT, "out", "app", "ft_cmd_scores.parquet")).drop_duplicates("oracle_id")
    df["ft"] = df["oracle_id"].map(fs.set_index("oracle_id")["ft_pred"])
    pre = df["y"].notna() & df["ft"].notna() & (df["first_release"] < SPLITS["test"][0])
    w = ft_age_curve(df.loc[pre, "age_days"].to_numpy(), (df.loc[pre, "y"] - df.loc[pre, "ft"]).to_numpy())
    r = read_run("ridge_struct_a1", "test").merge(df[["oracle_id", "ft", "age_days"]], on="oracle_id")
    pc = 0.5 * r["pred"] + 0.5 * (r["ft"] + ft_age_shift(r["age_days"].to_numpy(), w))
    cov = float(np.mean((r["y"] >= pc + cmd["q80"][0]) & (r["y"] <= pc + cmd["q80"][1])))
    p = get_pred("ridge_struct_a1*0.5+ft_cmd*0.5", "test")
    new = per_group(p)["spearman"].mean()
    old = per_group(get_pred("ridge_lim*0.5+ft_cmd*0.5", "test"))["spearman"].mean()
    m = {"test_spearman": float(new), "test_spearman_old": float(old), "coverage80": cov,
         "n_labelled": int(df["y"].notna().sum()), "snapshot": FETCH_DATE}
    json.dump(m, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    print(m)


# --------------------------------------------------------------------------- fine-tuning tables (H5)

def p2prep(args) -> None:
    """Card tables for phase2.py (--res commander/p2, commander/p2_all) so the
    EmbeddingGemma-2 recipe can be fine-tuned on the full Commander pool.

    One row per card; "set" is the first-printing set plus month (a set code
    such as SLD spans years), ordered by date in set_idx. The label is put in
    theta_vs_repl with a nominal SD of 10% of its spread (as phase2.py does for
    --target label). Struct = the Commander features + the age curve.
    p2:     labelled cards; is_train before the test window, is_test = test
            window; folds.json = validation folds A and B (phase2.py --val-fold A B).
    p2_all: every Commander card; trained on all labelled cards, predicts all.
    """
    df, f = load()
    f = pd.concat([f, pd.DataFrame(age_basis(df["age_days"].to_numpy()),
                                   columns=["age_log", "age_h3", "age_h24"], index=f.index)], axis=1)
    df["set"] = df["first_set"].str.upper() + "@" + df["first_release"].str[:7]
    first = df.groupby("set")["first_release"].min()
    order = {k: i for i, k in enumerate(first.sort_values(kind="stable").index)}
    df["set_idx"] = df["set"].map(order)
    sd = 0.1 * np.nanstd(df["y"])
    base = pd.DataFrame({
        "set": df["set"], "set_idx": df["set_idx"], "name": df["name"], "oracle_id": df["oracle_id"],
        "rarity": df["rarity"], "theta_vs_repl": df["y"], "theta_hand_sd": sd,
        "ame_vs_repl_pp": df["y"], "n_games_in_hand": df["num_decks"], "text": df["text"],
        "type_line": df["type_line"], "mana_cost": df["mana_cost"], "oracle_text": df["oracle_text"],
        "power": None, "toughness": None, "cmc": f["mana_value"], "reprint_of_train": False})
    lab = df["y"].notna().to_numpy()
    fr = df["first_release"].to_numpy()
    for name, rows, tr, va, te in (
            ("p2", lab, fr < SPLITS["test"][0], (fr >= SPLITS["B"][1]) & (fr < SPLITS["B"][2]),
             (fr >= SPLITS["test"][1]) & (fr < SPLITS["test"][2])),
            ("p2_all", np.ones(len(df), bool), lab, np.zeros(len(df), bool), np.ones(len(df), bool))):
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        c = base.assign(is_train=tr & lab, is_val=va & lab, is_test=te)[rows].reset_index(drop=True)
        c.to_parquet(os.path.join(d, "cards.parquet"), index=False)
        f[rows].reset_index(drop=True).astype(float).to_parquet(os.path.join(d, "struct.parquet"), index=False)
        log(f"{name}: {len(c)} rows, train {c['is_train'].sum()}, test {c['is_test'].sum()}, "
            f"{c['set'].nunique()} sets")
    c = pd.read_parquet(os.path.join(OUT, "p2", "cards.parquet"))
    fr = c["oracle_id"].map(df.set_index("oracle_id")["first_release"])
    folds = {k: sorted(c.loc[(fr >= SPLITS[k][1]) & (fr < SPLITS[k][2]), "set"].unique().tolist()) for k in VAL}
    json.dump(folds, open(os.path.join(OUT, "p2", "folds.json"), "w"), indent=1)
    log(f"folds: " + ", ".join(f"{k} {len(v)} sets" for k, v in folds.items()))


def import_ft(args) -> None:
    """phase2.py predictions on the Commander tables -> runs in the
    commander.py format: val (folds A, B; one run per epoch, <name>_e<k>) and
    test (<name>)."""
    df, _ = load()
    grp = df.set_index("oracle_id")["first_set"]
    y = df.set_index("oracle_id")["y"]
    rd = os.path.join(OUT, "p2", "runs", args.name)
    for fold in VAL:
        for x in sorted(os.listdir(rd)):
            if not x.startswith(f"val_pred_{fold}_seed"):
                continue
            seed = int(x.split("seed")[1].split(".")[0])
            p = pd.read_csv(os.path.join(rd, x))
            for ep, g in p.groupby("epoch"):
                d = os.path.join(RUNS, f"{args.name}_e{ep}")
                os.makedirs(d, exist_ok=True)
                g.assign(group=g["oracle_id"].map(grp), y=g["oracle_id"].map(y))[
                    ["oracle_id", "group", "y", "pred"]].to_csv(os.path.join(d, f"{fold}_seed{seed}.csv"), index=False)
    for x in sorted(os.listdir(rd)):
        if x.startswith("test_pred_seed"):
            p = pd.read_csv(os.path.join(rd, x))
            d = os.path.join(RUNS, args.name)
            os.makedirs(d, exist_ok=True)
            p.assign(group=p["oracle_id"].map(grp), y=p["oracle_id"].map(y))[
                ["oracle_id", "group", "y", "pred"]].to_csv(os.path.join(d, x.replace("test_pred_", "test_")), index=False)
    log(f"imported {args.name}")


# --------------------------------------------------------------------------- commands

ALL_MODELS = ["ridge_lim", "ridge_all", "ridge_sub", "ridge_struct", "ridge_struct_noage",
              "ridge_struct_a1", "ridge_struct_a3",
              "ridge_struct_lim", "ridge_struct_since2015", "ridge_struct_since2020",
              "ridge_struct_hl3", "ridge_struct_hl6",
              "gbt", "gbt_noage", "gbt_nosvd"]


def val(args) -> None:
    df, f = load()
    for s in VAL:
        run_models(df, f, s, args.models or ALL_MODELS)


def cmd_compare(args) -> None:
    splits = VAL if args.split == "val" else ["test"]
    pd.set_option("display.width", 220)
    for gf in args.groups:
        t = compare(args.runs, args.ref, splits, gf)
        print(f"\n== {args.split}, groups: {gf} (mean within-set Spearman; Δ vs {args.ref}) ==")
        print(t.round(4).to_string(index=False))


def test(args) -> None:
    df, f = load()
    run_models(df, f, "test", args.models)
    with open(os.path.join(ROOT, "out", "TEST_LOG.md"), "a") as fh:
        fh.write(f"| {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} | commander: "
                 f"{', '.join(args.models)} | {args.why} |\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    sub.add_parser("final")
    sub.add_parser("report")
    sub.add_parser("p2prep")
    p = sub.add_parser("import-ft")
    p.add_argument("name")
    p = sub.add_parser("val")
    p.add_argument("--models", nargs="*")
    p = sub.add_parser("compare")
    p.add_argument("runs", nargs="+")
    p.add_argument("--ref", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--groups", nargs="+", default=["all", "limited"], choices=["all", "limited", "nonlimited"])
    p = sub.add_parser("test")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--why", required=True)
    args = ap.parse_args()
    {"prep": prep, "val": val, "compare": cmd_compare, "test": test, "final": final, "report": report,
     "p2prep": p2prep, "import-ft": import_ft}[args.cmd](args)


if __name__ == "__main__":
    main()
