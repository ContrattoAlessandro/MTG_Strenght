"""Next step 3: Bayesian update of the card model with early 17lands play data.

    python scripts/early_update.py --prior-col "ens_embeddinggemma2_k4+gbt_tfidf"

The card model's prediction is the pre-release prior; the first D days of
PremierDraft games (D = 1, 2, 3, 5, 7) are the data. For each test set:

1. Target. A clean label from games played after day 7 only (Phase 0 GIH
   model, no archetypes; at most 800k games), so that no estimate is scored
   against the games it was fitted on. The official full-data label is
   reported too.
2. Estimates from the first D days of games:
   - early_only: the Phase 0 label model on those games (its own empirical
     Bayes shrinkage towards rarity x colour group means);
   - naive_update: precision-weighted mean of the prior and early_only
     (treats the early estimate's shrunk SD as a likelihood: approximate);
   - model_prior: the label model refitted with the card model inside the
     hierarchy, theta_c = mu_group + b * m_c + s_c * u_c, u_c ~ N(0, tau^2),
     where m_c is the model's prediction, s_c its predicted SD (relative to
     the set mean), and b and tau^2 are fitted on the early games. This is the
     proper Bayesian update: a card with few games stays near the model's
     prediction, a well-played card follows its data;
   - ata_D: average pick position over the same D days (17lands draft dump),
     and z-score averages of it with the estimates above.
3. Within-set metrics against both targets, averaged over the 7 test sets.

Writes out/early_update/{estimates.csv, metrics_per_set.csv, metrics_summary.csv}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mtgstrength import data, evaluate, labels  # noqa: E402

TEST_SETS = ["TLA", "ECL", "TMT", "SOS", "MSH", "HOB", "FRA"]
DAYS = [1, 2, 3, 5, 7]
LATE_START = 7      # target uses games from day LATE_START + 1 on
MAX_LATE_GAMES = 800_000


def game_days(path: str) -> tuple[pd.Timestamp, str, pd.Series]:
    """First day of PremierDraft (first day with >= 10% of the median daily
    game volume, which skips early-access play), the time column used, and
    the game times."""
    import pyarrow.csv as pacsv
    header = pd.read_csv(path, nrows=0).columns
    col = "game_time" if "game_time" in header else "draft_time"
    t = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(include_columns=[col]))
    times = pd.to_datetime(t.column(col).to_pandas())
    per_day = times.dt.floor("D").value_counts().sort_index()
    return per_day[per_day >= 0.1 * per_day.median()].index.min(), col, times


def window_filter(col: str, start: pd.Timestamp, end: pd.Timestamp | None, keep_p: float = 1.0,
                  seed: int = 0):
    rng = np.random.default_rng(seed)

    def f(batch):
        t = pd.to_datetime(batch.column(col).to_pandas()).to_numpy()
        m = t >= np.datetime64(start)
        if end is not None:
            m &= t < np.datetime64(end)
        if keep_p < 1:
            m &= rng.random(len(m)) < keep_p
        return m
    return f


def fit_labels(gd, meta, prior: pd.Series | None = None, prior_sd: pd.Series | None = None,
               em_iters: int = 6, group_means: bool = True) -> pd.DataFrame:
    """Phase 0 GIH label; with ``prior`` (indexed by card name, logit units),
    the card model enters the hierarchy (see module docstring). Without
    ``group_means`` the rarity x colour group means are replaced by one
    common mean, so the model prediction alone sets each card's prior."""
    y = gd.games["won"].to_numpy()
    nuis = labels.nuisance_block(gd.games)
    blocks = labels.card_blocks(gd, meta, "in_hand")
    if prior is not None:
        m = prior.reindex(gd.cards).to_numpy()
        has = ~np.isnan(m)
        m = np.where(has, m - np.nanmean(m), 0.0)
        H = blocks[1].X                       # delta_hand: in-hand matrix
        s = np.ones(len(gd.cards))
        if prior_sd is not None:
            sd = prior_sd.reindex(gd.cards).to_numpy()
            s = np.where(has & ~np.isnan(sd), sd / np.nanmean(sd), 1.0)
        pb = labels.Block("model_prior", ["b"], sp.csr_matrix(H @ m[:, None]), labels.WEAK_PRIOR_VAR)
        blocks[1] = labels.Block("delta_hand", gd.cards, sp.csr_matrix(H @ sp.diags(s)), 0.01,
                                 learn_var=True)
        blocks.insert(1, pb)
        if not group_means:
            blocks[0] = labels.Block("mu_hand", ["all"], sp.csr_matrix(H.sum(axis=1)),
                                     labels.WEAK_PRIOR_VAR)
    res = labels.fit(y, [nuis] + blocks, em_iters=em_iters, verbose=False)
    if prior is None:
        tab = labels.card_table(res, gd, meta, "in_hand")
        return tab.set_index("name")[["theta_vs_repl", "theta_hand_sd", "n_games_in_hand"]]
    # theta_c = mu_g + b m_c + s_c u_c, with its posterior SD.
    sl = res.slices
    groups = (meta["rarity"].fillna("unknown") + ":" + meta["color_group"]).to_numpy()
    if not group_means:
        groups = np.full(len(groups), "all")
    gnames = res.blocks[[b.name for b in res.blocks].index("mu_hand")].cols
    gi = np.array([gnames.index(g) for g in groups]) + sl["mu_hand"].start
    bi = sl["model_prior"].start
    di = np.arange(sl["delta_hand"].start, sl["delta_hand"].stop)
    theta = res.beta[gi] + res.beta[bi] * m + s * res.beta[di]
    C = res.cov
    var = (C[gi, gi] + m ** 2 * C[bi, bi] + s ** 2 * C[di, di] + 2 * m * C[gi, bi]
           + 2 * s * C[gi, di] + 2 * m * s * C[bi, di])
    out = pd.DataFrame({"theta_vs_repl": theta, "theta_hand_sd": np.sqrt(var),
                        "n_games_in_hand": gd.in_hand.sum(axis=0)}, index=gd.cards)
    out.attrs.update(b=float(res.beta[bi]), tau2=float(res.prior_vars["delta_hand"]))
    return out


def ata_by_day(path: str, days: list[int]) -> dict[int, pd.Series]:
    d = pd.read_csv(path, usecols=["draft_time", "pick_number", "pick"])
    d["day"] = pd.to_datetime(d["draft_time"]).dt.floor("D")
    per_day = d.groupby("day").size()
    day1 = per_day[per_day >= 0.1 * per_day.median()].index.min()
    out = {}
    for D in days:
        w = d[(d["day"] >= day1) & (d["day"] < day1 + pd.Timedelta(days=D))]
        out[D] = w.groupby("pick")["pick_number"].mean() + 1
    return out


def run_set(s: str, prior_tab: pd.DataFrame, args, log) -> pd.DataFrame:
    gpath = data.download(data.game_data_url(s), args.raw_dir)
    cpath = data.download(data.cards_url(), args.raw_dir)
    day1, col, times = game_days(gpath)
    log(f"{s}: day one {day1.date()} ({col})")
    extra = (col,)
    pt = prior_tab[prior_tab["set"] == s].set_index("name")
    rows = pt[["oracle_id", "rarity", "theta_vs_repl", "ame_vs_repl_pp", "prior", "prior_sd",
               "n_games_in_hand"]].rename(columns={"theta_vs_repl": "label_full",
                                                    "n_games_in_hand": "n_games_full"}).copy()

    # Target from late games only.
    t0 = time.time()
    late_start = day1 + pd.Timedelta(days=LATE_START)
    n_late = int((times >= late_start).sum())
    keep_p = min(1.0, MAX_LATE_GAMES / max(n_late, 1))
    gd = data.load_game_data(gpath, row_filter=window_filter(col, late_start, None, keep_p),
                             extra_cols=extra)
    meta = data.load_card_meta(cpath, s, gd.cards)
    data.infer_deck_colors(gd, meta)
    late = fit_labels(gd, meta)
    rows["label_late"] = late["theta_vs_repl"].reindex(rows.index)
    rows["label_late_sd"] = late["theta_hand_sd"].reindex(rows.index)
    log(f"{s}: late target from {len(gd.games):,} games ({time.time() - t0:.0f}s)")
    del gd

    ata = ata_by_day(os.path.join(args.draft_dir, f"draft_data_public.{s}.PremierDraft.csv.gz"), DAYS) \
        if args.draft_dir else {}
    for D in DAYS:
        t0 = time.time()
        gd = data.load_game_data(gpath, row_filter=window_filter(col, day1, day1 + pd.Timedelta(days=D)),
                                 extra_cols=extra)
        meta = data.load_card_meta(cpath, s, gd.cards)
        data.infer_deck_colors(gd, meta)
        early = fit_labels(gd, meta).reindex(rows.index)
        mp = fit_labels(gd, meta, prior=pt["prior"], prior_sd=pt["prior_sd"],
                        group_means=False).reindex(rows.index)
        mpg = fit_labels(gd, meta, prior=pt["prior"], prior_sd=pt["prior_sd"]).reindex(rows.index)
        rows[f"model_prior_grp_d{D}"] = mpg["theta_vs_repl"]
        rows[f"early_only_d{D}"] = early["theta_vs_repl"]
        rows[f"early_only_d{D}_sd"] = early["theta_hand_sd"]
        rows[f"games_in_hand_d{D}"] = early["n_games_in_hand"]
        rows[f"model_prior_d{D}"] = mp["theta_vs_repl"]
        rows[f"model_prior_d{D}_sd"] = mp["theta_hand_sd"]
        # Naive: precision-weighted average of prior and early estimate,
        # both centred on their set mean.
        p = rows["prior"] - rows["prior"].mean()
        e = early["theta_vs_repl"] - early["theta_vs_repl"].mean()
        wp, we = 1 / rows["prior_sd"] ** 2, 1 / early["theta_hand_sd"] ** 2
        rows[f"naive_update_d{D}"] = (wp * p + we * e) / (wp + we)
        if D in ata:
            a = -ata[D].reindex(rows.index)
            rows[f"ata_d{D}"] = a.fillna(a.min())
        log(json.dumps({"set": s, "days": D, "games": len(gd.games), "b": round(mp.attrs["b"], 3),
                        "tau2_resid": round(mp.attrs["tau2"], 4), "sec": round(time.time() - t0)}))
        del gd
    rows["set"] = s
    return rows.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--draft-dir", default="data/raw/draft")
    ap.add_argument("--pred-csv", default="out/phase2/test_predictions.csv")
    ap.add_argument("--prior-col", default="ens_embeddinggemma2_k4+gbt_tfidf")
    ap.add_argument("--prior-sd-col", default="ft_embeddinggemma2_k4_sd")
    ap.add_argument("--name", default="early_update")
    ap.add_argument("--sets", nargs="*", default=TEST_SETS)
    args = ap.parse_args()
    res = os.path.join(args.out_dir, args.name)
    os.makedirs(res, exist_ok=True)
    logf = open(os.path.join(res, "log.jsonl"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(json.dumps({"config": vars(args)}))
    p = pd.read_csv(args.pred_csv)
    p["prior"] = p[args.prior_col]
    p["prior_sd"] = p[args.prior_sd_col] if args.prior_sd_col in p else np.nan
    parts = []
    for s in args.sets:
        path = os.path.join(res, f"estimates_{s}.csv")
        if os.path.exists(path):
            parts.append(pd.read_csv(path))
            continue
        r = run_set(s, p, args, log)
        r.to_csv(path, index=False)
        parts.append(r)
    est = pd.concat(parts, ignore_index=True)
    est.to_csv(os.path.join(res, "estimates.csv"), index=False)
    summarize(est, res)


def summarize(est: pd.DataFrame, res: str) -> None:
    est = est.copy()
    models = ["prior"]
    for D in DAYS:
        cols = [f"early_only_d{D}", f"naive_update_d{D}", f"model_prior_d{D}",
                f"model_prior_grp_d{D}"]
        if f"ata_d{D}" in est:
            g = est.groupby("set")
            for c in ["prior", f"early_only_d{D}", f"model_prior_d{D}"]:
                name = f"{c}+ata_d{D}"
                est[name] = g[c].transform(lambda x: (x - x.mean()) / x.std()) + \
                    g[f"ata_d{D}"].transform(lambda x: (x - x.mean()) / x.std())
                cols.append(name)
            cols.append(f"ata_d{D}")
        models += cols
    rows = []
    for target in ["label_late", "label_full"]:
        ev = est.assign(theta_vs_repl=est[target]).dropna(subset=["theta_vs_repl"])
        for m in models:
            ev[m + "_pp"] = np.nan
        ev["ame_vs_repl_pp"] = np.nan
        per = evaluate.evaluate(ev, models, target)
        rows.append(per)
    per = pd.concat(rows)
    per.to_csv(os.path.join(res, "metrics_per_set.csv"), index=False)
    summ = (per.groupby(["subset", "model"], sort=False)[["spearman", "spearman_within_rarity",
                                                          "ndcg@20", "top20_overlap"]]
            .mean().round(3))
    summ.to_csv(os.path.join(res, "metrics_summary.csv"))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(summ)


if __name__ == "__main__":
    main()
