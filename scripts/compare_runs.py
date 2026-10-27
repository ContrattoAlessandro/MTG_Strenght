"""Compare Phase 2 runs with paired per-set statistics (docs/RESEARCH_PLAN.md).

    # validation (default): runs trained with phase2.py train --val-fold ...
    python scripts/compare_runs.py ft_a ft_b --ref ft_a
    python scripts/compare_runs.py ft_a --epoch 5 --folds V3
    # test sets: an explicit, logged look (appends a line to out/TEST_LOG.md)
    python scripts/compare_runs.py s_base s_env --test --why "screening, Kaggle v7" --ens

Validation scores come from val_pred_<fold>_seed<k>.csv (every epoch). Each
run is scored at one epoch for all folds: --epoch N, or by default the epoch
with the best mean Spearman over all its validation sets (seed-averaged
predictions); the chosen epoch is printed. Choosing the epoch on the same sets
favours every run a little, equally.

With --ref, each run gets Δ vs the reference: mean over sets ± SE across sets
(paired), the number of sets won, and a 95% interval from a card-level
bootstrap within each set (the same resampled cards for both runs). The rule
for a real difference is Δ > 2 SE and won on at least 2/3 of the sets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mtgstrength import evaluate  # noqa: E402

METRICS = ["spearman", "spearman_within_rarity", "ndcg@20", "top20_overlap"]
LABEL_COLS = ["set", "oracle_id", "rarity", "theta_vs_repl", "ame_vs_repl_pp"]


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])


def load_test(run_dir: str) -> list[pd.DataFrame]:
    return [pd.read_csv(os.path.join(run_dir, x)) for x in sorted(os.listdir(run_dir))
            if x.startswith("test_pred_seed")]


def load_val(run_dir: str, folds: list[str]) -> pd.DataFrame:
    """All validation predictions of a run: set, oracle_id, epoch, pred, fold, seed."""
    parts = []
    for x in sorted(os.listdir(run_dir)):
        m = re.fullmatch(r"val_pred_(V\d)_seed(\d+)\.csv", x)
        if m and m.group(1) in folds:
            parts.append(pd.read_csv(os.path.join(run_dir, x)).assign(fold=m.group(1),
                                                                      seed=int(m.group(2))))
    return pd.concat(parts) if parts else pd.DataFrame()


def per_set(d: pd.DataFrame, col: str) -> pd.DataFrame:
    return evaluate.evaluate(d.assign(**{col + "_pp": np.nan}), [col], "x").set_index("set")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--ref", default=None, help="reference run for paired deltas")
    ap.add_argument("--test", action="store_true", help="score the test sets (logged)")
    ap.add_argument("--why", default=None, help="reason for a test look (required with --test)")
    ap.add_argument("--folds", nargs="+", default=["V1", "V2", "V3"])
    ap.add_argument("--epoch", type=int, default=None, help="validation epoch (default: best)")
    ap.add_argument("--ens", action="store_true", help="test only: also avg(run, gbt_tfidf)")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap replicates (0 = off)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    if args.test and not args.why:
        ap.error("--test needs --why: every test look goes into out/TEST_LOG.md")
    runs_dir = os.path.join(args.out_dir, "phase2", "runs")
    cards = pd.read_parquet(os.path.join(args.out_dir, "phase2", "cards.parquet"))[LABEL_COLS]
    names = list(dict.fromkeys(args.runs + ([args.ref] if args.ref else [])))

    preds, info = {}, {}   # name -> DataFrame(set, oracle_id, pred); name -> (seeds, seed SD, epoch)
    if args.test:
        base = pd.read_csv(os.path.join(args.out_dir, "phase1", "test_predictions.csv"))
        preds["gbt_tfidf"] = base[["set", "oracle_id", "gbt_tfidf"]].rename(columns={"gbt_tfidf": "pred"})
        info["gbt_tfidf"] = (1, np.nan, None)
        for name in names:
            ps = load_test(os.path.join(runs_dir, name))
            if not ps:
                print(f"{name}: no test predictions")
                continue
            sd = np.std([per_set(cards.merge(p[["set", "oracle_id", "pred"]]), "pred")["spearman"].mean()
                         for p in ps])
            m = pd.concat(ps).groupby(["set", "oracle_id"])["pred"].mean().reset_index()
            preds[name], info[name] = m, (len(ps), sd, None)
            if args.ens:
                e = m.merge(preds["gbt_tfidf"], on=["set", "oracle_id"], suffixes=("", "_g"))
                preds[f"avg({name}, gbt_tfidf)"] = e.assign(pred=0.5 * e["pred"] + 0.5 * e["pred_g"])
                info[f"avg({name}, gbt_tfidf)"] = (len(ps), np.nan, None)
    else:
        for name in names:
            v = load_val(os.path.join(runs_dir, name), args.folds)
            if v.empty:
                print(f"{name}: no validation predictions for folds {args.folds}")
                continue
            mean = v.groupby(["set", "oracle_id", "epoch"])["pred"].mean().reset_index()
            curve = {ep: per_set(cards.merge(g), "pred")["spearman"].mean()
                     for ep, g in mean.groupby("epoch")}
            ep = args.epoch or max(curve, key=curve.get)
            vv = v[v["epoch"] == ep]
            sd = np.std([per_set(cards.merge(g), "pred")["spearman"].mean()
                         for _, g in vv.groupby("seed")])
            n_seeds = vv["seed"].nunique()
            preds[name] = mean[mean["epoch"] == ep].drop(columns="epoch")
            info[name] = (n_seeds, sd, ep)
            print(f"{name}: epoch curve " + " ".join(f"{e}:{s:.3f}" for e, s in sorted(curve.items()))
                  + f" -> epoch {ep}")

    # Restrict every model to the cards all of them predict, so deltas are paired.
    common = None
    for p in preds.values():
        k = set(zip(p["set"], p["oracle_id"]))
        common = k if common is None else common & k
    rows, by_set = [], {}
    for name, p in preds.items():
        d = cards.merge(p[["set", "oracle_id", "pred"]])
        d = d[[k in common for k in zip(d["set"], d["oracle_id"])]]
        ps = per_set(d, "pred")
        by_set[name] = d
        n, sd, ep = info[name]
        r = ps[METRICS].mean().to_dict()
        r.update(model=name, seeds=n, seed_sd=sd, epoch=ep, n_sets=len(ps))
        r.update({f"sp_{s}": v for s, v in ps["spearman"].items()})
        rows.append(r)
    out = pd.DataFrame(rows).set_index("model")

    if args.ref and args.ref in by_set:
        ref = by_set[args.ref]
        rng = np.random.default_rng(0)
        for name, d in by_set.items():
            if name == args.ref:
                continue
            j = ref.merge(d[["set", "oracle_id", "pred"]], on=["set", "oracle_id"], suffixes=("_r", ""))
            dl = {s: spearman(g["theta_vs_repl"], g["pred"]) - spearman(g["theta_vs_repl"], g["pred_r"])
                  for s, g in j.groupby("set")}
            dv = np.array(list(dl.values()))
            out.loc[name, "d_spearman"] = dv.mean()
            out.loc[name, "d_se"] = dv.std(ddof=1) / np.sqrt(len(dv))
            out.loc[name, "sets_won"] = f"{(dv > 0).sum()}/{len(dv)}"
            dn = (out.loc[name, "ndcg@20"] - out.loc[args.ref, "ndcg@20"])
            out.loc[name, "d_ndcg20"] = dn
            if args.boot:
                groups = [(g["theta_vs_repl"].to_numpy(), g["pred"].to_numpy(), g["pred_r"].to_numpy())
                          for _, g in j.groupby("set")]
                bs = []
                for _ in range(args.boot):
                    tot = 0.0
                    for y, a, b in groups:
                        i = rng.integers(0, len(y), len(y))
                        tot += spearman(y[i], a[i]) - spearman(y[i], b[i])
                    bs.append(tot / len(groups))
                out.loc[name, "boot95"] = f"[{np.quantile(bs, 0.025):+.3f}, {np.quantile(bs, 0.975):+.3f}]"
            ok = dv.mean() > 2 * out.loc[name, "d_se"] and (dv > 0).sum() >= 2 * len(dv) / 3
            out.loc[name, "verdict"] = "better" if ok else (
                "worse" if -dv.mean() > 2 * out.loc[name, "d_se"] and (dv < 0).sum() >= 2 * len(dv) / 3
                else "tie")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(f"split: {'TEST' if args.test else 'validation ' + ' '.join(args.folds)}")
    print(out.round(4))
    if args.csv:
        out.to_csv(args.csv)
    if args.test:
        log = os.path.join(args.out_dir, "TEST_LOG.md")
        new = not os.path.exists(log)
        with open(log, "a") as fh:
            if new:
                fh.write("# Test-set access log\n\nOne line per look at the 7 test sets "
                         "(docs/RESEARCH_PLAN.md, rule 2).\n\n| date (UTC) | runs | why |\n|---|---|---|\n")
            now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
            fh.write(f"| {now} | {', '.join(names)} | {args.why} |\n")
        print(f"logged in {log}")


if __name__ == "__main__":
    main()
