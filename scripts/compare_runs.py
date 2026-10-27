"""Compare Phase 2 runs on the test sets (seed-averaged predictions), with
per-seed spread, and optionally their 50/50 average with gbt_tfidf.

    python scripts/compare_runs.py s_base s_lam05 s_env --ens
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mtgstrength import evaluate  # noqa: E402

METRICS = ["spearman", "spearman_within_rarity", "ndcg@20", "top20_overlap"]


def load_run(run_dir: str) -> list[pd.DataFrame]:
    return [pd.read_csv(os.path.join(run_dir, x)) for x in sorted(os.listdir(run_dir))
            if x.startswith("test_pred_seed")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--ens", action="store_true", help="also score avg(run, gbt_tfidf)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    base = pd.read_csv(os.path.join(args.out_dir, "phase1", "test_predictions.csv"))
    base = base[["set", "oracle_id", "rarity", "theta_vs_repl", "ame_vs_repl_pp", "gbt_tfidf"]]
    rows = []

    def score(df, col, label, seed_sd=np.nan):
        ev = df.assign(**{col + "_pp": np.nan})
        per = evaluate.evaluate(ev, [col], label)
        r = per[METRICS].mean().to_dict()
        r.update(model=label, seeds_spearman_sd=seed_sd)
        r.update({f"sp_{s}": v for s, v in zip(per["set"], per["spearman"])})
        rows.append(r)

    score(base, "gbt_tfidf", "gbt_tfidf")
    for name in args.runs:
        ps = load_run(os.path.join(args.out_dir, "phase2", "runs", name))
        if not ps:
            continue
        per_seed = []
        for p in ps:
            d = base.merge(p[["set", "oracle_id", "pred"]], on=["set", "oracle_id"])
            per_seed.append(evaluate.evaluate(d.assign(pred_pp=np.nan), ["pred"], "x")["spearman"].mean())
        m = pd.concat(ps).groupby(["set", "oracle_id"])["pred"].mean().rename("pred").reset_index()
        d = base.merge(m, on=["set", "oracle_id"])
        score(d, "pred", f"{name} ({len(ps)} seeds)", float(np.std(per_seed)))
        if args.ens:
            d["ens"] = 0.5 * d["pred"] + 0.5 * d["gbt_tfidf"]
            score(d, "ens", f"avg({name}, gbt_tfidf)")
    out = pd.DataFrame(rows).set_index("model").round(3)
    pd.set_option("display.width", 250)
    print(out)
    if args.csv:
        out.to_csv(args.csv)


if __name__ == "__main__":
    main()
