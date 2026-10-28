"""Behavioural probes of the final card model (docs/RESEARCH_PLAN.md, step G;
design section 5.5).

    python app/build.py                                   # bundle with ft_curve / ft_ctx
    python scripts/behaviour_probes.py --ft-dir models/ft_card [--device cuda] [--sample 300]

Edits each card and re-scores it through the app pipeline (engine.normalise ->
featurize -> model), once with the fine-tuned model and once with the app's
gbt_tfidf, so both models see exactly the same edited cards:

- power +1, toughness +1 (creatures with numeric P/T): a better card, so the
  prediction should not fall;
- one less generic mana (cards with {N} in the cost, N >= 1): same.
A violation is a drop of more than --tol logits. Also, without edits:
- reprints: for cards labelled in two or more sets, does the change in the
  prediction between environments follow the change in the label?
- new mechanics: test-set cards with a Scryfall keyword that no training-set
  card has; is the predicted SD larger for them, and does it track the error?

Writes out/probes/probe_cards.csv and out/probes/summary.json. A high
violation rate re-opens the monotonicity regularizer (plan, parked items).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "app"))

import engine  # noqa: E402


def edit_power(c: dict, d: str) -> dict | None:
    if "Creature" not in (c["type_line"] or "") or not re.fullmatch(r"\d+", str(c[d] or "")):
        return None
    return {**c, d: str(int(c[d]) + 1)}


def edit_cost(c: dict) -> dict | None:
    m = re.match(r"^\{(\d+)\}", c["mana_cost"] or "")
    if not m or int(m.group(1)) < 1 or " // " in (c["mana_cost"] or ""):
        return None
    n = int(m.group(1)) - 1
    return {**c, "mana_cost": ("{%d}" % n if n else "") + c["mana_cost"][m.end():], "cmc": None}


EDITS = {"power+1": lambda c: edit_power(c, "power"),
         "toughness+1": lambda c: edit_power(c, "toughness"),
         "generic-1": edit_cost}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    ap.add_argument("--ft-dir", default=os.path.join(ROOT, "models", "ft_card"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sample", type=int, default=0, help="probe a random sample of the test cards")
    ap.add_argument("--tol", type=float, default=0.01, help="violation = drop of more than tol logits")
    ap.add_argument("--oracle", default=os.path.join(ROOT, "data", "raw", "scryfall_oracle_cards.jsonl.gz"))
    args = ap.parse_args()
    import joblib
    bundle = joblib.load(os.path.join(ROOT, "app", "data", "bundle.joblib"))
    paths = engine.find_ft_weights(args.ft_dir)
    if not paths:
        sys.exit(f"no seed*.pt under {args.ft_dir}")
    ft = engine.FineTuned(paths, threads=args.threads, device=args.device)
    cards = pd.read_parquet(os.path.join(args.out_dir, "phase2", "cards.parquet"))
    base = cards.loc[cards["is_test"]].copy()
    if args.sample:
        base = base.sample(min(args.sample, len(base)), random_state=0)

    def as_dicts(df):
        return [{"name": r.name_, "mana_cost": r.mana_cost, "type_line": r.type_line,
                 "oracle_text": r.oracle_text, "power": r.power, "toughness": r.toughness,
                 "rarity": r.rarity, "cmc": r.cmc}
                for r in df.rename(columns={"name": "name_"}).itertuples()]

    def score(dicts, sets):
        df = engine.normalise(dicts, bundle["kw_vocab"])
        mu, sd = ft.predict(df, lambda cols: engine.featurize(df, sets, bundle, cols, ft=True),
                            batch=args.batch)
        g, _ = engine.limited_gbt(df, engine.featurize(df, sets, bundle), bundle)
        return mu, sd, g

    orig = as_dicts(base)
    sets = base["set"].tolist()
    mu0, sd0, g0 = score(orig, sets)
    out = base[["set", "oracle_id", "name", "rarity", "theta_vs_repl"]].assign(ft=mu0, ft_sd=sd0, gbt=g0)
    summary = {"n_cards": len(base), "tol": args.tol, "weights": paths}
    for name, fn in EDITS.items():
        new = [fn(c) for c in orig]
        idx = [i for i, c in enumerate(new) if c is not None]
        mu, _, g = score([new[i] for i in idx], [sets[i] for i in idx])
        d_ft, d_g = np.full(len(orig), np.nan), np.full(len(orig), np.nan)
        d_ft[idx], d_g[idx] = mu - mu0[idx], g - g0[idx]
        out[f"d_ft_{name}"], out[f"d_gbt_{name}"] = d_ft, d_g
        summary[name] = {"n": len(idx),
                         "ft_violation_rate": float(np.mean(d_ft[idx] < -args.tol)),
                         "ft_mean_delta": float(np.mean(d_ft[idx])),
                         "gbt_violation_rate": float(np.mean(d_g[idx] < -args.tol)),
                         "gbt_mean_delta": float(np.mean(d_g[idx]))}
        print(name, summary[name], flush=True)

    # Reprints: the same card labelled in several sets (training sets included).
    multi = cards[cards.groupby("oracle_id")["set"].transform("nunique") > 1]
    if args.sample:   # quick runs: a few reprinted cards only
        multi = multi[multi["oracle_id"].isin(multi["oracle_id"].unique()[:max(3, args.sample // 10)])]
    if len(multi):
        mu, _, g = score(as_dicts(multi), multi["set"].tolist())
        m = multi[["oracle_id", "set", "theta_vs_repl"]].assign(ft=mu, gbt=g)
        dev = m.groupby("oracle_id")[["theta_vs_repl", "ft", "gbt"]].transform(lambda x: x - x.mean())
        summary["reprints"] = {"n_rows": len(m), "n_cards": int(m["oracle_id"].nunique()),
                               "corr_ft": float(np.corrcoef(dev["theta_vs_repl"], dev["ft"])[0, 1]),
                               "corr_gbt": float(np.corrcoef(dev["theta_vs_repl"], dev["gbt"])[0, 1]),
                               "sd_label_dev": float(dev["theta_vs_repl"].std()),
                               "sd_ft_dev": float(dev["ft"].std())}
        print("reprints", summary["reprints"], flush=True)

    # New mechanics: a Scryfall keyword that no training-set card has.
    if os.path.exists(args.oracle):
        kw = {}
        with gzip.open(args.oracle, "rt") as fh:
            for line in fh:
                c = json.loads(line)
                kw[c.get("oracle_id")] = set(c.get("keywords") or [])
        seen = set().union(*(kw.get(o, set()) for o in cards.loc[cards["is_train"], "oracle_id"]))
        out["new_mechanic"] = [bool(kw.get(o, set()) - seen) for o in out["oracle_id"]]
        err = (out["ft"] - out["theta_vs_repl"]).abs()
        nm = out["new_mechanic"].to_numpy()
        summary["new_mechanics"] = {
            "n_new": int(nm.sum()),
            "keywords": sorted(set().union(*(kw.get(o, set()) for o in out["oracle_id"])) - seen),
            "ft_sd_new": float(out.loc[nm, "ft_sd"].mean()), "ft_sd_old": float(out.loc[~nm, "ft_sd"].mean()),
            "abs_err_new": float(err[nm].mean()), "abs_err_old": float(err[~nm].mean()),
            "spearman_sd_abs_err": float(spearmanr(out["ft_sd"], err).statistic)}
        print("new mechanics", summary["new_mechanics"], flush=True)
    res = os.path.join(args.out_dir, "probes")
    os.makedirs(res, exist_ok=True)
    out.to_csv(os.path.join(res, "probe_cards.csv"), index=False)
    with open(os.path.join(res, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {res}/")


if __name__ == "__main__":
    main()
