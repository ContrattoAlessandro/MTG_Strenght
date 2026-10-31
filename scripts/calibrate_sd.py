"""Post-hoc calibration of a fine-tuned run's predicted SD (plan, step G).

    python scripts/calibrate_sd.py --val-run d3_c00 --epoch 10 \
        --weights models/ft_card/g_final --scores out/app/ft_scores.parquet

With 10 epochs on top-12 layers the Gaussian head becomes overconfident: its
predicted SD (on top of the label SD) is ~6x too small. One scale factor k is
fitted on the validation folds of the same recipe, so that
(theta - pred) / sqrt((k * pred_sd)^2 + label_sd^2) has variance 1, and is
written into each seed*.pt as meta["sd_scale"] (app/engine.py applies it) and
applied to the ft_sd column of precomputed scores. Prints test coverage as a
check (no selection is made on it).
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def zstats(d: pd.DataFrame, k: float) -> tuple[float, float]:
    z = (d["theta_vs_repl"] - d["pred"]) / np.sqrt((k * d["pred_sd"]) ** 2 + d["theta_hand_sd"] ** 2)
    return float(z.std()), float(np.mean(np.abs(z) < 1.645))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--val-run", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--test-run", default=None, help="run with test_pred_seed*.csv, for the coverage check")
    ap.add_argument("--weights", default=None, help="folder with seed*.pt to stamp with sd_scale")
    ap.add_argument("--scores", nargs="*", default=[], help="parquet files whose ft_sd is rescaled")
    args = ap.parse_args()
    runs = os.path.join(args.out_dir, "phase2", "runs")
    cards = pd.read_parquet(os.path.join(args.out_dir, "phase2", "cards.parquet"))[
        ["set", "oracle_id", "theta_vs_repl", "theta_hand_sd"]]
    v = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(runs, args.val_run, "val_pred_V*_seed*.csv"))])
    v = v[v["epoch"] == args.epoch].groupby(["set", "oracle_id"])[["pred", "pred_sd"]].mean().reset_index()
    v = v.merge(cards)
    r2 = (v["theta_vs_repl"] - v["pred"]) ** 2
    k = brentq(lambda k: np.mean(r2 / ((k * v["pred_sd"]) ** 2 + v["theta_hand_sd"] ** 2)) - 1, 0.05, 1000)
    print(f"sd_scale k = {k:.3f}; val z SD {zstats(v, 1)[0]:.2f} -> {zstats(v, k)[0]:.2f}, "
          f"90% coverage {zstats(v, k)[1]:.3f}")
    if args.test_run:
        t = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(runs, args.test_run, "test_pred_seed*.csv"))])
        t = t.groupby(["set", "oracle_id"])[["pred", "pred_sd"]].mean().reset_index().merge(cards)
        print(f"test check: z SD {zstats(t, k)[0]:.2f}, 90% coverage {zstats(t, k)[1]:.3f}")
        import json
        json.dump({"sd_scale": k, "val_run": args.val_run, "epoch": args.epoch},
                  open(os.path.join(runs, args.test_run, "sd_scale.json"), "w"))
    if args.weights:
        import torch
        for p in sorted(glob.glob(os.path.join(args.weights, "seed*.pt"))):
            meta = torch.load(p, weights_only=False)
            meta["sd_scale"] = k
            torch.save(meta, p)
            print(f"stamped {p}")
    for p in args.scores:
        s = pd.read_parquet(p)
        if "sd_scale" in s.attrs or "ft_sd_raw" in s:
            print(f"{p}: already rescaled, skipped")
            continue
        s["ft_sd_raw"] = s["ft_sd"]
        s["ft_sd"] = s["ft_sd"] * k
        s.to_parquet(p)
        print(f"rescaled {p}")


if __name__ == "__main__":
    main()
