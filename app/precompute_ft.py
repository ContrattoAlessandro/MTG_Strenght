"""Score every Scryfall card with the fine-tuned EmbeddingGemma-2 model.

    python app/precompute_ft.py                        # CPU, ~2 h on 4 cores
    python app/precompute_ft.py --device cuda          # GPU (Kaggle T4: minutes)

Writes out/app/ft_scores.parquet (oracle_id, ft_pred, ft_sd), in a typical
recent environment, the same one the app uses for cards without 17lands
data. The file is small and kept in git; app/build.py averages it with the
GBT, so search results and card pages show the Gemma + GBT score. The run
saves after every chunk and resumes where it stopped, so it can be
interrupted and restarted.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
from engine import ROOT, TYPICAL  # noqa: E402

DATA = os.path.join(ROOT, "app", "data")
OUT = os.path.join(ROOT, "out", "app", "ft_scores.parquet")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", default=os.path.join(ROOT, "models", "ft_card"),
                    help="folder with the fine-tuned weights (<run>/seed*.pt)")
    ap.add_argument("--device", default="auto", help="cpu, cuda or auto")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=1000, help="cards between saves")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=None, help="only the first N cards (testing)")
    args = ap.parse_args()

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    paths = engine.find_ft_weights(args.ft_dir)
    if not paths:
        sys.exit(f"no seed*.pt files under {args.ft_dir}")
    if not os.path.exists(os.path.join(DATA, "cards.parquet")):
        log("app/data/ is missing: building it first")
        import build
        build.main()
    b = joblib.load(os.path.join(DATA, "bundle.joblib"))
    cards = pd.read_parquet(os.path.join(DATA, "cards.parquet"))
    if args.limit:
        cards = cards.head(args.limit)
    run = os.path.basename(os.path.dirname(paths[0]))
    done = pd.DataFrame(columns=["oracle_id", "ft_pred", "ft_sd", "ft_run"])
    if os.path.exists(args.out):
        done = pd.read_parquet(args.out)
        done = done[done["ft_run"] == run]
    todo = cards[~cards["oracle_id"].isin(done["oracle_id"])].reset_index(drop=True)
    log(f"{len(paths)} seeds of {run} on {device}; {len(done)} cards done, {len(todo)} to go")
    if todo.empty:
        return
    ft = engine.FineTuned(paths, args.threads, device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    for start in range(0, len(todo), args.chunk):
        u = todo.iloc[start:start + args.chunk].reset_index(drop=True)
        recs = u.to_dict("records")
        for r in recs:
            r["keywords"] = [k for k in (r.get("keywords") or "").split(", ") if k]
        df = engine.normalise(recs, b["kw_vocab"])
        df["front_name"] = u["front_name"].to_numpy()
        mu, sd = ft.predict(df, lambda cols: engine.featurize(df, [TYPICAL] * len(df), b, cols, ft=True),
                            batch=args.batch)
        new = pd.DataFrame({"oracle_id": u["oracle_id"], "ft_pred": mu, "ft_sd": sd, "ft_run": run})
        done = pd.concat([done, new], ignore_index=True)
        done.to_parquet(args.out, index=False)
        n = start + len(u)
        rate = n / (time.time() - t0)
        log(f"{n} / {len(todo)} cards ({rate:.1f}/s, ~{(len(todo) - n) / rate / 60:.0f} min left)")
    log(f"done: {args.out} ({len(done)} cards). Run `python app/build.py` to use them.")


if __name__ == "__main__":
    main()
