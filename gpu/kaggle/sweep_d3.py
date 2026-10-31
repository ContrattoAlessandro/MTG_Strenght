"""Write gpu/kaggle/lanes.json for the step D3 hyperparameter sweep
(docs/RESEARCH_PLAN.md): N configs drawn without replacement from the grid
below (fixed seed, so the draw is reproducible and pre-registered), 1 seed x
3 validation folds each, split round-robin over the two T4 lanes. Config 0 is
always the current default, so the sweep contains its own reference.

    python gpu/kaggle/sweep_d3.py --base "--top-k 4" --n 20
"""

import argparse
import itertools
import json
import os
import random

GRID = {
    "enc-lr": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
    "head-lr": [3e-4, 1e-3, 3e-3],
    "pair-w": [0.3, 1.0, 3.0],
    "dropout": [0.05, 0.1, 0.2],
    "batch-size": [16, 32, 64],
    "weight-decay": [0.01, 0.1],
}
DEFAULT = {"enc-lr": 5e-5, "head-lr": 1e-3, "pair-w": 1.0, "dropout": 0.1, "batch-size": 32,
           "weight-decay": 0.01}
FOLD = ("python scripts/phase2.py train --encoder google/embeddinggemma-2 --val-fold V1 V2 V3"
        " --max-epochs 8 --device cuda --cache-dir {cache} --seeds 0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="--top-k 4", help="flags fixed by D1/D2")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--lanes", type=int, default=2)
    args = ap.parse_args()
    keys = list(GRID)
    allc = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    allc = [c for c in allc if c != DEFAULT]
    random.Random(0).shuffle(allc)
    configs = [DEFAULT] + allc[: args.n - 1]
    lanes = [[] for _ in range(args.lanes)]
    rows = []
    for i, c in enumerate(configs):
        flags = " ".join(f"--{k} {v:g}" for k, v in c.items())
        lanes[i % args.lanes].append(f"{FOLD} {args.base} {flags} --name d3_c{i:02d}")
        rows.append({"name": f"d3_c{i:02d}", **c})
    path = os.path.join(os.path.dirname(__file__), "lanes.json")
    json.dump(lanes, open(path, "w"), indent=1)
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "sweep_d3_configs.json"), "w"), indent=1)
    print(f"wrote {path}: {len(configs)} configs")


if __name__ == "__main__":
    main()
