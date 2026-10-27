"""Human baselines for the test sets (design section 5.4).

    python scripts/human_baselines.py

1. Draftsim pre-release ratings (0-5, set by Draftsim staff before release and
   used by their draft bots). Draftsim revises them after release, so the
   current files would leak game data. The raw files in data/raw/human/ are
   Wayback Machine snapshots: for each set, the latest one before the first
   day of PremierDraft on Arena (TMT's is from the morning of launch day;
   it is the only one archived).

   | set | snapshot | file |
   |---|---|---|
   | TLA | 2025-11-10 | draftsim.com/generated/TLA.js?v=1.01 |
   | ECL | 2026-01-14 | draftsim.com/generated/ECL.js?v=1.03 |
   | TMT | 2026-03-03 | draftsim.com/generated/TMT.js?v=1.03 |
   | SOS | 2026-04-16 | draftsim.com/generated/SOS/SOS.js?v=1.09 |
   | MSH | 2026-06-14 | ratings table inside draft-app/dist/assets/index-OGnbEQqb.js |
   | HOB | 2026-08-07 | ratings table inside draft-app/dist/assets/index-DmeA26Yz.js |
   | FRA | 2026-09-25 | ratings table inside draft-app/dist/assets/index-DPnGdPQd.js |

   FRA_ratings_2026-09-22.txt is an earlier FRA snapshot, kept for reference:
   Draftsim changed most FRA ratings between 22 and 25 September.

2. First-week ATA (average pick position, lower = better) over the first 7
   days of PremierDraft, computed from the 17lands public draft dumps (the
   card_ratings endpoint no longer returns date-filtered data for these
   sets). It is the drafting community's consensus after a week, when much
   less game data exists, so it is a stronger and less clean baseline than a
   pre-release grade.

Writes out/phase2/human_ratings.csv keyed by (set, 17lands name).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TEST_SETS = ["TLA", "ECL", "TMT", "SOS", "MSH", "HOB", "FRA"]
def norm(name: str) -> str:
    name = name.replace("_", " ").split(" // ")[0].lower()
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def draftsim_js(path: str) -> pd.DataFrame:
    txt = open(path, encoding="utf-8").read()
    rows = re.findall(r'name:"((?:[^"\\]|\\.)*)".*?myrating:"([-\d.]+)"', txt)
    d = pd.DataFrame(rows, columns=["ds_name", "draftsim"])
    # One file has a typo ("2..8").
    d["draftsim"] = pd.to_numeric(d["draftsim"].str.replace("..", ".", regex=False))
    return d


def draftsim_txt(path: str) -> pd.DataFrame:
    t = pd.read_csv(path, sep="\t")
    # The rating is the 7th column; its header is empty in some versions.
    return pd.DataFrame({"ds_name": t["Name"], "draftsim": pd.to_numeric(t.iloc[:, 6])})


def first_week_ata(s: str, raw_dir: str, days: int = 7) -> pd.DataFrame:
    """ATA (average pick 1-14 within the pack) over the first ``days`` days
    of PremierDraft, from the 17lands public draft dump. Day one is the first
    day with at least 10% of the median daily pick volume, which skips
    early-access drafts."""
    path = os.path.join(raw_dir, f"draft_data_public.{s}.PremierDraft.csv.gz")
    if not os.path.exists(path):
        url = ("https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/"
               f"draft_data_public.{s}.PremierDraft.csv.gz")
        urllib.request.urlretrieve(url, path)
    parts = []
    for chunk in pd.read_csv(path, usecols=["draft_time", "pick_number", "pick"], chunksize=500_000):
        parts.append(chunk)
    d = pd.concat(parts, ignore_index=True)
    d["day"] = pd.to_datetime(d["draft_time"]).dt.floor("D")
    per_day = d.groupby("day").size()
    day1 = per_day[per_day >= 0.1 * per_day.median()].index.min()
    w = d[(d["day"] >= day1) & (d["day"] < day1 + pd.Timedelta(days=days))]
    g = w.groupby("pick")["pick_number"].agg(["mean", "size"])
    print(f"{s}: week-one window starts {day1.date()}, {len(w)} picks", flush=True)
    return pd.DataFrame({"l_name": g.index, "ata_week1": g["mean"].to_numpy() + 1,
                         "pick_count_week1": g["size"].to_numpy()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/human")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--draft-dir", default="data/raw/draft",
                    help="where the 17lands draft dumps are cached")
    args = ap.parse_args()
    os.makedirs(args.draft_dir, exist_ok=True)
    frames = []
    for s in TEST_SETS:
        js = os.path.join(args.raw_dir, f"{s}_set.js")
        ds = draftsim_js(js) if os.path.exists(js) else draftsim_txt(
            os.path.join(args.raw_dir, f"{s}_ratings.txt"))
        ds = ds[ds["draftsim"] >= 0].assign(key=lambda d: d["ds_name"].map(norm))
        ds = ds.drop_duplicates("key")
        labels = pd.read_csv(os.path.join(args.out_dir, s, "card_labels.csv"), usecols=["name"])
        labels["key"] = labels["name"].map(norm)
        ata = first_week_ata(s, args.draft_dir).assign(key=lambda d: d["l_name"].map(norm)).drop_duplicates("key")
        m = labels.merge(ds[["key", "draftsim"]], on="key", how="left") \
                  .merge(ata[["key", "ata_week1", "pick_count_week1"]], on="key", how="left")
        m["set"] = s
        print(f"{s}: {len(labels)} label rows, draftsim matched {m['draftsim'].notna().sum()}, "
              f"ATA matched {m['ata_week1'].notna().sum()}", flush=True)
        frames.append(m.drop(columns="key"))
    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(args.out_dir, "phase2", "human_ratings.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
