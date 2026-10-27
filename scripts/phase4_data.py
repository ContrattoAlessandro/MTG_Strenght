"""Phase 4 data: Commander and Constructed play for every card with a label.

    python scripts/phase4_data.py edhrec
    python scripts/phase4_data.py goldfish

- edhrec: EDHREC's per-card JSON (json.edhrec.com/pages/cards/<slug>.json)
  gives the number of decks that play the card and the number that could
  (its colour identity fits). Inclusion = num_decks / potential_decks.
  Written to out/phase4/edhrec.csv. Resumable.
- goldfish: MTGGoldfish "format staples" pages list the top 50 cards of a
  format by share of decks (overall, and separately for creatures, spells
  and lands). Archived copies (Wayback Machine, one per
  month) from 2021 on give, for every card, its highest share of decks in
  any snapshot of Standard, Pioneer, Modern, Legacy or Pauper. Cards that
  never made a top 50 are censored below that snapshot's 50th share.
  Written to out/phase4/goldfish_staples.csv (one row per snapshot and card)
  and goldfish_thresholds.csv (the 50th share per snapshot).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

FORMATS = ["standard", "pioneer", "modern", "legacy", "pauper"]
# "all" is the top 50 overall; the other pages are the top 50 of each kind.
CATEGORIES = ["all", "creatures", "spells", "lands"]


def slug(name: str) -> str:
    name = name.split(" // ")[0].lower().replace("'", "").replace(",", "")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def get(url: str, tries: int = 4, timeout: int = 60) -> bytes | None:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mtg-strength-research/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** (i + 1))
        except Exception:
            time.sleep(2 ** (i + 1))
    return None


def edhrec(args) -> None:
    cards = pd.read_parquet(os.path.join(args.out_dir, "phase2", "cards.parquet"))
    names = cards.drop_duplicates("oracle_id")[["oracle_id", "name"]]
    path = os.path.join(args.out_dir, "phase4", "edhrec.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set(pd.read_csv(path)["oracle_id"]) if os.path.exists(path) else set()
    todo = names[~names["oracle_id"].isin(done)]
    print(f"{len(todo)} cards to fetch", flush=True)

    def one(row):
        raw = get(f"https://json.edhrec.com/pages/cards/{slug(row.name)}.json")
        if raw is None:
            return row.oracle_id, row.name, None, None
        c = json.loads(raw).get("container", {}).get("json_dict", {}).get("card", {})
        return row.oracle_id, row.name, c.get("num_decks"), c.get("potential_decks")

    rows = []
    with cf.ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(one, todo.itertuples(index=False))):
            rows.append(r)
            if len(rows) >= 200 or i == len(todo) - 1:
                out = pd.DataFrame(rows, columns=["oracle_id", "name", "num_decks", "potential_decks"])
                out.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                rows = []
                print(f"{i + 1}/{len(todo)}", flush=True)


def parse_staples(html: str) -> list[tuple[str, float]]:
    out = []
    for m in re.finditer(r'data-card-id="([^"]+?)(?: \[[^\]]*\])?".*?<td class=.text-end.>(\d+)%</td>',
                         html, re.S):
        name = re.sub(r"\s*&lt;.*?&gt;", "", m.group(1))
        name = re.sub(r"\s*\[.*$", "", name).replace("&#39;", "'").replace("&amp;", "&").strip()
        out.append((name, float(m.group(2))))
    return out


def goldfish(args) -> None:
    res = os.path.join(args.out_dir, "phase4")
    os.makedirs(res, exist_ok=True)
    rows, thr = [], []
    for fmt, cat in [(f, c) for f in FORMATS for c in CATEGORIES]:
        url = f"mtggoldfish.com/format-staples/{fmt}/full/{cat}"
        cdx = get(f"https://web.archive.org/cdx/search/cdx?url={url}&output=json&filter=statuscode:200"
                  f"&collapse=timestamp:6&from=2021", timeout=180)
        snaps = [r[1] for r in json.loads(cdx)[1:]] if cdx else []
        snaps.append("live")
        print(f"{fmt}/{cat}: {len(snaps)} snapshots", flush=True)
        for ts in snaps:
            src = f"https://www.{url}" if ts == "live" else f"https://web.archive.org/web/{ts}id_/https://www.{url}"
            raw = get(src, timeout=180)
            if raw is None:
                continue
            st = parse_staples(raw.decode("utf-8", "replace"))
            if not st:
                continue
            date = "2026-10-26" if ts == "live" else f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
            rows += [(fmt, cat, date, n, p) for n, p in st]
            thr.append((fmt, cat, date, len(st), min(p for _, p in st)))
            time.sleep(1)
    pd.DataFrame(rows, columns=["format", "category", "date", "name", "pct_decks"]).to_csv(
        os.path.join(res, "goldfish_staples.csv"), index=False)
    pd.DataFrame(thr, columns=["format", "category", "date", "n_listed", "min_pct"]).to_csv(
        os.path.join(res, "goldfish_thresholds.csv"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["edhrec", "goldfish"])
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()
    {"edhrec": edhrec, "goldfish": goldfish}[args.cmd](args)


if __name__ == "__main__":
    main()
