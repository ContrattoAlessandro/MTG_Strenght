"""Commander data for every Commander-legal card (step I of docs/COMMANDER_PLAN.md).

    python scripts/cmd_data.py cards  --default-cards <default-cards.jsonl.gz>
    python scripts/cmd_data.py edhrec [--threads 4]

- cards: one row per oracle_id with the card's *first* paper printing
  (release date, set code, set type), from Scryfall's "Default Cards" bulk
  file (every printing). The oracle bulk file only has one, usually recent,
  printing per card, which is useless for a temporal split. Also keeps the
  Scryfall `game_changer` flag (WotC's official list of the strongest
  Commander cards), used only for external validation, never as a feature.
  -> out/commander/first_print.parquet
- edhrec: EDHREC card JSON for every Commander-legal card in the oracle bulk
  file: num_decks, potential_decks, salt and the precon field. Resumable.
  -> out/commander/edhrec_all.csv (+ fetched_at column)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from phase4_data import get, slug  # noqa: E402

OUT = os.path.join(ROOT, "out", "commander")
ORACLE = os.path.join(ROOT, "data", "raw", "scryfall_oracle_cards.jsonl.gz")


def cards(args) -> None:
    first: dict[str, dict] = {}
    with gzip.open(args.default_cards, "rt", encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            o = c.get("oracle_id") or (c.get("card_faces") or [{}])[0].get("oracle_id")
            if not o or c.get("digital") or "paper" not in c.get("games", []):
                continue
            # Promo printings (prerelease stamps, buy-a-box) come out days before
            # their set: a card's first printing is its first non-promo one.
            key = (c["set_type"] in ("memorabilia", "token", "promo") or c.get("promo", False),
                   c["released_at"])
            r = first.get(o)
            if r is None or key < (r["_memo"], r["first_release"]):
                first[o] = {"oracle_id": o, "first_release": c["released_at"], "_memo": key[0],
                            "first_set": c["set"], "first_set_type": c["set_type"]}
    df = pd.DataFrame(first.values()).drop(columns="_memo")
    gc = {}
    with gzip.open(ORACLE, "rt", encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            gc[c["oracle_id"]] = bool(c.get("game_changer"))
    df["game_changer"] = df["oracle_id"].map(gc)
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(os.path.join(OUT, "first_print.parquet"), index=False)
    print(f"{len(df)} oracle ids; {df['game_changer'].sum()} game changers")


def edhrec(args) -> None:
    rows = []
    with gzip.open(ORACLE, "rt", encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            if c.get("legalities", {}).get("commander") == "legal" and c.get("layout") not in ("token", "emblem"):
                rows.append((c["oracle_id"], c["name"]))
    names = pd.DataFrame(rows, columns=["oracle_id", "name"]).drop_duplicates("oracle_id")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "edhrec_all.csv")
    done = set(pd.read_csv(path)["oracle_id"]) if os.path.exists(path) else set()
    todo = names[~names["oracle_id"].isin(done)]
    print(f"{len(todo)} of {len(names)} cards to fetch", flush=True)
    cols = ["oracle_id", "name", "num_decks", "potential_decks", "salt", "precon", "fetched_at"]

    def one(row):
        raw = get(f"https://json.edhrec.com/pages/cards/{slug(row.name)}.json", tries=3, timeout=30)
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        if raw is None:
            return row.oracle_id, row.name, None, None, None, None, now
        try:
            c = json.loads(raw).get("container", {}).get("json_dict", {}).get("card", {})
        except ValueError:
            c = {}
        return (row.oracle_id, row.name, c.get("num_decks"), c.get("potential_decks"),
                c.get("salt"), c.get("precon"), now)

    buf = []
    with cf.ThreadPoolExecutor(args.threads) as ex:
        for i, r in enumerate(ex.map(one, todo.itertuples(index=False))):
            buf.append(r)
            if len(buf) >= 500 or i == len(todo) - 1:
                pd.DataFrame(buf, columns=cols).to_csv(path, mode="a", header=not os.path.exists(path),
                                                       index=False)
                buf = []
                print(f"{i + 1}/{len(todo)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cards")
    p.add_argument("--default-cards", required=True)
    p = sub.add_parser("edhrec")
    p.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    {"cards": cards, "edhrec": edhrec}[args.cmd](args)


if __name__ == "__main__":
    main()
