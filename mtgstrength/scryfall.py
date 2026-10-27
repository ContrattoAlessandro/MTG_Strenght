"""Scryfall oracle data and the name -> oracle_id join.

17lands identifies cards by name. Double-faced and adventure cards appear
under their front-face name, split cards under "A // B". We index Scryfall's
oracle cards by full name and by front-face name, preferring full names.
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.request

import pandas as pd

BULK_INDEX = "https://api.scryfall.com/bulk-data"
HEADERS = {"User-Agent": "mtg-strength/0.1", "Accept": "application/json"}

KEEP = ["oracle_id", "name", "mana_cost", "cmc", "type_line", "oracle_text",
        "power", "toughness", "loyalty", "colors", "color_identity",
        "keywords", "layout", "rarity", "set", "released_at"]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as r:
        return r.read()


def download_oracle_cards(dest_dir: str, force: bool = False) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    # Scryfall's bulk index now offers only gzipped JSON Lines
    # (jsonl_download_uri); older indexes had a plain JSON download_uri.
    path = os.path.join(dest_dir, "scryfall_oracle_cards.jsonl.gz")
    if force or not os.path.exists(path):
        index = json.loads(_get(BULK_INDEX))
        entry = next(d for d in index["data"] if d["type"] == "oracle_cards")
        uri = entry.get("jsonl_download_uri") or entry["download_uri"]
        if not uri.endswith(".jsonl.gz"):
            path = os.path.join(dest_dir, "scryfall_oracle_cards.json")
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(_get(uri))
        os.replace(tmp, path)
    return path


def load_oracle_cards(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path) as f:
            raw = json.load(f)
    rows = []
    for c in raw:
        faces = c.get("card_faces") or []
        row = {k: c.get(k) for k in KEEP}
        if faces:
            # Face-level fields are missing at the top level for DFCs.
            for k in ("mana_cost", "oracle_text", "power", "toughness", "loyalty"):
                if row[k] is None:
                    row[k] = " // ".join(str(f.get(k, "")) for f in faces)
        row["front_name"] = faces[0]["name"] if faces else c["name"]
        row["face_names"] = [f["name"] for f in faces[1:]]
        rows.append(row)
    return pd.DataFrame(rows)


# Non-game objects that share names with real cards.
SKIP_LAYOUTS = {"art_series", "token", "double_faced_token", "emblem", "vanguard",
                "scheme", "planar"}


def join_oracle_ids(names: list[str], oracle: pd.DataFrame) -> pd.DataFrame:
    oracle = oracle[~oracle["layout"].isin(SKIP_LAYOUTS)].rename(columns={"name": "scryfall_name"})
    oracle["name"] = oracle["scryfall_name"]
    by_full = oracle.drop_duplicates("name").set_index("name")
    by_front = oracle.drop_duplicates("front_name").set_index("front_name")
    # Some older 17lands exports list the back face of an MDFC as its own card.
    by_back = (oracle.explode("face_names").dropna(subset=["face_names"])
               .drop_duplicates("face_names").set_index("face_names"))
    out = []
    for n in names:
        if n in by_full.index:
            out.append({**by_full.loc[n].drop("face_names").to_dict(), "name": n, "match": "full"})
        elif n in by_front.index:
            out.append({**by_front.loc[n].drop(["name", "face_names"]).to_dict(), "name": n, "match": "front"})
        elif n in by_back.index:
            out.append({**by_back.loc[n].drop(["name", "front_name"]).to_dict(), "name": n, "match": "back"})
        else:
            out.append({"name": n, "match": None})
    return pd.DataFrame(out).rename(columns={"name": "name_17l"})
