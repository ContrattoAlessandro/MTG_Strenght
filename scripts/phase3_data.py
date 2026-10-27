"""Phase 3 data: decklists and outcomes from the 17lands game data.

    python scripts/phase3_data.py --raw-dir <cache> --work-dir <work dir> --max-games 200000

For each set, keeps a random subset of games and writes <work-dir>/<SET>.npz:

- ``tok``: (n_games, K) row indices into out/phase2/cards.parquet for the
  non-basic maindeck cards (-1 = padding). Cards without a Phase 2 row
  (bonus-sheet cards, basics) are not tokens.
- ``cnt``: copies in the maindeck; ``seen``: 1 if a copy was in hand at some
  point (opening hand, drawn or tutored).
- per-game covariates: won, on_play, skill (logit of the player's 17lands
  win-rate bucket), log games-played bucket, mulligans, number of basics and
  of maindeck cards that are not tokens, main colours.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mtgstrength import data  # noqa: E402
from phase1 import SETS  # noqa: E402

BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
K = 40


def logit(p, eps=0.02):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def build(set_code: str, cards: pd.DataFrame, args) -> None:
    out_path = os.path.join(args.work_dir, f"{set_code}.npz")
    if os.path.exists(out_path):
        return
    path = data.download(data.game_data_url(set_code), args.raw_dir)
    gd = data.load_game_data(path, max_games=args.max_games, seed=0)
    rows = cards[cards["set"] == set_code]
    by_name = dict(zip(rows["name"], rows.index))
    col_row = np.array([by_name.get(n, -1) for n in gd.cards])
    is_basic = np.array([n in BASICS for n in gd.cards])
    deck = gd.deck.astype(np.int16)
    seen = ((gd.opening_hand + gd.drawn + gd.tutored) > 0)
    tokcols = np.flatnonzero(col_row >= 0)
    n = len(deck)
    tok = np.full((n, K), -1, np.int32)
    cnt = np.zeros((n, K), np.int8)
    sn = np.zeros((n, K), np.int8)
    D = deck[:, tokcols]
    for i in range(n):
        nz = np.flatnonzero(D[i])[:K]
        tok[i, :len(nz)] = col_row[tokcols[nz]]
        cnt[i, :len(nz)] = D[i, nz]
        sn[i, :len(nz)] = seen[i, tokcols[nz]]
    g = gd.games
    if g["main_colors"].isna().all():
        cpath = os.path.join(args.raw_dir, "cards.csv")   # plain CSV, not gzipped
        if not os.path.exists(cpath):
            import urllib.request
            urllib.request.urlretrieve(data.cards_url(), cpath)
        meta = data.load_card_meta(cpath, set_code, gd.cards)
        data.infer_deck_colors(gd, meta)
        g = gd.games
    skill = pd.to_numeric(g["user_game_win_rate_bucket"], errors="coerce").fillna(0.55).to_numpy()
    ngames = pd.to_numeric(g["user_n_games_bucket"], errors="coerce").fillna(10).to_numpy()
    np.savez_compressed(
        out_path, tok=tok, cnt=cnt, seen=sn,
        won=g["won"].astype(float).to_numpy().astype(np.int8),
        on_play=g["on_play"].astype(float).fillna(0.5).to_numpy().astype(np.float32),
        skill=logit(skill).astype(np.float32),
        log_ngames=np.log1p(ngames).astype(np.float32),
        mull=pd.to_numeric(g["num_mulligans"], errors="coerce").fillna(0).to_numpy().astype(np.float32),
        opp_mull=pd.to_numeric(g["opp_num_mulligans"], errors="coerce").fillna(0).to_numpy().astype(np.float32),
        n_basics=deck[:, is_basic].sum(1).astype(np.int16),
        n_other=deck[:, (col_row < 0) & ~is_basic].sum(1).astype(np.int16),
        main_colors=g["main_colors"].fillna("").astype(str).to_numpy(),
    )
    print(f"{set_code}: {n} games, {len(tokcols)} token cards, "
          f"{(tok >= 0).sum(1).mean():.1f} tokens/deck", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--max-games", type=int, default=200_000)
    ap.add_argument("--sets", nargs="*", default=SETS)
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    cards = pd.read_parquet(os.path.join(args.out_dir, "phase2", "cards.parquet"))
    for s in args.sets:
        build(s, cards, args)


if __name__ == "__main__":
    main()
