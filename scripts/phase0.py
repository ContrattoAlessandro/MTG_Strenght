"""Phase 0: per-set debiased card labels.

    python scripts/phase0.py --set ECL --data-dir data --out-dir out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mtgstrength import data, labels, scryfall  # noqa: E402

TWO_COLOR = ["WU", "UB", "BR", "RG", "WG", "WB", "UR", "BG", "WR", "UG"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--event-type", default="PremierDraft")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--em-iters", type=int, default=6)
    ap.add_argument("--min-arch-games", type=int, default=20000,
                    help="fit archetype-conditional labels for two-colour decks "
                         "with at least this many games")
    ap.add_argument("--skip-oh", action="store_true")
    ap.add_argument("--max-games", type=int, default=None,
                    help="fit on a random subset of at most this many games")
    args = ap.parse_args()

    raw = os.path.join(args.data_dir, "raw")
    out = os.path.join(args.out_dir, args.set)
    os.makedirs(out, exist_ok=True)

    t = time.time()
    gpath = data.download(data.game_data_url(args.set, args.event_type), raw)
    cpath = data.download(data.cards_url(), raw)
    gd = data.load_game_data(gpath, max_games=args.max_games)
    meta = data.load_card_meta(cpath, args.set, gd.cards)
    data.infer_deck_colors(gd, meta)
    print(f"loaded {len(gd.games):,} games, {len(gd.cards)} cards "
          f"in {time.time() - t:.0f}s", flush=True)

    y = gd.games["won"].to_numpy()
    nuis = labels.nuisance_block(gd.games)
    counts = gd.games["main_colors"].value_counts()
    archs = [a for a in TWO_COLOR if counts.get(a, 0) >= args.min_arch_games]

    summary = {"set": args.set, "event_type": args.event_type,
               "n_games": int(len(gd.games)), "n_cards": len(gd.cards),
               "archetypes": archs, "max_games": args.max_games}

    # Main label: games in hand, with archetype-conditional deviations.
    t = time.time()
    blocks = [nuis] + labels.card_blocks(gd, meta, "in_hand", archetypes=archs)
    res = labels.fit(y, blocks, em_iters=args.em_iters)
    tab = labels.card_table(res, gd, meta, "in_hand")
    arch_tab = labels.archetype_table(res, tab)
    summary["gih"] = {"prior_vars": res.prior_vars, "history": res.history,
                      "replacement_theta": tab.attrs["replacement_theta"],
                      "fit_seconds": round(time.time() - t)}
    nuis_coef = dict(zip(nuis.cols, res.beta[res.slices["nuisance"]].round(4)))
    summary["gih"]["nuisance"] = {k: float(v) for k, v in nuis_coef.items()}

    # Robustness label: opening hand only (no long-game survivorship).
    if not args.skip_oh:
        t = time.time()
        blocks_oh = [nuis] + labels.card_blocks(gd, meta, "opening_hand")
        res_oh = labels.fit(y, blocks_oh, em_iters=args.em_iters)
        tab_oh = labels.card_table(res_oh, gd, meta, "opening_hand")
        tab_oh = tab_oh.rename(columns={"theta_hand": "theta_oh", "theta_hand_sd": "theta_oh_sd",
                                        "theta_vs_repl": "theta_oh_vs_repl"})
        tab = tab.merge(
            tab_oh[["name", "theta_oh", "theta_oh_sd", "theta_oh_vs_repl",
                    "n_games_opening_hand", "raw_wr_opening_hand"]],
            on="name", how="left")
        summary["oh"] = {"prior_vars": res_oh.prior_vars, "history": res_oh.history,
                         "fit_seconds": round(time.time() - t)}

    # Oracle IDs from Scryfall, when the host is reachable.
    try:
        opath = scryfall.download_oracle_cards(raw)
        ids = scryfall.join_oracle_ids(gd.cards, scryfall.load_oracle_cards(opath))
        tab = tab.merge(ids[["name_17l", "oracle_id", "match"]]
                        .rename(columns={"name_17l": "name"}), on="name", how="left")
        ids.to_parquet(os.path.join(out, "cards_scryfall.parquet"))
        summary["scryfall_unmatched"] = ids.loc[ids["match"].isna(), "name_17l"].tolist()
    except Exception as e:  # network policy, outage, etc.
        print(f"scryfall join skipped: {e}", flush=True)
        tab["oracle_id"] = None
        summary["scryfall_unmatched"] = f"skipped: {e}"

    tab.to_csv(os.path.join(out, "card_labels.csv"), index=False)
    arch_tab.to_csv(os.path.join(out, "card_archetype_labels.csv"), index=False)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
