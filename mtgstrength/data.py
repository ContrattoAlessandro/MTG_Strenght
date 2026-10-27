"""Download and load the public 17lands datasets.

17lands publishes one CSV per (set, event type) with a row per game. Card
columns are wide: ``opening_hand_<name>``, ``drawn_<name>``, ``tutored_<name>``,
``deck_<name>`` and ``sideboard_<name>`` hold copy counts for every card that
can appear in the set (including bonus-sheet cards).
"""

from __future__ import annotations

import gzip
import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

S3 = "https://17lands-public.s3.amazonaws.com/analysis_data"
CARD_PREFIXES = ("opening_hand_", "drawn_", "tutored_", "deck_", "sideboard_")

# Per-game covariates we keep. All are known before the game starts, apart
# from user_game_win_rate_bucket, which is computed over the player's games on
# 17lands and so includes this game (a small leak for players with many games).
GAME_COLS = [
    "draft_id", "draft_time", "build_index", "match_number", "game_number",
    "rank", "main_colors", "splash_colors", "on_play", "num_mulligans",
    "opp_num_mulligans", "opp_colors", "num_turns", "won",
    "user_n_games_bucket", "user_game_win_rate_bucket",
]


def game_data_url(set_code: str, event_type: str = "PremierDraft") -> str:
    return f"{S3}/game_data/game_data_public.{set_code}.{event_type}.csv.gz"


def cards_url() -> str:
    return f"{S3}/cards/cards.csv"


def download(url: str, dest_dir: str, force: bool = False) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, url.rsplit("/", 1)[-1])
    if force or not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
        if path.endswith(".gz"):
            _untar_if_needed(path)
    return path


def _untar_if_needed(path: str) -> None:
    """Some re-exported files (KHM to VOW) are a gzipped tar holding one CSV.
    Rewrite those in place as a plain gzipped CSV."""
    with gzip.open(path, "rb") as f:
        head = f.read(512)
    if head[257:262] != b"ustar":
        return
    tmp = path + ".plain"
    with tarfile.open(path, "r:gz") as tf, gzip.open(tmp, "wb", compresslevel=3) as out:
        member = next(m for m in tf if m.isfile())
        shutil.copyfileobj(tf.extractfile(member), out, 1 << 24)
    os.replace(tmp, path)


@dataclass
class GameData:
    games: pd.DataFrame          # one row per game, GAME_COLS
    cards: list[str]             # card names, in column order
    deck: np.ndarray             # (n_games, n_cards) int8 copies in maindeck
    opening_hand: np.ndarray     # (n_games, n_cards) int8
    drawn: np.ndarray            # (n_games, n_cards) int8, drawn after opening hand
    tutored: np.ndarray          # (n_games, n_cards) int8

    @property
    def in_hand(self) -> np.ndarray:
        """Games-in-hand indicator, as 17lands defines GIH: opening hand or drawn."""
        return ((self.opening_hand + self.drawn) > 0).astype(np.int8)


def _count_rows(path: str) -> int:
    n = 0
    with gzip.open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
    return n - 1


def load_game_data(path: str, max_games: int | None = None,
                   seed: int = 0, row_filter=None, extra_cols: tuple[str, ...] = ()) -> GameData:
    """Load one game-data file, streaming it in batches so that peak memory
    stays near the size of the int8 arrays. ``max_games`` keeps a random
    subset of about that many games (Bernoulli sampling per game).
    ``row_filter(batch) -> bool array`` keeps only some games (e.g. a date
    window); ``extra_cols`` are kept in ``games`` too (e.g. game_time)."""
    header = pd.read_csv(path, nrows=0).columns
    cards = [c[len("deck_"):] for c in header if c.startswith("deck_")]
    card_cols = [c for c in header if c.startswith(CARD_PREFIXES[:4])]
    game_cols = [c for c in GAME_COLS + list(extra_cols) if c in header]
    if "user_win_rate_bucket" in header:
        game_cols.append("user_win_rate_bucket")
    keep_p = 1.0
    if max_games is not None:
        keep_p = min(1.0, max_games / max(_count_rows(path), 1))
    rng = np.random.default_rng(seed)

    reader = pacsv.open_csv(
        path, read_options=pacsv.ReadOptions(block_size=1 << 26),
        convert_options=pacsv.ConvertOptions(
            include_columns=game_cols + card_cols,
            # Some files write counts as "0.0", so parse as float and cast.
            column_types={c: pa.float32() for c in card_cols}))
    frames, mats = [], {p: [] for p in CARD_PREFIXES[:4]}
    for batch in reader:
        if row_filter is not None:
            batch = batch.filter(pa.array(np.asarray(row_filter(batch), bool)))
        if keep_p < 1.0:
            batch = batch.filter(pa.array(rng.random(batch.num_rows) < keep_p))
        frames.append(batch.select(game_cols).to_pandas())
        for prefix in mats:
            cols = [prefix + c for c in cards]
            if cols[0] in header:
                mats[prefix].append(np.column_stack(
                    [batch.column(c).to_numpy(zero_copy_only=False) for c in cols]
                ).astype(np.int8))
    df = pd.concat(frames, ignore_index=True)
    # Older exports name the skill bucket differently and write ranks as
    # "Gold-2-0-0-0"; main_colors/splash_colors are filled later from the
    # decklists (see infer_deck_colors).
    if "user_win_rate_bucket" in header and "user_game_win_rate_bucket" not in header:
        df["user_game_win_rate_bucket"] = df["user_win_rate_bucket"]
    if "rank" in df:
        df["rank"] = df["rank"].astype("string").str.split("-").str[0].str.lower()
    for c in GAME_COLS:
        if c not in df:
            df[c] = np.nan

    def block(prefix: str) -> np.ndarray:
        if not mats[prefix]:   # older files have no tutored_ columns
            return np.zeros((len(df), len(cards)), np.int8)
        return np.concatenate(mats[prefix])

    return GameData(
        games=df[GAME_COLS + [c for c in extra_cols if c in df]].reset_index(drop=True),
        # Older files (2021-22) replaced the comma in names with a space,
        # e.g. "Ao  the Dawn Sky" for "Ao, the Dawn Sky".
        cards=[c.replace("  ", ", ") for c in cards],
        deck=block("deck_"),
        opening_hand=block("opening_hand_"),
        drawn=block("drawn_"),
        tutored=block("tutored_"),
    )


def infer_deck_colors(gd: GameData, meta: pd.DataFrame,
                      main_min: int = 4) -> None:
    """Fill main_colors and splash_colors where the file lacks them.

    A colour is main if the deck has at least ``main_min`` non-land cards of
    that colour (gold cards count for each of their colours), and splash if it
    has 1 to main_min - 1.
    """
    g = gd.games
    if g["main_colors"].notna().any():
        return
    nonland = ~meta["types"].fillna("").str.contains("Land").to_numpy()
    ci = meta["color_identity"].fillna("").to_numpy()
    counts = {}
    for col in "WUBRG":
        has = np.array([col in x for x in ci]) & nonland
        counts[col] = gd.deck[:, has].sum(axis=1)
    main = np.full(len(g), "", dtype=object)
    splash = np.full(len(g), "", dtype=object)
    for col in "WUBRG":
        n = counts[col]
        main = np.where(n >= main_min, main + col, main)
        splash = np.where((n > 0) & (n < main_min), splash + col, splash)
    g["main_colors"] = main
    g["splash_colors"] = splash


def load_card_meta(path: str, set_code: str, names: list[str]) -> pd.DataFrame:
    """Rarity, colour identity, mana value and types for each card name.

    Uses the card's printing in ``set_code`` when there is one, otherwise its
    most recent printing (bonus-sheet cards such as SPG are printed elsewhere).
    """
    meta = pd.read_csv(path)
    meta["in_set"] = meta["expansion"] == set_code
    meta = (meta[meta["name"].isin(names)]
            .sort_values(["in_set", "id"], ascending=False)
            .drop_duplicates("name"))
    out = pd.DataFrame({"name": names}).merge(meta, on="name", how="left")
    out["color_identity"] = out["color_identity"].fillna("")
    out["color_group"] = out["color_identity"].map(_color_group)
    out["bonus_sheet"] = ~out["in_set"].fillna(False).astype(bool)
    return out.drop(columns=["in_set"])


def _color_group(ci: str) -> str:
    if not isinstance(ci, str) or ci == "":
        return "C"
    return ci if len(ci) == 1 else "M"
