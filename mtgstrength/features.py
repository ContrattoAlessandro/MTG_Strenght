"""Card features for the Phase 1 baselines.

Builds one row per (set, card) from the Phase 0 labels and the Scryfall
oracle data:

- engineered features (mana value, pips, types, P/T, stat efficiency against
  a vanilla curve, keywords, rules-text patterns), plus a few set-level
  context features that stand in for the environment encoder;
- the text that the text models read: rules text with the card's own name
  replaced by CARDNAME, plus the type line and mana cost.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from . import scryfall

RARITIES = ["common", "uncommon", "rare", "mythic"]
TYPES = ["Creature", "Instant", "Sorcery", "Enchantment", "Artifact",
         "Planeswalker", "Land", "Battle"]

# (feature name, regex on CARDNAME-normalised, lower-cased rules text)
TEXT_PATTERNS = [
    ("draw", r"draws? (a|two|three|\w+) cards?"),
    ("destroy_target", r"destroy target"),
    ("exile_target", r"exile target"),
    ("damage", r"deals? (\d+|x) damage"),
    ("damage_any", r"damage to any target"),
    ("fight", r"\bfights?\b"),
    ("minus_counters", r"-\d+/-\d+|-x/-x"),
    ("counter_spell", r"counter target"),
    ("bounce", r"return target .* to its owner's hand"),
    ("tap_down", r"tap target|doesn't untap"),
    ("cant_block", r"can't block"),
    ("plus_counters", r"\+1/\+1 counter"),
    ("pump", r"gets? \+\d+/\+\d+"),
    ("token", r"create[s]? .*token"),
    ("tutor", r"search your library"),
    ("ramp", r"add \{|add one mana|basic land card"),
    ("lifegain", r"gain \d+ life|gains? life"),
    ("drain", r"each opponent loses"),
    ("discard", r"discards?"),
    ("mill", r"\bmill"),
    ("recursion", r"from your graveyard"),
    ("etb", r"when cardname enters"),
    ("dies", r"when cardname dies"),
    ("attacks", r"whenever cardname attacks"),
    ("each_creature", r"(destroy|exile) all|each creature|all creatures"),
    ("opponent_controls", r"opponent controls"),
    ("activated", r"^[^\n]*:"),
    ("x_spell", r"\bx\b"),
    ("modal", r"choose (one|two|three)"),
    ("scry_surveil", r"\bscry\b|\bsurveil\b"),
    ("sacrifice", r"sacrifice"),
    ("cost_reduction", r"costs? \{?\d\}? less"),
    ("extra_cost", r"as an additional cost"),
    ("drawback", r"you lose \d+ life|can't attack|enters tapped"),
]


def load_labels(out_dir: str, sets: list[str]) -> pd.DataFrame:
    frames = []
    for s in sets:
        p = os.path.join(out_dir, s, "card_labels.csv")
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        t["set"] = s
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def attach_oracle(labels: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    """Re-run the name join (with back-face matching) and attach oracle data.

    Drops basic lands, bonus-sheet cards and unmatched names, and keeps one row
    per oracle ID per set (the 17lands name with the most games in hand).
    """
    ids = scryfall.join_oracle_ids(sorted(labels["name"].unique()), oracle)
    ids = ids.rename(columns={"name_17l": "name", "rarity": "scryfall_rarity",
                              "set": "scryfall_set", "oracle_id": "oid",
                              "color_identity": "sf_color_identity"})
    df = labels.drop(columns=["oracle_id", "match"], errors="ignore").merge(ids, on="name", how="left")
    df = df[df["oid"].notna() & (df["rarity"] != "basic") & ~df["bonus_sheet"].astype(bool)]
    df = df[~df["type_line"].fillna("").str.contains("Basic Land")]
    df = (df.sort_values("n_games_in_hand", ascending=False)
            .drop_duplicates(["set", "oid"]).rename(columns={"oid": "oracle_id"}))
    return df.reset_index(drop=True)


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def card_text(row: pd.Series) -> str:
    text = str(row["oracle_text"] or "")
    for n in {row["scryfall_name"], row["front_name"], row["name"],
              *str(row["scryfall_name"]).split(" // ")}:
        if isinstance(n, str) and n:
            text = text.replace(n, "CARDNAME")
            # "Ao, the Dawn Sky" is often called "Ao" in its own text.
            if "," in n:
                text = text.replace(n.split(",")[0], "CARDNAME")
    return text


def engineered(df: pd.DataFrame) -> pd.DataFrame:
    """One row of numeric features per card (index aligned with df)."""
    f = pd.DataFrame(index=df.index)
    f["mana_value"] = df["cmc"].astype(float)
    cost = df["mana_cost"].fillna("").str.split(" // ").str[0]
    for c in "WUBRG":
        f[f"pips_{c}"] = cost.str.count(r"\{" + c + r"\}") + cost.str.count(r"\{" + c + r"/")
    f["pips_total"] = f[[f"pips_{c}" for c in "WUBRG"]].sum(axis=1)
    f["generic"] = cost.str.extract(r"\{(\d+)\}")[0].astype(float).fillna(0)
    f["hybrid_or_phyrexian"] = cost.str.contains("/").astype(int)
    f["x_cost"] = cost.str.contains(r"\{X\}").astype(int)
    ci = df["color_identity"].fillna("")
    f["n_colors"] = ci.map(lambda x: len([c for c in x if c in "WUBRG"]))
    for c in "WUBRG":
        f[f"color_{c}"] = ci.str.contains(c).astype(int)

    tl = df["type_line"].fillna("")
    front_tl = tl.str.split(" // ").str[0]
    for t in TYPES:
        f[f"type_{t.lower()}"] = front_tl.str.contains(t).astype(int)
    f["legendary"] = front_tl.str.contains("Legendary").astype(int)
    f["aura"] = front_tl.str.contains("Aura").astype(int)
    f["equipment"] = front_tl.str.contains("Equipment").astype(int)
    f["vehicle"] = front_tl.str.contains("Vehicle").astype(int)
    f["saga"] = front_tl.str.contains("Saga").astype(int)
    f["two_faced"] = df["layout"].isin(["transform", "modal_dfc", "adventure", "split",
                                         "flip", "prototype"]).astype(int)
    f["layout_mdfc"] = (df["layout"] == "modal_dfc").astype(int)
    f["layout_adventure"] = (df["layout"] == "adventure").astype(int)

    p = df["power"].map(lambda x: _num(str(x).split(" // ")[0]) if x is not None else np.nan)
    t = df["toughness"].map(lambda x: _num(str(x).split(" // ")[0]) if x is not None else np.nan)
    f["power"] = p
    f["toughness"] = t
    f["pt_sum"] = p + t
    f["pt_star"] = df["power"].fillna("").astype(str).str.contains(r"\*").astype(int)
    f["loyalty"] = df["loyalty"].map(lambda x: _num(str(x).split(" // ")[0]) if x is not None else np.nan)

    text = df.apply(card_text, axis=1).str.lower()
    f["text_len"] = text.str.split().str.len().fillna(0)
    f["n_abilities"] = text.str.count("\n") + (text.str.len() > 0)
    for name, pat in TEXT_PATTERNS:
        f[f"txt_{name}"] = text.str.count(pat, flags=re.M)
    dmg = text.str.extractall(r"deals? (\d+) damage")[0].astype(float)
    f["max_damage"] = dmg.groupby(level=0).max().reindex(df.index).fillna(0)

    for i, r in enumerate(RARITIES):
        f[f"rarity_{r}"] = (df["rarity"] == r).astype(int)
    f["rarity_ord"] = df["rarity"].map({r: i for i, r in enumerate(RARITIES)}).astype(float)
    return f


def add_keywords(f: pd.DataFrame, df: pd.DataFrame, vocab: list[str]) -> pd.DataFrame:
    kw = df["keywords"].map(lambda k: set(k) if isinstance(k, (list, np.ndarray)) else set())
    cols = {f"kw_{k.lower().replace(' ', '_')}": kw.map(lambda s: int(k in s)) for k in vocab}
    return pd.concat([f, pd.DataFrame(cols, index=f.index)], axis=1)


def keyword_vocab(df: pd.DataFrame, min_count: int = 15) -> list[str]:
    counts = pd.Series([k for ks in df["keywords"] if isinstance(ks, (list, np.ndarray))
                        for k in ks]).value_counts()
    return sorted(counts[counts >= min_count].index)


def add_curve_and_context(f: pd.DataFrame, df: pd.DataFrame,
                          curve: np.ndarray | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Stat efficiency against a vanilla P+T-by-mana-value line, and
    set-level context features (a crude stand-in for the environment encoder).

    ``curve`` is the (slope, intercept) fitted on training creatures only.
    """
    crea = (f["type_creature"] == 1) & f["pt_sum"].notna() & (f["pt_star"] == 0)
    if curve is None:
        m = crea & df["is_train"]
        curve = np.polyfit(f.loc[m, "mana_value"], f.loc[m, "pt_sum"], 1)
    f = f.copy()
    f["stat_eff"] = np.where(crea, f["pt_sum"] - np.polyval(curve, f["mana_value"]), np.nan)

    g = df["set"]
    tmp = pd.DataFrame({"set": g, "stat_eff": f["stat_eff"], "crea": crea,
                        "removal": (f["txt_destroy_target"] + f["txt_exile_target"]
                                    + f["txt_damage"] + f["txt_minus_counters"]
                                    + f["txt_fight"]).clip(0, 1),
                        "mv": f["mana_value"],
                        "common": df["rarity"] == "common"})
    com = tmp[tmp["common"]]
    ctx = pd.DataFrame({
        "set_stat_eff": com.groupby("set")["stat_eff"].mean(),
        "set_removal_rate": com.groupby("set")["removal"].mean(),
        "set_creature_rate": com.groupby("set")["crea"].mean(),
        "set_mean_mv": com.groupby("set")["mv"].mean(),
    })
    f = pd.concat([f, ctx.reindex(g).set_index(f.index)], axis=1)
    f["stat_eff_vs_set"] = f["stat_eff"] - f["set_stat_eff"]
    return f, curve
