"""Within-set evaluation metrics (design section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ndcg_at_k(y: np.ndarray, pred: np.ndarray, k: int = 20, n_tiebreaks: int = 50,
              seed: int = 0) -> float:
    """NDCG@k with gain = y - min(y), so every card has a non-negative gain
    that is linear in its strength. Ties in ``pred`` (e.g. the rarity-only
    baseline) are broken at random and the result averaged."""
    gain = y - y.min()
    ideal = np.sort(gain)[::-1][:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = float(np.sum(ideal * disc[:len(ideal)]))
    rng = np.random.default_rng(seed)
    has_ties = len(np.unique(pred)) < len(pred)
    vals = []
    for _ in range(n_tiebreaks if has_ties else 1):
        order = np.lexsort((rng.random(len(pred)), -pred))[:k]
        vals.append(float(np.sum(gain[order] * disc[:len(order)])) / idcg)
    return float(np.mean(vals))


def top_k_overlap(y: np.ndarray, pred: np.ndarray, k: int = 20, n_tiebreaks: int = 50,
                  seed: int = 0) -> float:
    """Share of the true top k that the model also ranks in its top k."""
    true_top = set(np.argsort(-y)[:k])
    rng = np.random.default_rng(seed)
    has_ties = len(np.unique(pred)) < len(pred)
    vals = []
    for _ in range(n_tiebreaks if has_ties else 1):
        order = np.lexsort((rng.random(len(pred)), -pred))[:k]
        vals.append(len(true_top & set(order)) / k)
    return float(np.mean(vals))


def set_metrics(df: pd.DataFrame, pred_col: str, k: int = 20) -> dict:
    y = df["theta_vs_repl"].to_numpy()
    p = df[pred_col].to_numpy()
    rho = spearmanr(y, p).statistic if len(np.unique(p)) > 1 else np.nan
    # Ranking inside each rarity, averaged with card-count weights: the
    # signal a model has beyond knowing the rarity (0 for rarity_only).
    within, n = 0.0, 0
    for _, g in df.groupby("rarity"):
        if len(g) >= 10:
            pg = g[pred_col].to_numpy()
            r = spearmanr(g["theta_vs_repl"], pg).statistic if len(np.unique(pg)) > 1 else 0.0
            within += r * len(g)
            n += len(g)
    return {
        "n_cards": len(df),
        "spearman": float(rho),
        "spearman_within_rarity": within / n if n else np.nan,
        f"ndcg@{k}": ndcg_at_k(y, p, k),
        f"top{k}_overlap": top_k_overlap(y, p, k),
        "mae_pp": float(np.mean(np.abs(df["ame_vs_repl_pp"] - df[pred_col + "_pp"]))),
        "mae_logit": float(np.mean(np.abs(y - p))),
    }


def evaluate(df: pd.DataFrame, models: list[str], subset: str) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("set", sort=False):
        for m in models:
            rows.append({"subset": subset, "set": s, "model": m, **set_metrics(g, m)})
    return pd.DataFrame(rows)
