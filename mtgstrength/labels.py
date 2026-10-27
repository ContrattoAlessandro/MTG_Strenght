"""Debiased per-card strength labels from 17lands game data.

Model, per set (one row per game g, player-side outcome)::

    logit P(won_g) = b0 + nuisance_g
                     + sum_c theta_c  * in_hand_{c,g}
                     + sum_c theta'_c * deck_copies_{c,g}
                     [+ sum_{c,a} eta_{c,a} * in_hand_{c,g} * [archetype_g == a]]

``nuisance_g`` holds the confounders: player skill (17lands win-rate bucket,
with a slope per games-played bucket so that low-sample buckets count for
less), rank, own archetype (main colours, splash), opponent colours, play or
draw, and mulligans.

Partial pooling: theta_c = mu_{g(c)} + delta_c, where g(c) is the card's
rarity x colour group, mu has a weak prior, and delta_c ~ N(0, tau^2). The
same holds for theta' and, around zero, for eta. The tau^2 values are set by
empirical Bayes (EM on the Laplace approximation). Posterior variances come
from the inverse Hessian at the MAP, so Var(theta_c) includes the uncertainty
in its group mean.

theta_c is the "drawn" effect: the change in log-odds from having c in hand,
given the deck it is in. theta'_c soaks up the deck-selection signal (which
decks and drafters pick c). eta_{c,a} is the archetype-conditional deviation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp

from .data import GameData

WEAK_PRIOR_VAR = 10.0  # for intercept, nuisance and group means (logit scale)


@dataclass
class Block:
    name: str
    cols: list[str]
    X: sp.csr_matrix
    prior_var: float
    learn_var: bool = False   # estimate prior_var by empirical Bayes


@dataclass
class FitResult:
    beta: np.ndarray
    cov: np.ndarray
    blocks: list[Block]
    slices: dict[str, slice]
    prior_vars: dict[str, float]
    eta: np.ndarray            # linear predictor per game
    history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- design

def _one_hot(values: pd.Series, prefix: str, keep: list | None = None,
             drop_first: bool = False) -> tuple[list[str], sp.csr_matrix]:
    v = values.astype(str)
    if keep is not None:
        v = v.where(v.isin([str(k) for k in keep]), "other")
    cats = sorted(v.unique())
    if drop_first:
        cats = cats[1:]
    idx = {c: i for i, c in enumerate(cats)}
    rows, cols = [], []
    for r, x in enumerate(v):
        j = idx.get(x)
        if j is not None:
            rows.append(r)
            cols.append(j)
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(len(v), len(cats)))
    return [f"{prefix}={c}" for c in cats], X


def nuisance_block(games: pd.DataFrame, min_arch_games: int = 2000) -> Block:
    parts: list[tuple[list[str], sp.csr_matrix]] = []
    n = len(games)

    # Skill: logit of the player's 17lands win-rate bucket, slope per
    # games-played bucket, plus a level per games-played bucket.
    wr = games["user_game_win_rate_bucket"].fillna(0.5).clip(0.3, 0.75)
    skill = np.log(wr / (1 - wr)).to_numpy()
    names, nb = _one_hot(games["user_n_games_bucket"], "n_games")
    parts.append((names, nb))
    parts.append(([f"skill_x_{c}" for c in names],
                  sp.csr_matrix(nb.multiply(skill[:, None]))))

    parts.append(_one_hot(games["rank"].fillna("none"), "rank", drop_first=True))

    arch_counts = games["main_colors"].value_counts()
    common_arch = arch_counts[arch_counts >= min_arch_games].index.tolist()
    parts.append(_one_hot(games["main_colors"].fillna("none"), "arch",
                          keep=common_arch, drop_first=True))
    splash = games["splash_colors"].fillna("").astype(str).str.len().clip(0, 2)
    parts.append(_one_hot(splash, "n_splash", drop_first=True))

    opp_counts = games["opp_colors"].value_counts()
    common_opp = opp_counts[opp_counts >= min_arch_games].index.tolist()
    parts.append(_one_hot(games["opp_colors"].fillna("none"), "opp",
                          keep=common_opp, drop_first=True))

    parts.append((["on_play"], sp.csr_matrix(
        games["on_play"].astype(float).to_numpy()[:, None])))
    parts.append(_one_hot(games["num_mulligans"].clip(0, 2), "mull",
                          drop_first=True))
    parts.append(_one_hot(games["opp_num_mulligans"].clip(0, 2), "opp_mull",
                          drop_first=True))

    parts.insert(0, (["intercept"], sp.csr_matrix(np.ones((n, 1)))))
    cols = [c for names, _ in parts for c in names]
    X = sp.hstack([m for _, m in parts], format="csr")
    return Block("nuisance", cols, X, WEAK_PRIOR_VAR)


def card_blocks(gd: GameData, meta: pd.DataFrame, use: str = "in_hand",
                archetypes: list[str] | None = None,
                min_arch_card_games: int = 500) -> list[Block]:
    """Group-mean and card-deviation blocks for theta and theta'.

    ``use`` is "in_hand" (the GIH label) or "opening_hand" (the OH label,
    which is free of the survivorship effect of cards drawn late in long
    games).
    """
    # Convert to sparse before casting to float: a dense float64 copy of an
    # 800k x 400 game matrix is 2.5 GB.
    if use == "in_hand":
        H = sp.csr_matrix(gd.in_hand).astype(np.float64)
    elif use == "opening_hand":
        H = sp.csr_matrix((gd.opening_hand > 0).astype(np.int8)).astype(np.float64)
    else:
        raise ValueError(use)
    D = sp.csr_matrix(gd.deck).astype(np.float64)

    groups = (meta["rarity"].fillna("unknown") + ":" + meta["color_group"]).to_numpy()
    group_names = sorted(set(groups))
    G = sp.csr_matrix((np.ones(len(groups)),
                       (np.arange(len(groups)), [group_names.index(g) for g in groups])),
                      shape=(len(groups), len(group_names)))

    blocks = [
        Block("mu_hand", group_names, sp.csr_matrix(H @ G), WEAK_PRIOR_VAR),
        Block("delta_hand", gd.cards, H, 0.01, learn_var=True),
        Block("mu_deck", group_names, sp.csr_matrix(D @ G), WEAK_PRIOR_VAR),
        Block("delta_deck", gd.cards, D, 0.001, learn_var=True),
    ]

    if archetypes:
        arch = gd.games["main_colors"].astype(str).to_numpy()
        cols, mats = [], []
        Hc = H.tocsc()
        is_basic = (meta["rarity"] == "basic").to_numpy()
        for a in archetypes:
            mask = sp.diags((arch == a).astype(np.float64))
            Ha = (mask @ Hc).tocsc()
            counts = np.asarray(Ha.sum(axis=0)).ravel()
            # Basic lands are excluded: within an archetype nearly every game
            # has them in hand, so they would duplicate the archetype effect.
            keep = np.flatnonzero((counts >= min_arch_card_games) & ~is_basic)
            cols += [f"{gd.cards[j]}|{a}" for j in keep]
            mats.append(Ha[:, keep])
        blocks.append(Block("eta_arch", cols, sp.hstack(mats, format="csr"),
                            0.01, learn_var=True))
    return blocks


# ---------------------------------------------------------------- fitting

def _layout(blocks: list[Block]) -> dict[str, slice]:
    out, start = {}, 0
    for b in blocks:
        out[b.name] = slice(start, start + len(b.cols))
        start += len(b.cols)
    return out


def fit(y: np.ndarray, blocks: list[Block], em_iters: int = 6,
        newton_tol: float = 1e-6, max_newton: int = 25,
        verbose: bool = True) -> FitResult:
    X = sp.hstack([b.X for b in blocks], format="csr")
    Xt = X.T.tocsr()
    slices = _layout(blocks)
    prior_vars = {b.name: b.prior_var for b in blocks}
    beta = np.zeros(X.shape[1])
    y = y.astype(np.float64)
    history = []

    def penalty_diag() -> np.ndarray:
        lam = np.empty(X.shape[1])
        for b in blocks:
            lam[slices[b.name]] = 1.0 / prior_vars[b.name]
        return lam

    for em in range(em_iters + 1):
        lam = penalty_diag()
        for it in range(max_newton):
            eta = X @ beta
            p = 1.0 / (1.0 + np.exp(-eta))
            grad = Xt @ (y - p) - lam * beta
            w = p * (1 - p)
            Hm = (Xt @ sp.diags(w) @ X).toarray()
            Hm[np.diag_indices_from(Hm)] += lam
            cf = sla.cho_factor(Hm, lower=False, check_finite=False)
            step = sla.cho_solve(cf, grad, check_finite=False)
            beta += step
            if np.max(np.abs(step)) < newton_tol:
                break
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        loglik = float(np.sum(y * eta - np.logaddexp(0, eta)))
        cov = sla.cho_solve(cf, np.eye(len(beta)), check_finite=False)
        rec = {"em": em, "newton_iters": it + 1, "loglik": loglik,
               **{f"tau2_{k}": v for k, v in prior_vars.items()
                  if any(b.learn_var and b.name == k for b in blocks)}}
        history.append(rec)
        if verbose:
            print(rec, flush=True)
        if em == em_iters:
            break
        # EM update of the learned prior variances (Laplace approximation).
        for b in blocks:
            if b.learn_var:
                s = slices[b.name]
                k = s.stop - s.start
                v = (np.sum(beta[s] ** 2) + np.trace(cov[s, s])) / k
                prior_vars[b.name] = max(v, 1e-6)

    return FitResult(beta, cov, blocks, slices, prior_vars, eta, history)


# ---------------------------------------------------------------- readout

def card_table(res: FitResult, gd: GameData, meta: pd.DataFrame,
               use: str = "in_hand", repl_min_games: int = 2000) -> pd.DataFrame:
    """Per-card theta (logit), its posterior SD, and win-rate-scale readouts."""
    groups = (meta["rarity"].fillna("unknown") + ":" + meta["color_group"]).to_numpy()
    out = meta[["name", "rarity", "color_identity", "mana_value", "types",
                "bonus_sheet"]].copy()

    for kind, mu_name, d_name in (("hand", "mu_hand", "delta_hand"),
                                  ("deck", "mu_deck", "delta_deck")):
        mu_s, d_s = res.slices[mu_name], res.slices[d_name]
        gnames = res.blocks[[b.name for b in res.blocks].index(mu_name)].cols
        gi = np.array([gnames.index(g) for g in groups]) + mu_s.start
        di = np.arange(d_s.start, d_s.stop)
        theta = res.beta[gi] + res.beta[di]
        var = (res.cov[gi, gi] + res.cov[di, di] + 2 * res.cov[gi, di])
        out[f"theta_{kind}"] = theta
        out[f"theta_{kind}_sd"] = np.sqrt(var)

    H = gd.in_hand if use == "in_hand" else (gd.opening_hand > 0).astype(np.int8)
    won = gd.games["won"].to_numpy().astype(bool)
    n_hand = H.sum(axis=0)
    out["n_games_deck"] = (gd.deck > 0).sum(axis=0)
    out[f"n_games_{use}"] = n_hand
    with np.errstate(invalid="ignore", divide="ignore"):
        out[f"raw_wr_{use}"] = (H.astype(bool) & won[:, None]).sum(axis=0) / n_hand
        not_hand = (gd.deck > 0) & ~H.astype(bool)
        out["raw_wr_not_drawn"] = (not_hand & won[:, None]).sum(axis=0) / not_hand.sum(axis=0)
    out["raw_iwd"] = out[f"raw_wr_{use}"] - out["raw_wr_not_drawn"]

    # Replacement level: the median theta among regularly played commons.
    pool = out[(out["rarity"] == "common") & (out[f"n_games_{use}"] >= repl_min_games)
               & ~out["bonus_sheet"]]
    repl = float(pool["theta_hand"].median()) if len(pool) else 0.0
    out["theta_vs_repl"] = out["theta_hand"] - repl

    # Average marginal effect, in win-rate points, of having c in hand instead
    # of a replacement-level card, over the games where c was in hand.
    p_with = 1 / (1 + np.exp(-res.eta))
    ame = np.full(len(gd.cards), np.nan)
    for j in range(len(gd.cards)):
        rows = np.flatnonzero(H[:, j])
        if len(rows):
            p_repl = 1 / (1 + np.exp(-(res.eta[rows] - out["theta_vs_repl"].iat[j])))
            ame[j] = np.mean(p_with[rows] - p_repl)
    out["ame_vs_repl_pp"] = 100 * ame
    out["precision_weight"] = 1.0 / out["theta_hand_sd"] ** 2
    out.attrs["replacement_theta"] = repl
    return out.sort_values("theta_vs_repl", ascending=False).reset_index(drop=True)


def archetype_table(res: FitResult, card_tab: pd.DataFrame) -> pd.DataFrame:
    if "eta_arch" not in res.slices:
        return pd.DataFrame()
    s = res.slices["eta_arch"]
    b = res.blocks[[x.name for x in res.blocks].index("eta_arch")]
    theta = card_tab.set_index("name")
    rows = []
    for k, col in enumerate(b.cols):
        name, arch = col.rsplit("|", 1)
        i = s.start + k
        rows.append({"name": name, "archetype": arch,
                     "eta": res.beta[i], "eta_sd": np.sqrt(res.cov[i, i]),
                     "theta_arch": theta.at[name, "theta_hand"] + res.beta[i],
                     "n_games_in_hand_arch": int(b.X[:, k].sum())})
    return pd.DataFrame(rows)
