"""Feature building and prediction shared by app/build.py and app/server.py.

A card is a dict (or DataFrame row) with Scryfall-style fields: name,
mana_cost, type_line, oracle_text, power, toughness, loyalty, rarity, and
optionally cmc, color_identity, keywords, layout. Cards are always scored
in an environment: one of the labelled sets, or "typical" (the average of
the most recent sets), which fixes the set-context features.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from mtgstrength import features  # noqa: E402

CTX_BASE = ["set_stat_eff", "set_removal_rate", "set_creature_rate", "set_mean_mv"]
TYPICAL = "typical"
PP_PER_LOGIT = 22.14   # win-rate points per logit of theta (out/phase1/run_info.json)
Z80 = 1.2816

GRADES = [(99, "A+"), (95, "A"), (90, "A-"), (80, "B+"), (70, "B"), (60, "B-"), (50, "C+"),
          (40, "C"), (30, "C-"), (20, "D+"), (10, "D"), (4, "D-"), (0, "F")]


def grade(pct: float) -> str | None:
    if pct is None or not np.isfinite(pct):
        return None
    return next(g for cut, g in GRADES if pct >= cut)


# --------------------------------------------------------------------------- card normalisation

def _mana_value(cost: str) -> float:
    front = cost.split(" // ")[0]
    mv = 0.0
    for sym in re.findall(r"\{([^}]+)\}", front):
        if sym.isdigit():
            mv += int(sym)
        elif sym in ("X", "Y", "Z"):
            continue
        elif "/" in sym and sym.split("/")[0].isdigit():
            mv += int(sym.split("/")[0])   # {2/W}
        else:
            mv += 1
    return mv


def _colors(cost: str, text: str) -> str:
    syms = re.findall(r"\{([^}]+)\}", cost + " " + text)
    return "".join(c for c in "WUBRG" if any(c in s for s in syms))


def parse_keywords(text: str, vocab: list[str]) -> list[str]:
    """Keywords the card itself has: lines that are comma-separated keyword
    lists ("Flying, vigilance") or start with a keyword ("Ward {2}",
    "Kicker {1}{R}"). A keyword only mentioned ("creatures with flying")
    does not count."""
    found = set()
    low = {k.lower(): k for k in vocab}
    for line in text.split("\n"):
        line = re.sub(r"\(.*?\)", "", line).strip()
        for part in re.split(r"[,;]\s*", line):
            p = part.strip().lower()
            for kl, k in low.items():
                if p == kl or p.startswith(kl + " ") or p.startswith(kl + "—") or p.startswith(kl + " —"):
                    found.add(k)
    return sorted(found)


def normalise(cards: list[dict], vocab: list[str]) -> pd.DataFrame:
    """Fill in the derived fields features.engineered needs."""
    rows = []
    for c in cards:
        name = (c.get("name") or "Custom card").strip()
        cost = c.get("mana_cost") or ""
        text = c.get("oracle_text") or ""
        rarity = (c.get("rarity") or "common").lower()
        if rarity not in features.RARITIES:
            rarity = "rare" if rarity in ("special", "bonus") else "common"
        kw = c.get("keywords")
        if kw is None or (isinstance(kw, float) and np.isnan(kw)):
            kw = parse_keywords(text.replace("CARDNAME", name), vocab)

        def blank(v):
            return None if v is None or (isinstance(v, float) and np.isnan(v)) or v == "" else str(v)

        cmc = c.get("cmc")
        rows.append({
            "name": name, "scryfall_name": name, "front_name": name.split(" // ")[0],
            "mana_cost": cost, "cmc": float(cmc) if cmc is not None and np.isfinite(float(cmc)) else _mana_value(cost),
            "type_line": c.get("type_line") or "",
            "oracle_text": text,
            "power": blank(c.get("power")), "toughness": blank(c.get("toughness")),
            "loyalty": blank(c.get("loyalty")),
            "color_identity": c.get("color_identity") if isinstance(c.get("color_identity"), str)
            else "".join(c.get("color_identity") or []) or _colors(cost, text),
            "keywords": list(kw), "layout": c.get("layout") or "normal", "rarity": rarity,
        })
    return pd.DataFrame(rows)


def tfidf_text(df: pd.DataFrame) -> pd.Series:
    """Text read by the TF-IDF models (as in scripts/phase1.py)."""
    return (df["type_line"].fillna("") + " | " + df["mana_cost"].fillna("") + " | "
            + df.apply(features.card_text, axis=1)).str.replace("\n", " ")


def input_text(df: pd.DataFrame) -> pd.Series:
    """Text read by the fine-tuned encoder (scripts/phase2.py card_input_text)."""
    from phase2 import card_input_text
    return card_input_text(df)


def featurize(df: pd.DataFrame, env: pd.Series | list, bundle: dict,
              cols: list[str] | None = None, ft: bool = False) -> pd.DataFrame:
    """Engineered features + keywords + stat efficiency + set context of env,
    as the columns ``cols`` (default: the app models' columns). With ``ft``,
    the vanilla curve and set context are the ones the fine-tuned encoder was
    trained with (fitted on the 24 training sets only)."""
    cols = cols or bundle["cols"]
    curve, ctx = (bundle["ft_curve"], bundle["ft_ctx"]) if ft else (bundle["curve"], bundle["ctx"])
    f = features.engineered(df)
    kw = [bundle["kw_colmap"][c] for c in cols if c.startswith("kw_") and c in bundle["kw_colmap"]]
    f = features.add_keywords(f, df, kw)
    crea = (f["type_creature"] == 1) & f["pt_sum"].notna() & (f["pt_star"] == 0)
    f["stat_eff"] = np.where(crea, f["pt_sum"] - np.polyval(curve, f["mana_value"]), np.nan)
    env = pd.Series(list(env), index=f.index)
    for c in CTX_BASE:
        f[c] = env.map(ctx[c]).astype(float)
    f["stat_eff_vs_set"] = f["stat_eff"] - f["set_stat_eff"]
    return f.reindex(columns=cols, fill_value=0).astype(float)


def limited_gbt(df: pd.DataFrame, f: pd.DataFrame, bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """GBT + TF-IDF SVD prediction of theta_vs_repl and its predictive SD."""
    S = bundle["svd"].transform(bundle["tfidf"].transform(tfidf_text(df)))
    X = np.hstack([f.to_numpy(), S])
    mu = bundle["gbt"].predict(X)
    sd = np.clip(bundle["sd_model"].predict(f.to_numpy()), 0.02, None) * bundle["sd_scale"]
    return mu, sd


def commander_logit(df: pd.DataFrame, f: pd.DataFrame, bundle: dict) -> np.ndarray:
    from scipy import sparse
    T = bundle["cmd_tfidf"].transform(input_text(df))
    R = sparse.csr_matrix(f[[f"rarity_{r}" for r in features.RARITIES]].to_numpy() * 3.0)
    return bundle["cmd_ridge"].predict(sparse.hstack([T, R], format="csr"))


def constructed_prob(f: pd.DataFrame, bundle: dict) -> np.ndarray:
    return bundle["con_gbt"].predict(f.to_numpy())


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- fine-tuned encoder

class FineTuned:
    """Fine-tuned EmbeddingGemma-2 card model (scripts/phase2.py --save-dir),
    one or more seeds. Lower encoder layers are run live (frozen, from the
    Hugging Face checkpoint); the saved top layers, struct MLP and head on top."""

    def __init__(self, paths: list[str], threads: int = 4):
        import torch
        from phase2 import load_model
        from mtgstrength.partial import load_encoder
        torch.set_num_threads(threads)
        self.models = []
        for p in paths:
            model, meta = load_model(p)
            cfg = meta["config"]
            if cfg.get("env", "none") != "none":
                raise ValueError(f"{p}: models with an environment encoder are not supported")
            self.models.append((model, meta))
        cfg = self.models[0][1]["config"]
        self.encoder = None if cfg["encoder"] == "none" else cfg["encoder"]
        self.top_k, self.max_len = cfg["top_k"], cfg["max_len"]
        if self.encoder:
            self.tok, self.full, _ = load_encoder(self.encoder)
            n = len(self.full.layers)
            self.grab_layer = self.full.layers[n - self.top_k]

    def _lower_states(self, texts: list[str]):
        import torch
        from mtgstrength.partial import _first_arg, _Stop
        ids = self.tok(texts, truncation=True, max_length=self.max_len)["input_ids"]
        grabbed = {}

        def grab(_m, args, kwargs):
            grabbed["h"] = _first_arg(args, kwargs)
            raise _Stop

        hook = self.grab_layer.register_forward_pre_hook(grab, with_kwargs=True)
        states = []
        try:
            with torch.no_grad():
                for i in ids:
                    t = torch.tensor([i])
                    try:
                        self.full(input_ids=t, attention_mask=torch.ones_like(t))
                    except _Stop:
                        pass
                    states.append(grabbed["h"][0].float().numpy())
        finally:
            hook.remove()
        return ids, states

    def predict(self, df: pd.DataFrame, f, cache_dir: str | None = None,
                batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """f: function from a list of feature columns to the feature frame.
        With ``cache_dir`` the lower-layer states are computed in batches and
        cached on disk (mtgstrength.partial.cache_hidden, as in training);
        otherwise they are computed one card at a time."""
        import torch
        from phase2 import struct_apply
        from mtgstrength.partial import cache_hidden, pad_batch
        texts = input_text(df).tolist()
        if self.encoder and cache_dir:
            ids = self.tok(texts, truncation=True, max_length=self.max_len)["input_ids"]
            states = cache_hidden(self.encoder, texts, self.top_k, self.max_len, cache_dir)
        elif self.encoder:
            ids, states = self._lower_states(texts)
        xs = [torch.tensor(struct_apply(f(meta["struct"]["cols"]), meta["struct"]))
              if not meta["config"]["no_struct"] else None for _, meta in self.models]
        mus, sds = [], []
        with torch.no_grad():
            for (model, meta), x in zip(self.models, xs):
                mu_all, sd_all = np.empty(len(texts)), np.empty(len(texts))
                for j in np.array_split(np.arange(len(texts)), max(1, len(texts) // batch)):
                    kw = {} if x is None else {"x": x[j]}
                    if self.encoder:
                        t, h, m = pad_batch(states, ids, list(j), self.tok.pad_token_id or 0)
                        kw.update(input_ids=t, h=h, attention_mask=m)
                    mu, var = model(**kw)
                    mu_all[j], sd_all[j] = mu.numpy(), np.sqrt(var.numpy())
                mus.append(mu_all * meta["y_sd"] + meta["y_mu"])
                sds.append(sd_all * meta["y_sd"])
        return np.mean(mus, 0), np.mean(sds, 0)


def find_ft_weights(model_dir: str) -> list[str]:
    """seed*.pt files of the first run found under model_dir (or model_dir itself)."""
    if not model_dir or not os.path.isdir(model_dir):
        return []
    direct = sorted(os.path.join(model_dir, x) for x in os.listdir(model_dir)
                    if x.startswith("seed") and x.endswith(".pt"))
    if direct:
        return direct
    for sub in sorted(os.listdir(model_dir)):
        got = find_ft_weights(os.path.join(model_dir, sub)) if os.path.isdir(os.path.join(model_dir, sub)) else []
        if got:
            return got
    return []
