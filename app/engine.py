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
    Hugging Face checkpoint); the saved top layers, struct MLP and head on top.
    Runs on CPU or, with device="cuda", on a GPU."""

    def __init__(self, paths: list[str], threads: int = 4, device: str = "cpu"):
        import torch
        from phase2 import load_model
        from mtgstrength.partial import load_encoder
        torch.set_num_threads(threads)
        self.device = torch.device(device)
        import gc
        self.models = []
        shared = {}   # frozen token embeddings: one copy for all seeds (~0.8 GB each for Gemma)
        for p in paths:
            model, meta = load_model(p)
            cfg = meta["config"]
            if cfg.get("env", "none") != "none":
                raise ValueError(f"{p}: models with an environment encoder are not supported")
            text = getattr(getattr(model, "encoder", None), "text", None)
            if text is not None:
                for nm, mod in list(text.named_children()):
                    if isinstance(mod, torch.nn.Embedding) and not any(q.requires_grad for q in mod.parameters()):
                        setattr(text, nm, shared.setdefault(nm, mod))
            self.models.append((model.to(self.device), meta))
            gc.collect()
        cfg = self.models[0][1]["config"]
        self.encoder = None if cfg["encoder"] == "none" else cfg["encoder"]
        self.top_k, self.max_len = cfg["top_k"], cfg["max_len"]
        if self.encoder:
            self.tok, self.full, _ = load_encoder(self.encoder)
            self.full.to(self.device)
            n = len(self.full.layers)
            self.grab_layer = self.full.layers[n - self.top_k]

    def _lower_states(self, ids: list[list[int]]):
        """Hidden states entering the top k layers for a batch of token id
        lists, right-padded (as mtgstrength.partial.cache_hidden does)."""
        import torch
        from mtgstrength.partial import _first_arg, _Stop
        grabbed = {}

        def grab(_m, args, kwargs):
            grabbed["h"] = _first_arg(args, kwargs)
            raise _Stop

        p = self.tok.pad({"input_ids": ids}, return_tensors="pt", padding_side="right")
        t, m = p["input_ids"].to(self.device), p["attention_mask"].to(self.device)
        hook = self.grab_layer.register_forward_pre_hook(grab, with_kwargs=True)
        try:
            self.full(input_ids=t, attention_mask=m)
        except _Stop:
            pass
        finally:
            hook.remove()
        return t, grabbed["h"].float(), m

    def predict(self, df: pd.DataFrame, f, batch: int = 1, log=None) -> tuple[np.ndarray, np.ndarray]:
        """f: function from a list of feature columns to the feature frame.
        Cards are run in batches of similar length (batch=1 reproduces the
        training-time states exactly; larger batches differ by ~1e-4)."""
        import torch
        from phase2 import struct_apply
        texts = input_text(df).tolist()
        n = len(texts)
        xs = [torch.tensor(struct_apply(f(meta["struct"]["cols"]), meta["struct"]))
              if not meta["config"]["no_struct"] else None for _, meta in self.models]
        ids = self.tok(texts, truncation=True, max_length=self.max_len)["input_ids"] if self.encoder else None
        order = np.argsort([len(x) for x in ids]) if ids else np.arange(n)
        mus = np.empty((len(self.models), n))
        sds = np.empty((len(self.models), n))
        with torch.no_grad():
            for bi, j in enumerate(np.array_split(order, max(1, n // batch))):
                enc = self._lower_states([ids[i] for i in j]) if self.encoder else None
                for k, ((model, meta), x) in enumerate(zip(self.models, xs)):
                    kw = {} if x is None else {"x": x[j].to(self.device)}
                    if enc is not None:
                        kw.update(input_ids=enc[0], h=enc[1], attention_mask=enc[2])
                    mu, var = model(**kw)
                    mus[k, j] = mu.cpu().numpy() * meta["y_sd"] + meta["y_mu"]
                    # sd_scale: post-hoc calibration fitted on the validation folds
                    # (scripts/calibrate_sd.py); 1 for models saved before it.
                    sds[k, j] = np.sqrt(var.cpu().numpy()) * meta["y_sd"] * meta.get("sd_scale", 1.0)
                if log and bi % 20 == 0:
                    log(f"fine-tuned model: {min(n, (bi + 1) * batch)} / {n} cards")
        return mus.mean(0), sds.mean(0)


def find_ft_weights(model_dir: str) -> list[str]:
    """seed*.pt files of the first run found under model_dir (or model_dir itself)."""
    if not model_dir or not os.path.isdir(model_dir):
        return []
    direct = sorted(os.path.join(model_dir, x) for x in os.listdir(model_dir)
                    if x.startswith("seed") and x.endswith(".pt"))
    if direct:
        return direct
    # The step-G final model first, then any other run (alphabetical).
    subs = sorted(os.listdir(model_dir), key=lambda x: (x != "g_final", x))
    for sub in subs:
        got = find_ft_weights(os.path.join(model_dir, sub)) if os.path.isdir(os.path.join(model_dir, sub)) else []
        if got:
            return got
    return []
