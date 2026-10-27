"""Phase 2: fine-tuned card encoder (design section 3, card encoder + card head).

    python scripts/phase2.py prep
    python scripts/phase2.py train --encoder BAAI/bge-small-en-v1.5 --name bge_small --seeds 0 1 2
    python scripts/phase2.py train --encoder none --name struct_mlp          # structured-only ablation
    python scripts/phase2.py train --encoder ... --no-struct --name ..._text  # text-only ablation
    python scripts/phase2.py evaluate

Model: a pretrained text encoder (mean-pooled, fine-tuned end to end) reads
the card text; an MLP reads the Phase 1 engineered features (which include the
set-context features). The two are concatenated and a head predicts a mean
and a variance for theta_vs_repl.

Loss (design section 4):
- Gaussian NLL with total variance = predicted variance + the label's own
  posterior variance, so noisy labels count less without a hand-set weight;
- a within-set Bradley-Terry pairwise loss. Every batch holds cards from one
  set, and each pair's target is the probability that card i's true effect
  beats card j's, given both labels' posterior SDs.

Large encoders (--top-k K): only the top K layers are fine-tuned; the
frozen layers below are run once and their output cached
(mtgstrength/partial.py).

Protocol: the same temporal split as Phase 1. The epoch count is chosen on the
last three training sets (TDM, FIN, EOE), then the model is refitted on all 24
training sets and scored on the 7 test sets. Each seed is one full run; the
reported prediction is the mean over seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mtgstrength import evaluate, features, scryfall  # noqa: E402
from phase1 import FIRST_TEST, MIN_GAMES_WELL_MEASURED, N_VAL, SETS  # noqa: E402

RES = "phase2"


# --------------------------------------------------------------------------- data

def card_input_text(df: pd.DataFrame) -> pd.Series:
    """Type line, cost, P/T and rules text (own name -> CARDNAME), with the
    numbers kept as written so the encoder can see them."""
    pt = [("" if pd.isna(p) else f" {p}/{t}.") for p, t in zip(df["power"], df["toughness"])]
    loy = df["loyalty"].map(lambda x: "" if pd.isna(x) else f" Loyalty {x}.")
    return (df["rarity"] + ". " + df["type_line"].fillna("") + ". " + df["mana_cost"].fillna("")
            + "." + pd.Series(pt, index=df.index) + loy + " "
            + df.apply(features.card_text, axis=1)).str.replace("\n", " ")


def prep(args) -> None:
    res_dir = os.path.join(args.out_dir, RES)
    os.makedirs(res_dir, exist_ok=True)
    df = features.attach_oracle(features.load_labels(args.out_dir, SETS),
                                scryfall.load_oracle_cards(args.oracle))
    order = {s: i for i, s in enumerate(SETS)}
    df["set_idx"] = df["set"].map(order)
    df = df.sort_values(["set_idx", "theta_vs_repl"], ascending=[True, False]).reset_index(drop=True)
    train_sets = [s for s in SETS if order[s] < order[FIRST_TEST] and s in set(df["set"])]
    df["is_train"] = df["set"].isin(train_sets)
    df["is_val"] = df["set"].isin(train_sets[-N_VAL:])
    df["is_test"] = ~df["is_train"]
    seen = set(df.loc[df["is_train"], "oracle_id"])
    df["reprint_of_train"] = df["oracle_id"].isin(seen) & df["is_test"]

    f = features.engineered(df)
    f = features.add_keywords(f, df, features.keyword_vocab(df[df["is_train"]]))
    f, _ = features.add_curve_and_context(f, df)
    df["text"] = card_input_text(df)
    keep = ["set", "set_idx", "name", "oracle_id", "rarity", "theta_vs_repl", "theta_hand_sd",
            "ame_vs_repl_pp", "n_games_in_hand", "is_train", "is_val", "is_test",
            "reprint_of_train", "text", "type_line", "mana_cost", "power", "toughness",
            "oracle_text", "cmc"]
    df[keep].to_parquet(os.path.join(res_dir, "cards.parquet"))
    f.astype(float).to_parquet(os.path.join(res_dir, "struct.parquet"))
    print(f"{len(df)} rows, {f.shape[1]} structured features, train sets {train_sets}")


def struct_stats(f: pd.DataFrame, fit_mask: np.ndarray) -> dict:
    """Standardisation statistics (training rows only), saved with the model."""
    return {"cols": list(f.columns), "nan_cols": [c for c in f.columns if f[c].isna().any()],
            "mu": f[fit_mask].mean().to_numpy(),
            "sd": f[fit_mask].std().replace(0, 1).fillna(1).to_numpy()}


def struct_apply(f: pd.DataFrame, st: dict) -> np.ndarray:
    f = f[st["cols"]]
    miss = f[st["nan_cols"]].isna().astype(float)
    z = ((f - st["mu"]) / st["sd"]).fillna(0.0).clip(-6, 6)
    return np.hstack([z.to_numpy(), miss.to_numpy()]).astype(np.float32)


def struct_matrix(f: pd.DataFrame, fit_mask: np.ndarray) -> np.ndarray:
    """Standardise with training statistics; NaN -> 0 plus a missing flag."""
    return struct_apply(f, struct_stats(f, fit_mask))


# --------------------------------------------------------------------------- model

def build_model(encoder_name: str | None, n_struct: int, use_struct: bool, hidden: int = 256,
                top_k: int = 0, n_pool: int = 0, env_dim: int = 64):
    import torch
    from torch import nn

    class EnvEncoder(nn.Module):
        """Set Transformer over the set's card pool (design section 6).

        Pool tokens are the cards of the set (their structured features, plus
        a frozen text embedding when given). One self-attention block lets the
        cards see each other; then (a) attention pooling with a learned seed
        gives the environment vector z_E, and (b) the card being scored attends
        over the pool, which gives a card-specific context (e.g. how its stats
        compare with the set's other creatures)."""

        def __init__(self, n_in: int, n_query: int, d: int, heads: int = 4):
            super().__init__()
            self.tok = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))
            self.sab = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True,
                                                  norm_first=True)
            self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.pma = nn.MultiheadAttention(d, heads, batch_first=True)
            self.q = nn.Linear(n_query, d)
            self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
            self.out_dim = 2 * d

        def forward(self, pool, pool_mask, set_idx, query):
            # pool: (S, P, n_in); pool_mask: (S, P) True for real cards;
            # set_idx: (B,) index into S; query: (B, n_query).
            t = self.sab(self.tok(pool), src_key_padding_mask=~pool_mask)
            z, _ = self.pma(self.seed.expand(len(t), -1, -1), t, t, key_padding_mask=~pool_mask)
            q = self.q(query).unsqueeze(1)
            c, _ = self.cross(q, t[set_idx], t[set_idx], key_padding_mask=~pool_mask[set_idx])
            return torch.cat([z[set_idx, 0], c[:, 0]], -1)

    class CardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = None
            self.top_k = top_k
            d = 0
            if encoder_name and top_k:
                from mtgstrength.partial import TopLayers
                self.encoder = TopLayers(encoder_name, top_k)
                d += self.encoder.hidden_size
            elif encoder_name:
                from transformers import AutoModel
                self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)
                d += self.encoder.config.hidden_size
            self.use_struct = use_struct
            if use_struct:
                self.struct = nn.Sequential(nn.Linear(n_struct, hidden), nn.GELU(), nn.Dropout(0.1),
                                            nn.Linear(hidden, hidden), nn.GELU())
                d += hidden
            self.env = None
            if n_pool:
                self.env = EnvEncoder(n_pool, d, env_dim)
                d += self.env.out_dim
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, hidden), nn.GELU(),
                                      nn.Dropout(0.1), nn.Linear(hidden, 3))
            self.bt_scale = nn.Parameter(torch.tensor(2.0))   # Bradley-Terry slope, z-units

        def forward(self, input_ids=None, attention_mask=None, x=None, h=None, pool=None,
                    pool_mask=None, set_idx=None):
            parts = []
            if self.top_k:
                parts.append(self.encoder(input_ids, h, attention_mask))
            elif self.encoder is not None:
                h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                m = attention_mask.unsqueeze(-1).to(h.dtype)
                parts.append((h * m).sum(1) / m.sum(1).clamp(min=1))
            if self.use_struct:
                parts.append(self.struct(x))
            if self.env is not None:
                parts.append(self.env(pool, pool_mask, set_idx, torch.cat(parts, -1)))
            out = self.head(torch.cat(parts, -1))
            mu = out[:, 0]
            var = nn.functional.softplus(out[:, 1]) + 1e-3
            self.aux_out = out[:, 2]   # deck-model value head (Phase 3 distillation)
            return mu, var

    return CardModel()


def save_model(model, path: str, meta: dict) -> None:
    """Save the trained parameters only: the frozen encoder weights come from
    the Hugging Face checkpoint, so they are left out (float16 to halve the
    file; the reload check in load_model tolerates the rounding)."""
    import torch
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {k: (v.half() if v.is_floating_point() and k.startswith("encoder.") else v)
             for k, v in model.state_dict().items()
             if k in trainable or not k.startswith("encoder.")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({**meta, "state": state}, path)


def load_model(path: str):
    """Rebuild a model saved by --save-dir. Returns (model, meta); the model
    is in eval mode and takes the same inputs as during training."""
    import torch
    meta = torch.load(path, weights_only=False)
    cfg = meta["config"]
    enc = None if cfg["encoder"] == "none" else cfg["encoder"]
    n_struct = len(meta["struct"]["cols"]) + len(meta["struct"]["nan_cols"])
    model = build_model(enc, n_struct, not cfg["no_struct"], top_k=cfg["top_k"],
                        **model_kwargs(cfg))
    state = {k: v.float() if v.is_floating_point() else v for k, v in meta["state"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("encoder.") for k in missing), missing
    return model.eval(), meta


def model_kwargs(cfg: dict) -> dict:
    """Architecture options beyond the encoder."""
    if cfg.get("env", "none") == "none":
        return {}
    return {"n_pool": cfg["n_pool"], "env_dim": cfg["env_dim"]}


CTX_COLS = ["set_stat_eff", "set_removal_rate", "set_creature_rate", "set_mean_mv",
            "stat_eff_vs_set"]


def build_pools(cards: pd.DataFrame, X: np.ndarray, cfg: dict):
    """Per-set pool tensors for the environment encoder: (S, P, n) features,
    (S, P) mask, and each card's set index. Pool features are the card's
    standardised structured features, plus its frozen text embedding when
    --env-emb is given."""
    import torch
    feats = X
    if cfg.get("env_emb"):
        E = pd.read_parquet(cfg["env_emb"]).set_index("oracle_id")
        E = E.reindex(cards["oracle_id"]).fillna(0.0).to_numpy(np.float32)
        feats = np.hstack([X, E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)])
    sets = list(dict.fromkeys(cards["set"]))
    P = max((cards["set"] == s).sum() for s in sets)
    pool = torch.zeros(len(sets), P, feats.shape[1])
    mask = torch.zeros(len(sets), P, dtype=torch.bool)
    for i, s in enumerate(sets):
        rows = np.flatnonzero((cards["set"] == s).to_numpy())
        pool[i, :len(rows)] = torch.from_numpy(feats[rows].astype(np.float32))
        mask[i, :len(rows)] = True
    set_of = cards["set"].map({s: i for i, s in enumerate(sets)}).to_numpy()
    return pool, mask, set_of


def env_batch(pools, b) -> dict:
    """The pools of the sets in batch b, with set indices remapped."""
    import torch
    pool, mask, set_of = pools
    uniq, inv = np.unique(set_of[b], return_inverse=True)
    return {"pool": pool[uniq], "pool_mask": mask[uniq], "set_idx": torch.from_numpy(inv)}


def set_batches(sets: np.ndarray, idx: np.ndarray, bs: int, rng) -> list[np.ndarray]:
    """Batches that each hold cards from a single set (for the pairwise loss)."""
    out = []
    for s in np.unique(sets[idx]):
        ii = rng.permutation(idx[sets[idx] == s])
        n = max(1, round(len(ii) / bs))
        out += [b for b in np.array_split(ii, n) if len(b) > 1]
    rng.shuffle(out)
    return out


def loss_fn(mu, var, y, s, bt_scale, pair_w, lambda_w=0.0, n_set=None, lambda_k=0,
            fixed_var=0.0):
    """Gaussian NLL + within-set Bradley-Terry. With ``lambda_w`` > 0 each
    pair is reweighted LambdaRank-style by the NDCG change of swapping the two
    cards: |y_i - y_j| (linear gain, as in the NDCG metric) times the
    difference of their rank discounts 1/log2(1 + rank). Ranks are the
    current predicted ranks in the batch, scaled up to the set size
    ``n_set``. Weights are normalised to mean 1 and mixed with uniform
    weights: w = (1 - lambda_w) + lambda_w * lambda_ij. With ``lambda_k``
    the discount is cut to 0 below rank k, as in NDCG@k."""
    import torch
    if fixed_var:
        # Homoscedastic ablation: one residual variance for every card, so
        # rare cards with noisy labels are not discounted by a learned variance.
        var = torch.full_like(var, fixed_var)
    tv = var + s ** 2
    nll = 0.5 * (torch.log(tv) + (y - mu) ** 2 / tv)
    d = mu[:, None] - mu[None, :]
    target = 0.5 * (1 + torch.erf((y[:, None] - y[None, :])
                                  / torch.sqrt(2 * (s[:, None] ** 2 + s[None, :] ** 2))))
    off = ~torch.eye(len(y), dtype=torch.bool)
    w = None
    if lambda_w:
        with torch.no_grad():
            n = len(y)
            rank = torch.empty(n)
            rank[torch.argsort(mu, descending=True)] = torch.arange(n, dtype=torch.float32)
            rank = (rank + 0.5) * (n_set or n) / n + 0.5
            disc = 1.0 / torch.log2(1.0 + rank)
            if lambda_k:
                disc = torch.where(rank <= lambda_k, disc, torch.zeros_like(disc))
            lam = (y[:, None] - y[None, :]).abs() * (disc[:, None] - disc[None, :]).abs()
            lam = lam[off]
            w = (1 - lambda_w) + lambda_w * lam / lam.mean().clamp(min=1e-8)
    bt = torch.nn.functional.binary_cross_entropy_with_logits(bt_scale * d[off], target[off],
                                                              weight=w)
    return nll.mean() + pair_w * bt, nll.mean().item(), bt.item()


def run_one(cards, X, enc_name, use_struct, fit_mask, pred_mask, epochs, seed, args,
            eval_mask=None, log=print, tag="full", save_path=None):
    """Train on fit_mask; return predictions (mean, sd in label units) on
    pred_mask after the final epoch, plus per-epoch metrics on eval_mask."""
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y_all = cards["theta_vs_repl"].to_numpy()
    s_all = cards["theta_hand_sd"].to_numpy()
    if getattr(args, "target", "theta") == "label":
        # Another target (e.g. Commander inclusion, Phase 4), from --label-csv.
        y_all = cards["label"].to_numpy()
        s_all = np.full(len(cards), 0.1 * np.nanstd(y_all[fit_mask]))
        fit_mask = fit_mask & ~np.isnan(y_all)
    if getattr(args, "target", "theta") == "deck":
        # Distillation only: the deck model's counterfactual value is the
        # label, with a nominal noise SD of 10% of its spread.
        y_all = cards["deck_val"].to_numpy()
        s_all = np.full(len(cards), 0.1 * np.nanstd(y_all[fit_mask]))
        fit_mask = fit_mask & ~np.isnan(y_all)
    y_mu, y_sd = y_all[fit_mask].mean(), y_all[fit_mask].std()
    aux_w = getattr(args, "aux_w", 0.0)
    if aux_w:
        dv = cards["deck_val"].to_numpy()
        dz = torch.tensor((dv - np.nanmean(dv)) / np.nanstd(dv), dtype=torch.float32)
        has_dv = ~np.isnan(dv)
        # Consistency on unlabelled cards: test-set cards join training with
        # only the deck-value target (their labels stay unused).
        aux_extra = has_dv & cards["is_test"].to_numpy() if args.consistency_on_test else \
            np.zeros(len(cards), bool)
    labelled = fit_mask.copy()
    yz = torch.tensor((y_all - y_mu) / y_sd, dtype=torch.float32)
    sz = torch.tensor(s_all / y_sd, dtype=torch.float32)
    Xt = torch.tensor(X)

    pools = None
    if args.env != "none":
        pools = build_pools(cards, X, vars(args))
        args.n_pool = pools[0].shape[2]
    model = build_model(enc_name, X.shape[1], use_struct, top_k=args.top_k,
                        **model_kwargs(vars(args)))
    tok = None
    if enc_name:
        tok = AutoTokenizer.from_pretrained(enc_name)
        enc = tok(cards["text"].tolist(), truncation=True, max_length=args.max_len)
        ids = enc["input_ids"]
    if enc_name and args.top_k:
        from mtgstrength.partial import cache_hidden, pad_batch
        states = cache_hidden(enc_name, cards["text"].tolist(), args.top_k, args.max_len,
                              args.cache_dir, log=log)
        pad_id = tok.pad_token_id or 0

    def collate(b):
        out = {"x": Xt[b]} if use_struct else {}
        if tok is not None and args.top_k:
            t, h, m = pad_batch(states, ids, list(b), pad_id)
            out.update(input_ids=t, h=h, attention_mask=m)
        elif tok is not None:
            p = tok.pad({"input_ids": [ids[i] for i in b]}, return_tensors="pt")
            out.update(input_ids=p["input_ids"], attention_mask=p["attention_mask"])
        if pools is not None:
            out.update(env_batch(pools, b))
        return out

    enc_params = ([p for p in model.encoder.parameters() if p.requires_grad]
                  if model.encoder is not None else [])
    enc_ids = {id(p) for p in enc_params}
    other = [p for p in model.parameters() if id(p) not in enc_ids and p.requires_grad]
    groups = [{"params": other, "lr": args.head_lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sets = cards["set"].to_numpy()
    set_size = cards["set"].value_counts().to_dict()
    fit_idx = np.flatnonzero(fit_mask | aux_extra) if aux_w else np.flatnonzero(fit_mask)
    n_steps = epochs * len(set_batches(sets, fit_idx, args.batch_size, np.random.default_rng(0)))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * n_steps), n_steps)

    def predict(mask):
        model.eval()
        idx = np.flatnonzero(mask)
        # Sort by length so padding is small.
        if tok is not None:
            idx = idx[np.argsort([len(ids[i]) for i in idx])]
        mus, vs = np.empty(len(cards)), np.empty(len(cards))
        with torch.no_grad():
            for b in np.array_split(idx, max(1, len(idx) // 64)):
                mu, var = model(**collate(b))
                mus[b], vs[b] = mu.numpy(), var.numpy()
        model.train()
        return mus[mask] * y_sd + y_mu, np.sqrt(vs[mask]) * y_sd

    # Per-epoch checkpoint, so a restarted machine resumes the run.
    ckpt = os.path.join(args.cache_dir, f"ckpt_{args.name}_{tag}_seed{seed}.pt")
    hist, start = [], 0
    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        hist, start = c["hist"], c["epoch"]
        torch.set_rng_state(c["torch_rng"])
        for _ in range(start):          # replay the batch order of finished epochs
            set_batches(sets, fit_idx, args.batch_size, rng)
        log(json.dumps({"resumed": tag, "seed": seed, "after_epoch": start}))
    for ep in range(start, epochs):
        t0 = time.time()
        tot = [0.0, 0.0, 0]
        for b in set_batches(sets, fit_idx, args.batch_size, rng):
            mu, var = model(**collate(b))
            lab = torch.from_numpy(labelled[b])
            if lab.sum() >= 2:
                loss, nll, bt = loss_fn(mu[lab], var[lab], yz[b][lab], sz[b][lab], model.bt_scale,
                                        args.pair_w, args.lambda_w, set_size[sets[b[0]]],
                                        getattr(args, "lambda_k", 0),
                                        getattr(args, "fixed_var", 0.0))
            else:
                loss, nll, bt = mu.sum() * 0, 0.0, 0.0
            if aux_w:
                m = torch.from_numpy(has_dv[b])
                if m.any():
                    loss = loss + aux_w * ((model.aux_out[m] - dz[b][m]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot[0] += nll
            tot[1] += bt
            tot[2] += 1
        rec = {"epoch": ep + 1, "nll": tot[0] / tot[2], "bt": tot[1] / tot[2],
               "sec": round(time.time() - t0)}
        if eval_mask is not None:
            m, sd = predict(eval_mask)
            ev = cards[eval_mask].assign(pred=m, pred_pp=m)
            per = evaluate.evaluate(ev, ["pred"], "val")
            rec.update(val_spearman=per["spearman"].mean(), val_ndcg=per["ndcg@20"].mean(),
                       val_mae=float(np.mean(np.abs(m - ev["theta_vs_repl"]))),
                       val_nll=float(np.mean(0.5 * np.log(sd ** 2 + ev["theta_hand_sd"] ** 2)
                                             + (m - ev["theta_vs_repl"]) ** 2
                                             / (2 * (sd ** 2 + ev["theta_hand_sd"] ** 2)))))
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
        hist.append(rec)
        os.makedirs(args.cache_dir, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "hist": hist, "epoch": ep + 1, "torch_rng": torch.get_rng_state()}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)
    m, sd = predict(pred_mask)
    if save_path:
        save_model(model, save_path, {"config": vars(args), "seed": seed, "epochs": epochs,
                                      "y_mu": float(y_mu), "y_sd": float(y_sd)})
    if os.path.exists(ckpt):
        os.remove(ckpt)
    return m, sd, hist


def train(args) -> None:
    import torch
    torch.set_num_threads(args.threads)
    res_dir = os.path.join(args.out_dir, RES)
    run_dir = os.path.join(res_dir, "runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    cards = pd.read_parquet(os.path.join(res_dir, "cards.parquet"))
    f = pd.read_parquet(os.path.join(res_dir, "struct.parquet"))
    if args.no_ctx:
        f = f.drop(columns=CTX_COLS)
    if args.limit_sets:
        keep = cards["set"].isin(args.limit_sets).to_numpy()
        cards, f = cards[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    if args.deck_values:
        dv = pd.read_csv(args.deck_values)
        dv["deck_val"] = np.where(dv["is_train"], dv["value_real"], dv["value_synth"])
        cards = cards.merge(dv[["set", "oracle_id", "deck_val"]], on=["set", "oracle_id"], how="left")
        # Test-set values come from synthetic decks; keep them only when the
        # run is allowed to use them (consistency on unlabelled cards).
        if not args.consistency_on_test:
            cards.loc[cards["is_test"], "deck_val"] = np.nan
    if args.label_csv:
        lab = pd.read_csv(args.label_csv)[["set", "oracle_id", "label"]]
        cards = cards.merge(lab, on=["set", "oracle_id"], how="left")
    tr = cards["is_train"].to_numpy()
    va = cards["is_val"].to_numpy()
    te = cards["is_test"].to_numpy()
    enc = None if args.encoder == "none" else args.encoder
    use_struct = not args.no_struct
    logf = open(os.path.join(run_dir, "log.jsonl"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(json.dumps({"config": vars(args)}))
    epochs = args.epochs
    done_val = [json.loads(l)["chosen_epochs"] for l in open(os.path.join(run_dir, "log.jsonl"))
                if '"chosen_epochs"' in l]
    if done_val and not args.skip_val:
        epochs = done_val[-1]          # chosen before a restart
    elif not args.skip_val:
        # Stage 1: choose the epoch count on the validation sets (seed 0).
        X = struct_matrix(f, tr & ~va)
        _, _, hist = run_one(cards, X, enc, use_struct, tr & ~va, va, args.max_epochs, 0, args,
                             eval_mask=va, log=log, tag="val")
        h = pd.DataFrame(hist)
        # Mean of the rank-based validation scores, smoothed over neighbouring epochs.
        score = (h["val_spearman"] + h["val_ndcg"]).rolling(3, center=True, min_periods=1).mean()
        epochs = int(h["epoch"][score.idxmax()])
        log(json.dumps({"chosen_epochs": epochs}))
    X = struct_matrix(f, tr)
    preds = []
    for seed in args.seeds:
        if os.path.exists(os.path.join(run_dir, f"test_pred_seed{seed}.csv")):
            continue   # finished before a restart
        save_path = None
        if args.save_dir:
            save_path = os.path.join(args.save_dir, args.name, f"seed{seed}.pt")
        m, sd, _ = run_one(cards, X, enc, use_struct, tr, te, epochs, seed, args, log=log,
                           save_path=save_path)
        if save_path:
            import torch
            meta = torch.load(save_path, weights_only=False)
            meta["struct"] = struct_stats(f, tr)
            meta["test_pred"] = m
            torch.save(meta, save_path)
        preds.append((m, sd))
        out = cards.loc[te, ["set", "oracle_id"]].assign(pred=m, pred_sd=sd)
        out.to_csv(os.path.join(run_dir, f"test_pred_seed{seed}.csv"), index=False)
    log(json.dumps({"done": True, "epochs": epochs}))


def predict_saved(args) -> None:
    """Predict with saved weights (--model-path, one or more seeds) for the
    cards in cards.parquet (default: test sets) and check that the reloaded
    model reproduces the predictions made at the end of training."""
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(args.threads)
    res_dir = os.path.join(args.out_dir, RES)
    cards = pd.read_parquet(os.path.join(res_dir, "cards.parquet"))
    f = pd.read_parquet(os.path.join(res_dir, "struct.parquet"))
    mask = cards["set"].isin(args.limit_sets).to_numpy() if args.limit_sets else cards["is_test"].to_numpy()
    preds = []
    for path in args.model_path:
        model, meta = load_model(path)
        cfg = meta["config"]
        Xn = struct_apply(f, meta["struct"])
        X = torch.tensor(Xn)
        pools = build_pools(cards, Xn, cfg) if cfg.get("env", "none") != "none" else None
        idx = np.flatnonzero(mask)
        enc = None if cfg["encoder"] == "none" else cfg["encoder"]
        if enc:
            tok = AutoTokenizer.from_pretrained(enc)
            ids = tok(cards["text"].tolist(), truncation=True, max_length=cfg["max_len"])["input_ids"]
            from mtgstrength.partial import cache_hidden, pad_batch
            states = cache_hidden(enc, cards["text"].tolist(), cfg["top_k"], cfg["max_len"],
                                  args.cache_dir)
        mu, sd = np.empty(len(idx)), np.empty(len(idx))
        with torch.no_grad():
            for j in np.array_split(np.arange(len(idx)), max(1, len(idx) // 64)):
                b = idx[j]
                kw = {"x": X[b]} if not cfg["no_struct"] else {}
                if enc:
                    t, h, m = pad_batch(states, ids, list(b), tok.pad_token_id or 0)
                    kw.update(input_ids=t, h=h, attention_mask=m)
                if pools is not None:
                    kw.update(env_batch(pools, b))
                m_, v_ = model(**kw)
                mu[j], sd[j] = m_.numpy(), np.sqrt(v_.numpy())
        mu, sd = mu * meta["y_sd"] + meta["y_mu"], sd * meta["y_sd"]
        if not args.limit_sets and "test_pred" in meta:
            print(f"{path}: max |reloaded - saved| = {np.abs(mu - meta['test_pred']).max():.2e} logits")
        preds.append((mu, sd))
    out = cards.loc[mask, ["set", "name", "oracle_id"]].assign(
        pred=np.mean([p[0] for p in preds], 0), pred_sd=np.mean([p[1] for p in preds], 0))
    if args.pred_out:
        out.to_csv(args.pred_out, index=False)
        print(f"wrote {args.pred_out}")


# --------------------------------------------------------------------------- zero-shot

def zeroshot(args) -> None:
    """Frozen embeddings + ridge, exactly as the Phase 1 lm_* models: the
    embedding plus rarity one-hots, alpha chosen on the validation sets, then
    refitted on all training sets. Writes runs/zs_<name>/."""
    from sklearn.linear_model import Ridge

    res_dir = os.path.join(args.out_dir, RES)
    cards = pd.read_parquet(os.path.join(res_dir, "cards.parquet"))
    f = pd.read_parquet(os.path.join(res_dir, "struct.parquet"))
    tr, va, te = (cards[c].to_numpy() for c in ("is_train", "is_val", "is_test"))
    y = cards["theta_vs_repl"].to_numpy()
    w = 1.0 / (cards["theta_hand_sd"].to_numpy() ** 2 + 0.02)
    R = f[[f"rarity_{r}" for r in features.RARITIES]].to_numpy()
    for path in args.embeddings:
        name = os.path.basename(path).removesuffix(".parquet")
        E = pd.read_parquet(path).set_index("oracle_id")
        M = np.hstack([E.loc[cards["oracle_id"]].to_numpy(), R])
        fit = tr & ~va
        best = None
        for a in (1, 3, 10, 30, 100, 300, 1000, 3000):
            m = Ridge(alpha=a).fit(M[fit], y[fit], sample_weight=w[fit])
            err = np.average((m.predict(M[va]) - y[va]) ** 2, weights=w[va])
            if best is None or err < best[0]:
                best = (err, a)
        m = Ridge(alpha=best[1]).fit(M[tr], y[tr], sample_weight=w[tr])
        run_dir = os.path.join(res_dir, "runs", f"zs_{name}")
        os.makedirs(run_dir, exist_ok=True)
        cards.loc[te, ["set", "oracle_id"]].assign(pred=m.predict(M[te]), pred_sd=np.nan) \
            .to_csv(os.path.join(run_dir, "test_pred_seed0.csv"), index=False)
        print(f"zs_{name}: alpha {best[1]}")


# --------------------------------------------------------------------------- evaluation

def evaluate_all(args) -> None:
    res_dir = os.path.join(args.out_dir, RES)
    p1 = pd.read_csv(os.path.join(args.out_dir, "phase1", "test_predictions.csv"))
    k = json.load(open(os.path.join(args.out_dir, "phase1", "run_info.json")))["pp_per_logit"]
    base_models = ["rarity_only", "gbt", "tfidf", "gbt_tfidf", "lm_bge_base"]
    out = p1[["set", "name", "oracle_id", "rarity", "theta_vs_repl", "theta_hand_sd",
              "ame_vs_repl_pp", "n_games_in_hand", "reprint_of_train"] + base_models].copy()
    sd_cols = {}
    runs_dir = os.path.join(res_dir, "runs")
    for name in sorted(os.listdir(runs_dir)):
        if name.endswith("_commander"):
            continue   # other target (Phase 4)
        files = sorted(x for x in os.listdir(os.path.join(runs_dir, name)) if x.startswith("test_pred_seed"))
        if not files:
            continue
        ps = [pd.read_csv(os.path.join(runs_dir, name, x)) for x in files]
        m = pd.concat([p[["set", "oracle_id", "pred"]] for p in ps]).groupby(["set", "oracle_id"])["pred"].mean()
        sd = pd.concat([p[["set", "oracle_id", "pred_sd"]] for p in ps]).groupby(["set", "oracle_id"])["pred_sd"].mean()
        key = pd.MultiIndex.from_frame(out[["set", "oracle_id"]])
        out[name] = m.reindex(key).to_numpy()
        if sd.notna().any():
            sd_cols[name] = sd.reindex(key).to_numpy()
        print(f"{name}: {len(files)} seeds")
    # Ensembles with the best Phase 1 model.
    runs = [c for c in out.columns if c.startswith(("ft_", "zs_", "struct_", "deck_"))]
    for c in runs:
        if c.startswith("ft_") and not c.endswith("_text"):
            out[f"ens_{c[3:]}+gbt_tfidf"] = 0.5 * out[c] + 0.5 * out["gbt_tfidf"]
    models = base_models + runs + [c for c in out.columns if c.startswith("ens_")]
    for c in models:
        out[c + "_pp"] = k * out[c]
    # Human baselines (scripts/human_baselines.py): rank-only, no MAE.
    hpath = os.path.join(res_dir, "human_ratings.csv")
    if os.path.exists(hpath):
        h = pd.read_csv(hpath).drop_duplicates(["set", "name"])
        out = out.merge(h[["set", "name", "draftsim", "ata_week1"]], on=["set", "name"], how="left")
        out["human_draftsim_prerelease"] = out.pop("draftsim")
        out["human_ata_week1"] = -out.pop("ata_week1")
        for c in ["human_draftsim_prerelease", "human_ata_week1"]:
            out[c + "_pp"] = np.nan
            models.append(c)
    for c, sd in sd_cols.items():
        out[c + "_sd"] = sd
    out.to_csv(os.path.join(res_dir, "test_predictions.csv"), index=False)

    subsets = {
        "all": out,
        f"well_measured (>= {MIN_GAMES_WELL_MEASURED} games in hand)":
            out[out["n_games_in_hand"] >= MIN_GAMES_WELL_MEASURED],
        "new_cards (oracle ID not in training sets)": out[~out["reprint_of_train"]],
    }
    if "human_draftsim_prerelease" in out:
        subsets["human_rated (cards with both human ratings)"] = out[
            out["human_draftsim_prerelease"].notna() & out["human_ata_week1"].notna()]
    per_set = pd.concat([evaluate.evaluate(d, models, name) for name, d in subsets.items()])
    per_set.to_csv(os.path.join(res_dir, "metrics_per_set.csv"), index=False)
    summary = (per_set.groupby(["subset", "model"], sort=False)
               [["spearman", "spearman_within_rarity", "ndcg@20", "top20_overlap", "mae_pp"]]
               .agg(["mean", "std"]).round(3))
    summary.to_csv(os.path.join(res_dir, "metrics_summary.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    for sub in summary.index.get_level_values(0).unique():
        print(sub)
        print(summary.loc[sub])
    print(per_set[per_set["subset"] == "all"].pivot(index="model", columns="set", values="spearman")
          .round(2).loc[models])
    contamination_table(per_set[per_set["subset"] == "all"], res_dir)
    # Calibration of the predicted uncertainty: z-scores and interval coverage.
    rows = []
    for c, sd in sd_cols.items():
        tv = np.sqrt(sd ** 2 + out["theta_hand_sd"] ** 2)
        z = (out["theta_vs_repl"] - out[c]) / tv
        rows.append({"model": c, "z_sd": z.std(), "cover_50": (z.abs() < 0.674).mean(),
                     "cover_90": (z.abs() < 1.645).mean(),
                     "spearman_sd_vs_abs_err": pd.Series(sd).corr(
                         (out["theta_vs_repl"] - out[c]).abs(), method="spearman")})
    if rows:
        cal = pd.DataFrame(rows).round(3)
        cal.to_csv(os.path.join(res_dir, "calibration.csv"), index=False)
        print(cal)


# When each frozen encoder could have seen a test set: its release date, or
# the documented training-data cutoff when the model card gives one.
ENCODER_DATES = {
    "zs_mpnet_base": "2021-08-30", "zs_bge_base": "2023-09-12",
    "zs_qwen3_emb_0_6b": "2025-06-03", "zs_harrier_0_6b": "2026-03-30",
    "zs_nemotron35_1b": "2026-07-14",   # text tower = Nemotron-3-Embed-1B (Jul 2026)
    "zs_embeddinggemma2": "2025-01-31",  # model card: pre-training data cutoff Jan 2025
}
# First day of PremierDraft on Arena (17lands draft dumps); spoilers start
# about four weeks earlier.
SET_DATES = {"TLA": "2025-11-18", "ECL": "2026-01-20", "TMT": "2026-03-03", "SOS": "2026-04-21",
             "MSH": "2026-06-23", "HOB": "2026-08-11", "FRA": "2026-09-29"}
OLD_ENCODERS = ["zs_mpnet_base", "zs_bge_base", "zs_qwen3_emb_0_6b"]


def contamination_table(per: pd.DataFrame, res_dir: str) -> None:
    """Per-set Spearman gain of each newer encoder over the mean of the three
    older ones, split by whether the set's spoiler season (4 weeks before
    Arena release) came before the encoder's date. A model that memorised
    community ratings should gain more on the sets it could have seen."""
    sp = per.pivot(index="set", columns="model", values="spearman")
    if not set(OLD_ENCODERS) <= set(sp.columns):
        return
    base = sp[OLD_ENCODERS].mean(axis=1)
    rows = []
    for m, d in ENCODER_DATES.items():
        if m not in sp or m in OLD_ENCODERS:
            continue
        for s in sp.index:
            seen = pd.Timestamp(SET_DATES[s]) - pd.Timedelta(days=28) < pd.Timestamp(d)
            rows.append({"model": m, "set": s, "could_have_seen": seen,
                         "gain_vs_old": sp.loc[s, m] - base[s]})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(res_dir, "contamination_per_set.csv"), index=False)
    print(t.groupby(["model", "could_have_seen"])["gain_vs_old"].agg(["mean", "count"]).round(3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "zeroshot", "evaluate", "predict"])
    ap.add_argument("--model-path", nargs="*", default=[], help="predict: saved seed files")
    ap.add_argument("--pred-out", default=None, help="predict: output CSV")
    ap.add_argument("--embeddings", nargs="*", default=[])
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--oracle", default="data/raw/scryfall_oracle_cards.jsonl.gz")
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--name", default="bge_small")
    ap.add_argument("--no-struct", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=None, help="with --skip-val")
    ap.add_argument("--skip-val", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--enc-lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--pair-w", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=0,
                    help="fine-tune only the top k encoder layers, on cached lower-layer states")
    ap.add_argument("--cache-dir", default="/tmp/claude-0/cache")
    ap.add_argument("--deck-values", default=None,
                    help="out/phase3/deck_values.csv: enables --target deck and --aux-w")
    ap.add_argument("--target", choices=["theta", "deck", "label"], default="theta")
    ap.add_argument("--label-csv", default=None, help="set, oracle_id, label (for --target label)")
    ap.add_argument("--aux-w", type=float, default=0.0,
                    help="weight of an auxiliary head fitted to the deck-model values")
    ap.add_argument("--consistency-on-test", action="store_true",
                    help="also fit the auxiliary head on test-set cards (deck values from "
                         "synthetic decks, no labels)")
    ap.add_argument("--limit-sets", nargs="*", default=None, help="debugging")
    ap.add_argument("--lambda-w", type=float, default=0.0,
                    help="LambdaRank-style NDCG weighting of the pairwise loss (0 = off, 1 = full)")
    ap.add_argument("--lambda-k", type=int, default=0,
                    help="truncate the LambdaRank discount at this rank (NDCG@k); 0 = no cut")
    ap.add_argument("--fixed-var", type=float, default=0.0,
                    help="ablation: fixed residual variance (z-units) instead of the learned one")
    ap.add_argument("--env", choices=["none", "set"], default="none",
                    help="set: Set Transformer environment encoder over the set's card pool")
    ap.add_argument("--env-dim", type=int, default=64)
    ap.add_argument("--env-emb", default=None,
                    help="frozen embedding parquet added to the pool tokens (out/phase1/emb/...)")
    ap.add_argument("--no-ctx", action="store_true",
                    help="drop the 5 hand-made set-context features")
    ap.add_argument("--save-dir", default=None,
                    help="save each seed's trained weights to <save-dir>/<name>/seed<k>.pt")
    args = ap.parse_args()
    {"prep": prep, "train": train, "zeroshot": zeroshot, "evaluate": evaluate_all,
     "predict": predict_saved}[args.cmd](args)


if __name__ == "__main__":
    main()
