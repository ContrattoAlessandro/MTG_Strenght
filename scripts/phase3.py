"""Phase 3: deck model, counterfactual card values, distillation targets.

    python scripts/phase3.py train  --work-dir <phase3_data output>
    python scripts/phase3.py values --work-dir <phase3_data output>
    python scripts/phase3.py evaluate

Deck model (design section 3): P(win) of a 17lands game from its maindeck.

- Card token = MLP(card features) + per-card ID embedding + "seen" and copy
  count embeddings. Card features are the Phase 1 engineered features plus
  the frozen EmbeddingGemma-2 text embedding. The ID embedding (one per card
  and set) is dropped at random during training (``--id-drop``), so the
  feature path has to explain wins on its own. That path is what scores
  cards from sets the model never saw games for.
- Environment z_E = MLP(mean of the feature embeddings over the set's pool),
  added to every token and to the pooled deck vector.
- Deck encoder: a 2-layer transformer over the tokens (a Set Transformer
  without positions), pooled by attention with a learned query, then an MLP.
- Per-game nuisance terms enter linearly: player skill bucket, games-played
  bucket, on the play, mulligans.

Trained only on games from the 24 training sets, with 3% of games held out
for validation.

Counterfactual card value (design: "Delta P(win) from inserting c instead
of a replacement-level card, averaged over realistic decks that can cast
it"): for each card, sample decks that can cast it, remove one random
non-land card, insert the card as seen (drawn), and average the change in
P(win), with nuisance terms at their mean. Two deck sources:

- ``real``: real decklists from the card's own set (training sets only, as
  their games are training data anyway). Uses the card's ID embedding.
- ``synth``: synthetic two-colour decks built from the set's card pool alone
  (23 random cards of the pair, 17 basics, ~40% of cards seen), and no ID
  embeddings. Needs no game data from that set, so it is a clean prediction
  for the test sets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mtgstrength import evaluate as ev  # noqa: E402
from phase1 import SETS  # noqa: E402
from phase2 import struct_matrix  # noqa: E402

RES = "phase3"
COLORS = "WUBRG"
PAIRS = ["WU", "UB", "BR", "RG", "GW", "WB", "UR", "BG", "RW", "GU"]


# --------------------------------------------------------------------------- data

def card_table(out_dir: str, emb: str):
    cards = pd.read_parquet(os.path.join(out_dir, "phase2", "cards.parquet"))
    f = pd.read_parquet(os.path.join(out_dir, "phase2", "struct.parquet"))
    X = struct_matrix(f, cards["is_train"].to_numpy())
    E = pd.read_parquet(emb).set_index("oracle_id").loc[cards["oracle_id"]].to_numpy()
    feats = np.hstack([X, E * np.sqrt(E.shape[1]) / 4]).astype(np.float32)
    # Colour identity for "can this deck cast it" (from the pip/colour features).
    colors = ["".join(c for c in COLORS if f.loc[i, f"color_{c}"] > 0) for i in range(len(f))]
    cards["colors"] = colors
    cards["is_land"] = f["type_land"].to_numpy() > 0
    return cards, feats


def load_games(work_dir: str, sets: list[str]) -> dict:
    keys = ["tok", "cnt", "seen", "won", "on_play", "skill", "log_ngames", "mull", "opp_mull",
            "n_basics", "n_other"]
    parts = {k: [] for k in keys + ["set_id", "main_colors"]}
    for s in sets:
        z = np.load(os.path.join(work_dir, f"{s}.npz"), allow_pickle=True)
        for k in keys:
            parts[k].append(z[k])
        parts["main_colors"].append(z["main_colors"])
        parts["set_id"].append(np.full(len(z["won"]), SETS.index(s), np.int16))
    return {k: np.concatenate(v) for k, v in parts.items()}


# --------------------------------------------------------------------------- model

class DeckModel(nn.Module):
    def __init__(self, feats: np.ndarray, set_of_row: np.ndarray, d: int = 96,
                 n_layers: int = 2, id_drop: float = 0.5):
        super().__init__()
        self.register_buffer("feats", torch.from_numpy(feats))
        self.register_buffer("set_of_row", torch.from_numpy(set_of_row.astype(np.int64)))
        self.n_sets = int(set_of_row.max()) + 1
        self.card = nn.Sequential(nn.Linear(feats.shape[1], 2 * d), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(2 * d, d))
        self.ids = nn.Embedding(len(feats), d)
        nn.init.zeros_(self.ids.weight)
        self.id_drop = id_drop
        self.seen = nn.Embedding(2, d)
        self.copies = nn.Embedding(5, d)
        self.env = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(3 * d + 3, d), nn.GELU(), nn.Linear(d, 1))
        self.nuis = nn.Linear(5, 1)

    def env_embeddings(self) -> torch.Tensor:
        h = self.card(self.feats)
        sums = torch.zeros(self.n_sets, h.shape[1]).index_add_(0, self.set_of_row, h)
        n = torch.bincount(self.set_of_row, minlength=self.n_sets).clamp(min=1).unsqueeze(1)
        return self.env(sums / n)

    def forward(self, tok, cnt, seen, set_id, scal, nuis, use_ids=True, env=None):
        mask = tok >= 0
        t = tok.clamp(min=0)
        h = self.card(self.feats[t])
        if use_ids:
            idv = self.ids(t)
            if self.training and self.id_drop > 0:
                keep = (torch.rand(t.shape, device=t.device) > self.id_drop).unsqueeze(-1)
                idv = idv * keep
            h = h + idv
        env = self.env_embeddings() if env is None else env
        z = env[set_id]
        h = h + self.seen(seen.long()) + self.copies(cnt.long().clamp(0, 4)) + z.unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=~mask)
        p, _ = self.pool(self.query.expand(len(h), -1, -1), h, h, key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        logit = self.head(torch.cat([p.squeeze(1), mean, z, scal], -1)).squeeze(-1)
        return logit + self.nuis(nuis).squeeze(-1)


def batch_tensors(G: dict, idx: np.ndarray):
    tok = G["tok"][idx]
    L = max(1, int((tok >= 0).sum(1).max()))
    tok = torch.from_numpy(tok[:, :L].astype(np.int64))
    cnt = torch.from_numpy(G["cnt"][idx, :L])
    seen = torch.from_numpy(G["seen"][idx, :L])
    n_tok = (tok >= 0).sum(1, keepdim=True).float()
    scal = torch.cat([torch.from_numpy(G["n_basics"][idx].astype(np.float32)).unsqueeze(1) / 17,
                      torch.from_numpy(G["n_other"][idx].astype(np.float32)).unsqueeze(1) / 5,
                      n_tok / 25], 1)
    nuis = torch.from_numpy(np.stack([G["skill"][idx], G["log_ngames"][idx] / 5, G["on_play"][idx] - 0.5,
                                      G["mull"][idx], G["opp_mull"][idx]], 1).astype(np.float32))
    set_id = torch.from_numpy(G["set_id"][idx].astype(np.int64))
    return tok, cnt, seen, set_id, scal, nuis


# --------------------------------------------------------------------------- training

def train(args) -> None:
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    res = os.path.join(args.out_dir, RES)
    os.makedirs(res, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    cards, feats = card_table(args.out_dir, args.emb)
    train_sets = sorted(set(cards.loc[cards["is_train"], "set"]), key=SETS.index)
    if args.train_sets:
        train_sets = args.train_sets
    G = load_games(args.work_dir, train_sets)
    n = len(G["won"])
    rng = np.random.default_rng(0)
    is_val = rng.random(n) < 0.03
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    set_of_row = cards["set"].map(SETS.index).to_numpy()
    model = DeckModel(feats, set_of_row, d=args.d, id_drop=args.id_drop)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr_idx) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    y = torch.from_numpy(G["won"].astype(np.float32))
    bce = nn.BCEWithLogitsLoss()
    log = open(os.path.join(res, "deck_train_log.jsonl"), "a")
    print(f"{len(tr_idx)} training games, {len(va_idx)} validation games", flush=True)

    def validate():
        model.eval()
        out = {}
        with torch.no_grad():
            env = model.env_embeddings()
            for use_ids in (True, False):
                ls, ps = [], []
                for b in np.array_split(va_idx, max(1, len(va_idx) // 4096)):
                    lg = model(*batch_tensors(G, b), use_ids=use_ids, env=env)
                    ls.append(nn.functional.binary_cross_entropy_with_logits(lg, y[b], reduction="sum").item())
                    ps.append(lg.numpy())
                out["val_loss" + ("" if use_ids else "_noid")] = sum(ls) / len(va_idx)
            # Nuisance-only baseline: the same validation games, deck ignored.
        model.train()
        return out

    # Checkpoints every 500 steps, so a restarted machine resumes where it was.
    ckpt = os.path.join(args.model_dir, "deck_ckpt.pt")
    start_ep, start_i, step = 0, 0, 0
    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        start_ep, start_i, step = c["ep"], c["i"], c["step"]
        torch.set_rng_state(c["torch_rng"])
        print(f"resumed at epoch {start_ep + 1}, batch {start_i}", flush=True)
    t0 = time.time()
    perms = [rng.permutation(tr_idx) for _ in range(args.epochs)]
    for ep in range(start_ep, args.epochs):
        run = 0.0
        batches = np.array_split(perms[ep], len(perms[ep]) // args.batch_size)
        for i, b in enumerate(batches):
            if ep == start_ep and i < start_i:
                continue
            lg = model(*batch_tensors(G, b))
            loss = bce(lg, y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
            run = 0.98 * run + 0.02 * loss.item() if run else loss.item()
            if step % 500 == 0:
                rec = {"epoch": ep + 1, "step": step, "train_loss": round(run, 5),
                       "min": round((time.time() - t0) / 60, 1)}
                if step % 2000 == 0:
                    rec.update({k: round(v, 5) for k, v in validate().items()})
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "ep": ep, "i": i + 1, "step": step,
                            "torch_rng": torch.get_rng_state()}, ckpt + ".tmp")
                os.replace(ckpt + ".tmp", ckpt)
        torch.save(model.state_dict(), os.path.join(args.model_dir, "deck_model.pt"))
    rec = {"final": True, **{k: round(v, 5) for k, v in validate().items()}}
    # Reference: a nuisance-only logistic model (no deck information).
    from sklearn.linear_model import LogisticRegression
    nu = lambda idx: batch_tensors(G, idx)[5].numpy()
    sub = tr_idx[:500_000]
    lr = LogisticRegression(max_iter=200).fit(nu(sub), G["won"][sub])
    p = lr.predict_proba(nu(va_idx))[:, 1]
    yy = G["won"][va_idx]
    rec["val_loss_nuisance_only"] = round(float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))), 5)
    rec["val_loss_constant"] = round(float(-np.mean(yy * np.log(yy.mean()) + (1 - yy) * np.log(1 - yy.mean()))), 5)
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "deck_model.pt"))


# --------------------------------------------------------------------------- counterfactual values

def synth_decks(cards: pd.DataFrame, s: str, n: int, rng) -> dict:
    """Two-colour decks built from the set pool only: 23 random non-land
    cards castable in the pair, 17 basics, each card seen with p = 0.4."""
    pool = cards[(cards["set"] == s) & ~cards["is_land"]]
    tok = np.full((n, 40), -1, np.int32)
    cnt = np.zeros((n, 40), np.int8)
    seen = np.zeros((n, 40), np.int8)
    colors = np.empty(n, dtype=object)
    for i in range(n):
        pair = PAIRS[i % len(PAIRS)]
        ok = pool[pool["colors"].map(lambda c: set(c) <= set(pair))]
        pick = rng.choice(ok.index.to_numpy(), size=min(23, len(ok)), replace=False)
        tok[i, :len(pick)] = pick
        cnt[i, :len(pick)] = 1
        seen[i, :len(pick)] = rng.random(len(pick)) < 0.4
        colors[i] = pair
    z = np.zeros(n, np.float32)
    return {"tok": tok, "cnt": cnt, "seen": seen, "main_colors": colors,
            "set_id": np.full(n, SETS.index(s), np.int16), "won": np.zeros(n, np.int8),
            "on_play": z + 0.5, "skill": z + 0.2, "log_ngames": z + 3, "mull": z, "opp_mull": z,
            "n_basics": np.full(n, 17, np.int16), "n_other": np.zeros(n, np.int16)}


@torch.no_grad()
def card_values(model, D: dict, rows: np.ndarray, cards: pd.DataFrame, use_ids: bool,
                n_decks: int, rng, env) -> np.ndarray:
    """Mean change in P(win) from inserting each card (seen) in place of a
    random non-land card, over up to n_decks decks that can cast it."""
    # Nuisance at its mean, so values reflect the deck only.
    D = dict(D)
    for k, v in (("skill", 0.2), ("log_ngames", 3.0), ("on_play", 0.5), ("mull", 0.0), ("opp_mull", 0.0)):
        D[k] = np.full(len(D["won"]), v, np.float32)
    deck_colors = np.array([set(c) for c in D["main_colors"]], dtype=object)
    is_land = cards["is_land"].to_numpy()
    out = np.full(len(rows), np.nan)
    for j, r in enumerate(rows):
        need = set(cards.at[r, "colors"])
        ok = np.flatnonzero([need <= dc for dc in deck_colors])
        if len(ok) == 0:
            continue
        idx = rng.choice(ok, size=min(n_decks, len(ok)), replace=False)
        tok, cnt, seen, set_id, scal, nuis = batch_tensors(D, idx)
        base = torch.sigmoid(model(tok, cnt, seen, set_id, scal, nuis, use_ids=use_ids, env=env))
        tok2, cnt2, seen2 = tok.clone(), cnt.clone(), seen.clone()
        tok_np = tok.numpy()
        for i in range(len(idx)):
            cand = [k for k in range(tok_np.shape[1]) if tok_np[i, k] >= 0 and not is_land[tok_np[i, k]]
                    and tok_np[i, k] != r]
            k = cand[rng.integers(len(cand))]
            if cnt2[i, k] > 1:
                cnt2[i, k] -= 1
                # put the new card in a free slot (or replace if none free)
                free = np.flatnonzero(tok_np[i] < 0)
                k = free[0] if len(free) else k
            tok2[i, k], cnt2[i, k], seen2[i, k] = r, 1, 1
        n_tok = (tok2 >= 0).sum(1, keepdim=True).float()
        scal2 = scal.clone()
        scal2[:, 2:3] = n_tok / 25
        new = torch.sigmoid(model(tok2, cnt2, seen2, set_id, scal2, nuis, use_ids=use_ids, env=env))
        out[j] = float((new - base).mean())
    return out


def values(args) -> None:
    torch.set_num_threads(args.threads)
    res = os.path.join(args.out_dir, RES)
    cards, feats = card_table(args.out_dir, args.emb)
    model = DeckModel(feats, cards["set"].map(SETS.index).to_numpy(), d=args.d)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "deck_model.pt")))
    model.eval()
    env = model.env_embeddings()
    rng = np.random.default_rng(0)
    frames = []
    for s in [x for x in SETS if x in set(cards["set"])]:
        t0 = time.time()
        rows = cards.index[(cards["set"] == s) & ~cards["is_land"]].to_numpy()
        part = cards.loc[rows, ["set", "oracle_id", "name", "rarity", "theta_vs_repl", "is_train"]].copy()
        S = synth_decks(cards, s, args.n_synth, rng)
        part["value_synth"] = card_values(model, S, rows, cards, False, args.n_decks, rng, env)
        if cards.loc[rows[0], "is_train"]:
            G = load_games(args.work_dir, [s])
            sub = rng.choice(len(G["won"]), size=min(20_000, len(G["won"])), replace=False)
            G = {k: v[sub] for k, v in G.items()}
            part["value_real"] = card_values(model, G, rows, cards, True, args.n_decks, rng, env)
            part["value_real_noid"] = card_values(model, G, rows, cards, False, args.n_decks, rng, env)
        frames.append(part)
        print(f"{s}: {len(rows)} cards, {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(res, "deck_values.csv"), index=False)


def evaluate_values(args) -> None:
    from scipy.stats import spearmanr
    res = os.path.join(args.out_dir, RES)
    v = pd.read_csv(os.path.join(res, "deck_values.csv"))
    rows = []
    for s, g in v.groupby("set", sort=False):
        r = {"set": s, "train": bool(g["is_train"].iloc[0])}
        for c in ("value_synth", "value_real", "value_real_noid"):
            if c in g and g[c].notna().sum() > 10:
                m = g[c].notna()
                r[c] = spearmanr(g.loc[m, c], g.loc[m, "theta_vs_repl"]).statistic
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(res, "deck_values_vs_theta.csv"), index=False)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string())
    print(t.groupby("train").mean(numeric_only=True).round(3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "values", "evaluate"])
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--work-dir", default="/tmp/claude-0/p3")
    ap.add_argument("--model-dir", default="/tmp/claude-0/p3")
    ap.add_argument("--emb", default="out/phase1/emb/embeddinggemma2.parquet")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--id-drop", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-decks", type=int, default=200)
    ap.add_argument("--n-synth", type=int, default=2000)
    ap.add_argument("--train-sets", nargs="*", default=None, help="debugging")
    ap.add_argument("--max-steps", type=int, default=0, help="debugging")
    args = ap.parse_args()
    {"train": train, "values": values, "evaluate": evaluate_values}[args.cmd](args)


if __name__ == "__main__":
    main()
