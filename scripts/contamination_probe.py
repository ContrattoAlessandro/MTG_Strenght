"""Encoder-only contamination probes (docs/RESEARCH_PLAN.md, step C).

    # (a) name-only probe: embed each card's NAME alone, ridge on training-set names
    python scripts/contamination_probe.py names --device cuda
    # (b) paraphrase sensitivity of fine-tuned models (saved with phase2.py --save-dir)
    python scripts/contamination_probe.py paraphrase --model-path models/ft_card/<run>/seed*.pt \
        --encoder-date 2025-01-31 --sets test

(a) A card's name says little about its strength beyond rarity and creature
type, so an encoder that ranks cards from their names better on sets it could
have read about (spoiler season before the encoder's date) than on sets it
could not, relative to clean controls (mpnet 2021, bge 2023), has memorised
reputations. Eval sets: the 9 validation sets (each fold's ridge trained on the
sets before it) and the 7 test sets (ridge trained on all 24 training sets).
The statistic per encoder is gain_seen - gain_unseen, where gain is the
within-set Spearman minus the mean of the controls, with a Welch SE.
Flag: >= 0.03.

(b) Meaning-preserving rewrites of the rules text: ability lines in reverse
order, reminder text (in parentheses) removed, CARDNAME -> "this card". A
memoriser keys on surface form it has seen, so it should lose more on seen sets.
Flag: drop on seen sets >= 0.02 larger than on unseen sets.

Both write to out/contamination/. Scoring on the test sets is logged in
out/TEST_LOG.md (diagnostic look, no model choice).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from phase1 import SETS  # noqa: E402
from phase2 import FOLDS, SET_DATES, text_variants  # noqa: E402

# Arena release of the validation sets (spoilers start about 4 weeks earlier).
VAL_DATES = {"MKM": "2024-02-06", "OTJ": "2024-04-16", "MH3": "2024-06-11", "BLB": "2024-07-30",
             "DSK": "2024-09-24", "FDN": "2024-11-12", "TDM": "2025-04-08", "FIN": "2025-06-10",
             "EOE": "2025-07-29"}
ALL_DATES = {**VAL_DATES, **SET_DATES}
# HF id -> (short name, date it could have seen data up to: release date, or
# documented training cutoff).
ENCODERS = {
    "sentence-transformers/all-mpnet-base-v2": ("mpnet", "2021-08-30"),
    "BAAI/bge-base-en-v1.5": ("bge", "2023-09-12"),
    "google/embeddinggemma-2": ("embeddinggemma2", "2025-01-31"),
    "Qwen/Qwen3-Embedding-0.6B": ("qwen3_emb_0_6b", "2025-06-05"),
    "Qwen/Qwen3-Embedding-4B": ("qwen3_emb_4b", "2025-06-05"),
    "Qwen/Qwen3-Embedding-8B": ("qwen3_emb_8b", "2025-06-05"),
    "microsoft/harrier-oss-v1-0.6b": ("harrier_0_6b", "2026-03-30"),
    "nvidia/Nemotron-3.5-Embed-1B-BF16": ("nemotron35_1b", "2026-07-14"),
}
CONTROLS = ["mpnet", "bge"]
RES = "contamination"


def could_have_seen(set_code: str, date: str) -> bool:
    return pd.Timestamp(ALL_DATES[set_code]) - pd.Timedelta(days=28) < pd.Timestamp(date)


def log_test_look(out_dir: str, runs: str, why: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(out_dir, "TEST_LOG.md"), "a") as fh:
        fh.write(f"| {now} | {runs} | {why} |\n")


def eval_groups(cards: pd.DataFrame):
    """(name, fit mask, eval mask) for the 3 validation folds and the test sets."""
    order = {s: i for i, s in enumerate(SETS)}
    tr = cards["is_train"].to_numpy()
    for fold, sets in FOLDS.items():
        first = min(order[s] for s in sets)
        yield fold, tr & (cards["set_idx"] < first).to_numpy(), cards["set"].isin(sets).to_numpy()
    yield "test", tr, cards["is_test"].to_numpy()


def ridge_scores(M: np.ndarray, cards: pd.DataFrame) -> dict[str, float]:
    """Within-set Spearman of a ridge on M for every eval set; alpha by
    leave-one-set-out over the last 3 fit sets."""
    from sklearn.linear_model import Ridge
    y = cards["theta_vs_repl"].to_numpy()
    w = 1.0 / (cards["theta_hand_sd"].to_numpy() ** 2 + 0.02)
    out = {}
    for _, fit, ev in eval_groups(cards):
        fit_sets = list(dict.fromkeys(cards.loc[fit, "set"]))
        inner = cards["set"].isin(fit_sets[-3:]).to_numpy()
        best = None
        for a in (1, 3, 10, 30, 100, 300, 1000):
            m = Ridge(alpha=a).fit(M[fit & ~inner], y[fit & ~inner], sample_weight=w[fit & ~inner])
            p = m.predict(M[inner])
            sc = np.mean([spearmanr(y[inner][g], p[g]).statistic
                          for g in [(cards.loc[inner, "set"] == s).to_numpy() for s in fit_sets[-3:]]])
            if best is None or sc > best[0]:
                best = (sc, a)
        m = Ridge(alpha=best[1]).fit(M[fit], y[fit], sample_weight=w[fit])
        p = m.predict(M[ev])
        for s in dict.fromkeys(cards.loc[ev, "set"]):
            g = (cards.loc[ev, "set"] == s).to_numpy()
            out[s] = spearmanr(y[ev][g], p[g]).statistic
    return out


def names(args) -> None:
    import torch
    from sentence_transformers import SentenceTransformer
    res = os.path.join(args.out_dir, RES)
    os.makedirs(res, exist_ok=True)
    cards = pd.read_parquet(os.path.join(args.out_dir, "phase2", "cards.parquet"))
    R = pd.get_dummies(cards["rarity"]).to_numpy(float)
    rows = [{"encoder": "rarity_only", "date": None, "set": s, "spearman": v}
            for s, v in ridge_scores(R, cards).items()]
    uniq = list(dict.fromkeys(cards["name"]))
    for hf in args.encoders or ENCODERS:
        short, date = ENCODERS[hf]
        path = os.path.join(args.emb_dir, f"names_{short}.npy")
        if os.path.exists(path):
            E = np.load(path)
        else:
          try:
            fp16 = args.device == "cuda" and "gemma" not in hf   # EmbeddingGemma needs fp32
            model = SentenceTransformer(hf, device=args.device, trust_remote_code=True,
                                        model_kwargs={"dtype": torch.float16 if fp16 else torch.float32})
            E = model.encode(uniq, batch_size=args.batch_size, normalize_embeddings=True,
                             show_progress_bar=False)
            os.makedirs(args.emb_dir, exist_ok=True)
            np.save(path, E)
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
          except Exception as e:   # one encoder failing (download, remote code) skips it
            print(f"{short}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        pos = {n: i for i, n in enumerate(uniq)}
        M = np.hstack([E[[pos[n] for n in cards["name"]]], R])
        for s, v in ridge_scores(M, cards).items():
            rows.append({"encoder": short, "date": date, "set": s, "spearman": v})
        print(f"{short}: done", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(res, "names_per_set.csv"), index=False)
    summarise_names(per, res)
    log_test_look(args.out_dir, "contamination_probe names (all encoders)",
                  "step C: name-only contamination diagnostic, no model choice")


def summarise_names(per: pd.DataFrame, res: str) -> None:
    sp = per.pivot(index="set", columns="encoder", values="spearman")
    ctrl = sp[[c for c in CONTROLS if c in sp]].mean(axis=1)
    dates = per.dropna(subset=["date"]).groupby("encoder")["date"].first()
    rows = []
    for enc, date in dates.items():
        if enc in CONTROLS:
            continue
        g = (sp[enc] - ctrl).rename("gain").to_frame()
        g["seen"] = [could_have_seen(s, date) for s in g.index]
        a, b = g.loc[g["seen"], "gain"], g.loc[~g["seen"], "gain"]
        diff = a.mean() - b.mean() if len(a) and len(b) else np.nan
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 1 and len(b) > 1 else np.nan
        rows.append({"encoder": enc, "date": date, "n_seen": len(a), "n_unseen": len(b),
                     "gain_seen": a.mean(), "gain_unseen": b.mean(), "seen_minus_unseen": diff,
                     "se": se, "flag": bool(diff >= 0.03) if pd.notna(diff) else None,
                     "mean_spearman": sp[enc].mean()})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(res, "names_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(sp.round(3))
    print(t.round(3))


def paraphrase(args) -> None:
    import torch
    from phase2 import load_model, predict_with
    res = os.path.join(args.out_dir, RES)
    os.makedirs(res, exist_ok=True)
    cards = pd.read_parquet(os.path.join(args.out_dir, "phase2", "cards.parquet"))
    f = pd.read_parquet(os.path.join(args.out_dir, "phase2", "struct.parquet"))
    sets = list(SET_DATES) if args.sets == ["test"] else args.sets
    # Only the probed sets (whole sets, so env pools stay complete).
    keep = cards["set"].isin(sets).to_numpy()
    cards, f = cards[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    mask = np.ones(len(cards), bool)
    variants = {"original": cards["text"], **text_variants(cards)}
    preds = {k: [] for k in variants}
    for path in args.model_path:
        model, meta = load_model(path)
        if meta["config"].get("no_ctx"):
            from phase2 import CTX_COLS
            ff = f.drop(columns=CTX_COLS)
        else:
            ff = f
        for k, texts in variants.items():
            preds[k].append(predict_with(model, meta, cards, ff, mask, args.cache_dir,
                                         texts=texts.tolist(), device=args.device)[0])
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    sub = cards[mask]
    rows = []
    for k, ps in preds.items():
        p = np.mean(ps, 0)
        changed = (variants[k][mask] != cards["text"][mask]).to_numpy()
        for s in dict.fromkeys(sub["set"]):
            g = (sub["set"] == s).to_numpy()
            rows.append({"variant": k, "set": s, "seen": could_have_seen(s, args.encoder_date),
                         "share_changed": changed[g].mean(),
                         "spearman": spearmanr(sub["theta_vs_repl"][g], p[g]).statistic})
    per = pd.DataFrame(rows)
    orig = per[per["variant"] == "original"].set_index("set")["spearman"]
    per["drop"] = per["set"].map(orig) - per["spearman"]
    name = args.tag or os.path.basename(os.path.dirname(args.model_path[0]))
    per.to_csv(os.path.join(res, f"paraphrase_{name}.csv"), index=False)
    summ = per[per["variant"] != "original"].groupby(["variant", "seen"])["drop"].agg(["mean", "count"])
    print(per.round(3))
    print(summ.round(4))
    if any(s in SET_DATES for s in sets):
        log_test_look(args.out_dir, f"contamination_probe paraphrase {name}",
                      "step C: paraphrase-sensitivity diagnostic, no model choice")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["names", "paraphrase"])
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--encoders", nargs="*", default=None, help="HF ids (default: all in ENCODERS)")
    ap.add_argument("--emb-dir", default="work/names_emb")
    ap.add_argument("--model-path", nargs="*", default=[])
    ap.add_argument("--encoder-date", default="2025-01-31",
                    help="paraphrase: date the fine-tuned model's encoder could have seen data to")
    ap.add_argument("--sets", nargs="+", default=["test"], help="paraphrase: sets, or 'test'")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cache-dir", default="work/cache")
    args = ap.parse_args()
    {"names": names, "paraphrase": paraphrase}[args.cmd](args)


if __name__ == "__main__":
    main()
