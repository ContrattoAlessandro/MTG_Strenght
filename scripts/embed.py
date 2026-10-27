"""Frozen sentence embeddings of card text, for the lm_* ridge baselines.

    python scripts/embed.py --model BAAI/bge-base-en-v1.5 --name bge_base

Embeds every oracle ID that appears in the Phase 0 labels and writes
out/phase1/emb/<name>.parquet (oracle_id + one column per dimension). Needs
huggingface.co to be reachable for the first download.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mtgstrength import features, scryfall  # noqa: E402
from phase1 import SETS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--oracle", default="data/raw/scryfall_oracle_cards.jsonl.gz")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    df = features.attach_oracle(features.load_labels(args.out_dir, SETS),
                                scryfall.load_oracle_cards(args.oracle))
    df = df.drop_duplicates("oracle_id").reset_index(drop=True)
    text = (df["type_line"].fillna("") + ". " + df["mana_cost"].fillna("") + ". "
            + df.apply(features.card_text, axis=1)
            + df["power"].map(lambda p: "" if pd.isna(p) else f" Power {p}.")
            + df["toughness"].map(lambda t: "" if pd.isna(t) else f" Toughness {t}."))
    import torch
    model = SentenceTransformer(args.model, device="cpu",
                                model_kwargs={"dtype": torch.float32})
    E = model.encode(text.tolist(), batch_size=args.batch_size, show_progress_bar=True,
                     normalize_embeddings=True)
    out = pd.DataFrame(E, columns=[f"e{i}" for i in range(E.shape[1])])
    out.insert(0, "oracle_id", df["oracle_id"].to_numpy())
    path = os.path.join(args.out_dir, "phase1", "emb", f"{args.name}.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_parquet(path)
    print(f"wrote {path}: {E.shape}")


if __name__ == "__main__":
    main()
