"""Build the web app's models and precomputed card tables.

    python app/build.py            # ~3 min on 4 CPUs

Reads the Phase 0 labels (out/<SET>/), the Phase 2/4 outputs (out/phase2,
out/phase4) and the Scryfall oracle file (downloaded if missing), and writes
app/data/:

- bundle.joblib     models for live scoring of any card (Limited, Commander,
                    Constructed) and the per-set reference distributions
- set_cards.parquet one row per (set, card) with a 17lands label: observed
                    strength, out-of-sample model estimate, ranks, grades
- cards.parquet     every Scryfall card, scored in a typical environment
- synergy.json      top synergy partners per (set, card) from the deck model
- meta.json         set list, model description, metrics

Limited model: LightGBM on the 167 engineered features + 32 SVD components
of TF-IDF ("gbt_tfidf" in Phase 1), refitted on all 31 labelled sets, with a
second GBT for a card-specific predictive SD. The estimates shown for
labelled cards are out-of-sample: for the 7 test sets they are the Phase 2
ensemble (fine-tuned EmbeddingGemma-2 + gbt_tfidf, trained only on earlier
sets); for the 24 training sets, 6-fold cross-validation over whole sets.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
from engine import ROOT, TYPICAL  # noqa: E402

from mtgstrength import features, scryfall  # noqa: E402
from phase1 import FIRST_TEST, GBT_PARAMS, LABEL_NOISE_FLOOR, SETS  # noqa: E402

OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "app", "data")
GBT_ROUNDS = 305          # gbt_tfidf rounds chosen on the validation sets (out/phase1/run_info.json)
N_FOLDS = 6
N_TYPICAL = 8             # "typical" environment = mean context of the newest 8 sets

SET_INFO = {
    "KHM": ("Kaldheim", "2021-02-05"), "STX": ("Strixhaven: School of Mages", "2021-04-23"),
    "AFR": ("Adventures in the Forgotten Realms", "2021-07-23"),
    "MID": ("Innistrad: Midnight Hunt", "2021-09-24"), "VOW": ("Innistrad: Crimson Vow", "2021-11-19"),
    "NEO": ("Kamigawa: Neon Dynasty", "2022-02-18"), "SNC": ("Streets of New Capenna", "2022-04-29"),
    "DMU": ("Dominaria United", "2022-09-09"), "BRO": ("The Brothers' War", "2022-11-18"),
    "ONE": ("Phyrexia: All Will Be One", "2023-02-10"), "MOM": ("March of the Machine", "2023-04-21"),
    "LTR": ("The Lord of the Rings: Tales of Middle-earth", "2023-06-23"),
    "WOE": ("Wilds of Eldraine", "2023-09-08"), "LCI": ("The Lost Caverns of Ixalan", "2023-11-17"),
    "MKM": ("Murders at Karlov Manor", "2024-02-09"), "OTJ": ("Outlaws of Thunder Junction", "2024-04-19"),
    "MH3": ("Modern Horizons 3", "2024-06-14"), "BLB": ("Bloomburrow", "2024-08-02"),
    "DSK": ("Duskmourn: House of Horror", "2024-09-27"), "FDN": ("Foundations", "2024-11-15"),
    "DFT": ("Aetherdrift", "2025-02-14"), "TDM": ("Tarkir: Dragonstorm", "2025-04-11"),
    "FIN": ("Final Fantasy", "2025-06-13"), "EOE": ("Edge of Eternities", "2025-08-01"),
    "TLA": ("Avatar: The Last Airbender", "2025-11-21"), "ECL": ("Lorwyn Eclipsed", "2026-01-23"),
    "TMT": ("Teenage Mutant Ninja Turtles", "2026-03-06"), "SOS": ("Secrets of Strixhaven", "2026-04-24"),
    "MSH": ("Marvel Super Heroes", "2026-06-26"), "HOB": ("The Hobbit", "2026-08-14"),
    "FRA": ("Reality Fracture", "2026-10-02"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gbt(X, y, w, rounds=GBT_ROUNDS, **kw):
    return lgb.train(dict(GBT_PARAMS, num_threads=os.cpu_count() or 4, **kw),
                     lgb.Dataset(X, y, weight=w), rounds)


def fit_text(text_tr):
    tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                          token_pattern=r"[A-Za-z]+|\{[^}]+\}|[+-]?\d+/[+-]?\d+|\d+").fit(text_tr)
    svd = TruncatedSVD(32, random_state=0).fit(tfv.transform(text_tr))
    return tfv, svd


def oracle_extras(path: str) -> pd.DataFrame:
    """Image URLs and Scryfall links, which scryfall.load_oracle_cards drops."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            img = c.get("image_uris") or ((c.get("card_faces") or [{}])[0].get("image_uris")) or {}
            back = None
            faces = c.get("card_faces") or []
            if len(faces) > 1 and faces[1].get("image_uris"):
                back = faces[1]["image_uris"].get("normal")
            rows.append({"oracle_id": c.get("oracle_id"), "image": img.get("normal"),
                         "image_small": img.get("small"), "art": img.get("art_crop"),
                         "image_back": back, "scryfall_uri": c.get("scryfall_uri"),
                         "set_type": c.get("set_type"), "games": c.get("games") or []})
    return pd.DataFrame(rows).drop_duplicates("oracle_id")


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    oracle_path = scryfall.download_oracle_cards(os.path.join(ROOT, "data", "raw"))
    log("loading labels and Scryfall data")
    oracle = scryfall.load_oracle_cards(oracle_path)
    df = features.attach_oracle(features.load_labels(OUT, SETS), oracle)
    order = {s: i for i, s in enumerate(SETS)}
    df["set_idx"] = df["set"].map(order)
    df = df.sort_values(["set_idx", "theta_vs_repl"], ascending=[True, False]).reset_index(drop=True)
    df["is_train"] = True     # the app models use every labelled set
    df["keywords"] = df["keywords"].map(lambda k: list(k) if isinstance(k, (list, np.ndarray)) else [])
    sets = [s for s in SETS if s in set(df["set"])]
    log(f"{len(df)} labelled (set, card) rows in {len(sets)} sets")

    # Feature definition: vocabulary, vanilla curve and set context, from all labelled sets.
    vocab = features.keyword_vocab(df)
    f_lib = features.add_keywords(features.engineered(df), df, vocab)
    f_lib, curve = features.add_curve_and_context(f_lib, df)
    all_kw = {k for ks in oracle["keywords"] if isinstance(ks, (list, np.ndarray)) for k in ks}
    kw_colmap = {f"kw_{k.lower().replace(' ', '_')}": k for k in sorted(all_kw | set(vocab))}
    ctx = f_lib.assign(set=df["set"]).groupby("set")[engine.CTX_BASE].first()
    ctx.loc[TYPICAL] = ctx.loc[sets[-N_TYPICAL:]].mean()
    bundle = {"kw_vocab": vocab, "kw_colmap": kw_colmap, "curve": curve, "ctx": ctx,
              "cols": list(f_lib.columns)}
    f = engine.featurize(df, df["set"], bundle)
    assert np.allclose(f.fillna(-999).to_numpy(), f_lib.astype(float).fillna(-999).to_numpy()), \
        "app features differ from mtgstrength.features"
    X = f.to_numpy()
    y = df["theta_vs_repl"].to_numpy()
    w = 1.0 / (df["theta_hand_sd"].to_numpy() ** 2 + LABEL_NOISE_FLOOR)
    text = engine.tfidf_text(df)

    # Out-of-sample estimates: whole sets held out, folds spread over the years.
    log("Limited model: cross-validated estimates")
    fold = df["set_idx"].map({s: i % N_FOLDS for i, s in enumerate(sorted(df["set_idx"].unique()))}).to_numpy()
    oof = np.empty(len(df))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        tfv, svd = fit_text(text[tr])
        XS = np.hstack([X, svd.transform(tfv.transform(text))])
        oof[te] = gbt(XS[tr], y[tr], w[tr]).predict(XS[te])

    log("Limited model: final fit on all sets")
    tfv, svd = fit_text(text)
    XS = np.hstack([X, svd.transform(tfv.transform(text))])
    bundle.update(tfidf=tfv, svd=svd, gbt=gbt(XS, y, w))

    # Predictive SD: a small GBT on |out-of-sample residual|, rescaled so the
    # z-scores of the residuals (with the label's own SD) have SD 1.
    r = y - oof
    sd_model = gbt(X, np.abs(r) * np.sqrt(np.pi / 2), np.ones(len(r)), rounds=150,
                   num_leaves=7, min_data_in_leaf=50, learning_rate=0.03)
    s_raw = np.clip(sd_model.predict(X), 0.02, None)
    lab_sd = df["theta_hand_sd"].to_numpy()
    lo, hi = 0.3, 3.0
    for _ in range(50):   # bisection on the scale
        mid = (lo + hi) / 2
        z = r / np.sqrt((mid * s_raw) ** 2 + lab_sd ** 2)
        lo, hi = (mid, hi) if z.std() > 1 else (lo, mid)
    bundle.update(sd_model=sd_model, sd_scale=(lo + hi) / 2)
    pred_sd = bundle["sd_scale"] * s_raw
    z = r / np.sqrt(pred_sd ** 2 + lab_sd ** 2)
    log(f"SD scale {bundle['sd_scale']:.3f}; 80% interval coverage {np.mean(np.abs(z) < engine.Z80):.3f}")

    # Test sets: the Phase 2 ensemble (trained on earlier sets only) where available.
    df["model_pred"], df["model_sd"], df["model_source"] = oof, pred_sd, "cv"
    p2 = pd.read_csv(os.path.join(OUT, "phase2", "test_predictions.csv"))
    p2["ens"] = 0.5 * p2["ft_embeddinggemma2_k4"] + 0.5 * p2["gbt_tfidf"]
    ens = p2.dropna(subset=["ens"]).set_index(["set", "oracle_id"])["ens"]
    key = pd.MultiIndex.from_frame(df[["set", "oracle_id"]])
    has = key.isin(ens.index)
    df.loc[has, "model_pred"] = ens.reindex(key[has]).to_numpy()
    df.loc[has, "model_source"] = "prerelease"
    log(f"{has.sum()} test-set cards use the pre-release ensemble")

    # Ranks and grades within each set.
    g = df.groupby("set")
    df["n_in_set"] = g["theta_vs_repl"].transform("size")
    df["rank_obs"] = g["theta_vs_repl"].rank(ascending=False, method="min").astype(int)
    df["pct_obs"] = 100 * g["theta_vs_repl"].rank(pct=True, method="average")
    df["rank_model"] = g["model_pred"].rank(ascending=False, method="min").astype(int)
    df["pct_model"] = 100 * g["model_pred"].rank(pct=True, method="average")
    df["grade_obs"] = df["pct_obs"].map(engine.grade)
    df["grade_model"] = df["pct_model"].map(engine.grade)
    # Reference distributions for scoring new cards: model estimates per set.
    ref = {s: np.sort(df.loc[df["set"] == s, "model_pred"].to_numpy()) for s in sets}
    ref[TYPICAL] = np.sort(np.concatenate([ref[s] for s in sets[-N_TYPICAL:]]))
    ref_obs = {s: np.sort(df.loc[df["set"] == s, "theta_vs_repl"].to_numpy()) for s in sets}
    ref_obs[TYPICAL] = np.sort(np.concatenate([ref_obs[s] for s in sets[-N_TYPICAL:]]))
    bundle.update(ref=ref, ref_obs=ref_obs)

    # Human baselines (test sets): Draftsim pre-release rating and first-week ATA.
    h = pd.read_csv(os.path.join(OUT, "phase2", "human_ratings.csv")).drop_duplicates(["set", "name"])
    df = df.merge(h[["set", "name", "draftsim", "ata_week1"]], on=["set", "name"], how="left")

    # ------------------------------------------------------------------ Commander
    log("Commander model")
    e = pd.read_csv(os.path.join(OUT, "phase4", "edhrec.csv")).drop_duplicates("oracle_id")
    e = e[e["potential_decks"] > 1000]
    e["incl"] = (e["num_decks"] + 1) / (e["potential_decks"] + 2)
    one = df.drop_duplicates("oracle_id").reset_index(drop=True)
    f_one = engine.featurize(one, one["set"], bundle)
    cm = one.merge(e[["oracle_id", "incl"]], on="oracle_id", how="inner")
    f_cm = f_one.loc[one["oracle_id"].isin(cm["oracle_id"]).to_numpy()].reset_index(drop=True)
    yc = np.log(cm["incl"] / (1 - cm["incl"])).to_numpy()
    tfc = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                          token_pattern=r"[A-Za-z]+|\{[^}]+\}|[+-]?\d+/[+-]?\d+|\d+")
    T = tfc.fit_transform(engine.input_text(cm))
    R = sparse.csr_matrix(f_cm[[f"rarity_{r}" for r in features.RARITIES]].to_numpy() * 3.0)
    TR = sparse.hstack([T, R], format="csr")
    cfold = cm["set_idx"].to_numpy() % 5
    best = None
    for a in (1, 3, 10, 30):
        err = 0.0
        for k in range(5):
            m = Ridge(alpha=a).fit(TR[cfold != k], yc[cfold != k])
            err += np.sum((m.predict(TR[cfold == k]) - yc[cfold == k]) ** 2)
        if best is None or err < best[0]:
            best = (err, a)
    bundle.update(cmd_tfidf=tfc, cmd_ridge=Ridge(alpha=best[1]).fit(TR, yc),
                  cmd_ref=np.sort(yc))
    log(f"Commander ridge alpha {best[1]}, {len(cm)} cards with EDHREC data")

    # ------------------------------------------------------------------ Constructed
    log("Constructed model")
    st = pd.read_csv(os.path.join(OUT, "phase4", "goldfish_staples.csv"))
    st["k"] = st["name"].map(lambda n: n.split(" // ")[0].lower().strip())
    st_best = st.sort_values("pct_decks", ascending=False).drop_duplicates("k")
    staple_pct = st_best.set_index("k")["pct_decks"]
    staple_fmt = st.groupby("k")["format"].agg(lambda x: ",".join(sorted(set(x))))
    staple_top = st_best.set_index("k")["format"]
    yb = one["front_name"].str.lower().str.strip().isin(staple_pct.index).astype(float).to_numpy()
    bundle["con_gbt"] = gbt(f_one.to_numpy(), yb, np.ones(len(yb)), rounds=200, objective="binary")
    bundle["con_ref"] = np.sort(bundle["con_gbt"].predict(f_one.to_numpy()))
    log(f"{int(yb.sum())} staples among {len(yb)} labelled cards")

    # ------------------------------------------------------------------ every Scryfall card
    log("scoring every Scryfall card")
    ext = oracle_extras(oracle_path)
    u = oracle[~oracle["layout"].isin(scryfall.SKIP_LAYOUTS)].drop_duplicates("oracle_id")
    u = u.merge(ext, on="oracle_id", how="left")
    u = u[~u["set_type"].isin(["funny", "memorabilia", "token", "minigame"])
          & ~u["name"].str.startswith("A-") & u["type_line"].notna()
          & ~u["type_line"].fillna("").str.contains("Basic Land|Card|Stickers|Attraction|Conspiracy")]
    u = u.reset_index(drop=True)
    un = engine.normalise([{**r, "keywords": list(r["keywords"]) if isinstance(r["keywords"], (list, np.ndarray)) else []}
                           for r in u[["name", "mana_cost", "cmc", "type_line", "oracle_text", "power",
                                       "toughness", "loyalty", "color_identity", "keywords", "layout",
                                       "rarity"]].to_dict("records")], vocab)
    un["front_name"] = u["front_name"].to_numpy()
    fu = engine.featurize(un, [TYPICAL] * len(un), bundle)
    mu, sd = engine.limited_gbt(un, fu, bundle)
    u["lim_pred"], u["lim_sd"] = mu, sd
    u["lim_pct"] = 100 * np.searchsorted(ref[TYPICAL], mu) / len(ref[TYPICAL])
    u["lim_grade"] = u["lim_pct"].map(engine.grade)
    u["cmd_pred"] = engine.commander_logit(un, fu, bundle)
    u["con_prob"] = engine.constructed_prob(fu, bundle)
    u["color_identity"] = u["color_identity"].map(lambda c: "".join(c) if isinstance(c, (list, np.ndarray)) else (c or ""))
    k = u["front_name"].str.lower().str.strip()
    u["staple_pct"] = k.map(staple_pct)
    u["staple_format"] = k.map(staple_top)
    u["staple_formats"] = k.map(staple_fmt)
    eu = e.set_index("oracle_id")
    u["edhrec_incl"] = u["oracle_id"].map(eu["incl"])
    u["edhrec_decks"] = u["oracle_id"].map(eu["num_decks"])
    u["arena"] = u["games"].map(lambda g: "arena" in list(g) if isinstance(g, (list, np.ndarray)) else False)
    labelled_sets = df.groupby("oracle_id")["set"].agg(lambda x: ",".join(x))
    u["labelled_sets"] = u["oracle_id"].map(labelled_sets).fillna("")
    keep_u = ["oracle_id", "name", "front_name", "mana_cost", "cmc", "type_line", "oracle_text",
              "power", "toughness", "loyalty", "color_identity", "keywords", "layout", "rarity", "set",
              "released_at", "image", "image_small", "image_back", "art", "scryfall_uri", "lim_pred",
              "lim_sd", "lim_pct", "lim_grade", "cmd_pred", "con_prob", "staple_pct", "staple_format",
              "staple_formats", "edhrec_incl", "edhrec_decks", "labelled_sets", "arena"]
    u = u[keep_u].copy()
    u["keywords"] = u["keywords"].map(lambda k: ", ".join(k) if isinstance(k, (list, np.ndarray)) else "")
    for c in ("power", "toughness", "loyalty", "mana_cost", "oracle_text"):
        u[c] = u[c].astype("string")
    u.to_parquet(os.path.join(DATA, "cards.parquet"), index=False)
    log(f"{len(u)} Scryfall cards scored")

    # ------------------------------------------------------------------ labelled cards table
    df = df.merge(ext[["oracle_id", "image", "image_small", "image_back", "art", "scryfall_uri"]],
                  on="oracle_id", how="left")
    df["color_identity"] = df["color_identity"].map(lambda c: "".join(c) if isinstance(c, (list, np.ndarray)) else (c or ""))
    keep = ["set", "set_idx", "name", "oracle_id", "rarity", "color_identity", "mana_cost", "cmc",
            "type_line", "theta_vs_repl", "theta_hand_sd", "ame_vs_repl_pp", "n_games_in_hand",
            "raw_wr_in_hand", "raw_iwd", "model_pred", "model_sd", "model_source", "n_in_set",
            "rank_obs", "pct_obs", "rank_model", "pct_model", "grade_obs", "grade_model",
            "draftsim", "ata_week1", "image", "image_small", "art"]
    df[[c for c in keep if c in df.columns]].to_parquet(os.path.join(DATA, "set_cards.parquet"), index=False)

    # ------------------------------------------------------------------ synergy
    log("synergy pairs")
    syn = {}
    for s in sets:
        p = os.path.join(OUT, "phase4", f"interactions_{s}.csv")
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        both = pd.concat([t[["name_a", "name_b", "synergy", "pair"]],
                          t.rename(columns={"name_a": "name_b", "name_b": "name_a"})[["name_a", "name_b", "synergy", "pair"]]])
        both = both.sort_values("synergy", ascending=False).drop_duplicates(["name_a", "name_b"])
        for name, grp in both.groupby("name_a"):
            top = grp[grp["synergy"] > 0].head(6)
            if len(top):
                syn[f"{s}|{name}"] = [{"name": r.name_b, "synergy": round(float(r.synergy), 4),
                                       "pair": r.pair} for r in top.itertuples()]
    json.dump(syn, open(os.path.join(DATA, "synergy.json"), "w"))

    # ------------------------------------------------------------------ metadata
    m1 = pd.read_csv(os.path.join(OUT, "phase2", "metrics_summary.csv"), header=[0, 1], index_col=[0, 1])
    def metric(model):
        try:
            row = m1.loc[("all", model)]
            return {"spearman": float(row[("spearman", "mean")]), "ndcg20": float(row[("ndcg@20", "mean")]),
                    "mae_pp": float(row[("mae_pp", "mean")]) if pd.notna(row[("mae_pp", "mean")]) else None}
        except KeyError:
            return None
    meta = {
        "built": time.strftime("%Y-%m-%d"),
        "sets": [{"code": s, "name": SET_INFO.get(s, (s, ""))[0], "released": SET_INFO.get(s, ("", ""))[1],
                  "n_cards": int((df["set"] == s).sum()), "test": order[s] >= order[FIRST_TEST],
                  "has_synergy": any(k.startswith(s + "|") for k in syn)} for s in sets],
        "typical_sets": sets[-N_TYPICAL:],
        "pp_per_logit": engine.PP_PER_LOGIT,
        "metrics": {"ensemble": metric("ens_embeddinggemma2_k4+gbt_tfidf"),
                    "gbt_tfidf": metric("gbt_tfidf"), "ft_gemma2": metric("ft_embeddinggemma2_k4"),
                    "draftsim": metric("human_draftsim_prerelease"), "ata_week1": metric("human_ata_week1"),
                    "rarity_only": metric("rarity_only")},
        "sd_scale": float(bundle["sd_scale"]),
        "n_scryfall": int(len(u)), "n_labelled": int(len(df)),
    }
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=1)
    joblib.dump(bundle, os.path.join(DATA, "bundle.joblib"), compress=3)
    log("done: " + ", ".join(f"{x} {os.path.getsize(os.path.join(DATA, x)) / 1e6:.1f} MB"
                             for x in sorted(os.listdir(DATA))))


if __name__ == "__main__":
    main()
