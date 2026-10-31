"""MTG card strength web app (local, CPU only).

    python app/server.py                 # http://127.0.0.1:8000
    python app/server.py --ft-dir models/ft_card --port 8000

On first start it builds app/data/ (app/build.py, about a minute). With
fine-tuned EmbeddingGemma-2 weights (scripts/phase2.py --save-dir) under
--ft-dir, custom cards are scored by the ensemble of that model and the GBT;
otherwise by the GBT alone.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import os
import sys
import threading
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
from engine import ROOT, TYPICAL  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(APP_DIR, "data")
STATIC = os.path.join(APP_DIR, "static")


def clean(v):
    """JSON-safe scalars (NaN -> None, numpy -> python)."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else round(float(v), 5)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if v is pd.NA:
        return None
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]


class State:
    def __init__(self, ft_dir: str | None, threads: int, ft_seeds: int | None = None):
        needed = ["bundle.joblib", "cards.parquet", "set_cards.parquet", "synergy.json", "meta.json"]
        if not all(os.path.exists(os.path.join(DATA, x)) for x in needed):
            print("app/data/ is missing: building it (about a minute)...", flush=True)
            import build
            build.main()
        self.bundle = joblib.load(os.path.join(DATA, "bundle.joblib"))
        self.cards = pd.read_parquet(os.path.join(DATA, "cards.parquet"))
        self.cards["_key"] = self.cards["name"].str.lower()
        self.cards["_front"] = self.cards["front_name"].str.lower()
        self.set_cards = pd.read_parquet(os.path.join(DATA, "set_cards.parquet"))
        self.synergy = json.load(open(os.path.join(DATA, "synergy.json")))
        self.meta = json.load(open(os.path.join(DATA, "meta.json")))
        self.by_id = self.cards.set_index("oracle_id")
        self.ft = None
        self.ft_status = "not configured"
        self.ft_paths = (engine.find_ft_weights(ft_dir) if ft_dir else [])[:ft_seeds]
        if ft_dir and not self.ft_paths:
            self.ft_status = f"no seed*.pt files under {ft_dir}"
        if self.ft_paths:
            self.ft_status = "loading"
            threading.Thread(target=self._load_ft, args=(threads,), daemon=True).start()

    def _load_ft(self, threads: int):
        try:
            t = time.time()
            self.ft = engine.FineTuned(self.ft_paths, threads)
            self.ft_status = f"ready ({len(self.ft_paths)} seeds)"
            print(f"fine-tuned model loaded in {time.time() - t:.0f}s", flush=True)
        except Exception as e:   # noqa: BLE001 - keep serving with the GBT
            self.ft_status = f"failed: {type(e).__name__}: {e}"
            print(f"fine-tuned model not loaded: {e}", flush=True)


app = FastAPI(title="MTG card strength")
S: State | None = None


@app.get("/api/meta")
def meta():
    return {**S.meta, "ft_status": S.ft_status, "ft_active": S.ft is not None}


@app.get("/api/search")
def search(q: str, limit: int = 12, labelled_only: bool = False):
    q = q.strip().lower()
    if len(q) < 2:
        return []
    c = S.cards
    if labelled_only:
        c = c[c["labelled_sets"] != ""]
    starts = c[c["_key"].str.startswith(q) | c["_front"].str.startswith(q)]
    words = c[c["_key"].str.contains(r"\b" + re.escape(q), regex=True)]
    contains = c[c["_key"].str.contains(q, regex=False)]
    exact = c[(c["_key"] == q) | (c["_front"] == q)]
    hit = pd.concat([exact.assign(_s=-1), starts.assign(_s=0), words.assign(_s=1), contains.assign(_s=2)]) \
        .drop_duplicates("oracle_id")
    hit = hit.assign(_l=hit["labelled_sets"] == "").sort_values(["_s", "_l", "name"]).head(limit)
    return records(hit[["oracle_id", "name", "mana_cost", "type_line", "rarity", "image_small",
                        "lim_grade", "labelled_sets"]])


@app.get("/api/random")
def random_card(labelled: bool = True):
    c = S.cards[S.cards["labelled_sets"] != ""] if labelled else S.cards
    return {"oracle_id": c.sample(1)["oracle_id"].iloc[0]}


def synergy_for(set_code: str, name: str) -> list[dict]:
    out = S.synergy.get(f"{set_code}|{name}", [])
    if not out:
        return []
    sc = S.set_cards[S.set_cards["set"] == set_code].set_index("name")
    for p in out:
        if p["name"] in sc.index:
            r = sc.loc[p["name"]]
            p.update(oracle_id=r["oracle_id"], image_small=clean(r["image_small"]),
                     grade=clean(r["grade_obs"]), mana_cost=clean(r["mana_cost"]))
    return out


_typical_cache: dict[str, dict] = {}


def typical_score(oracle_id: str, c: dict) -> dict:
    """Limited estimate in a typical recent set for a card without 17lands
    data: precomputed (Gemma + GBT when app/precompute_ft.py was run, else
    GBT), or averaged live with the fine-tuned encoder when it is loaded
    (about half a second on first view, then cached)."""
    mu, sd, model = c["lim_pred"], c["lim_sd"], c.get("lim_model") or "gbt"
    if model == "gbt" and S.ft is not None:
        if oracle_id not in _typical_cache:
            b = S.bundle
            kw = [k for k in (c.get("keywords") or "").split(", ") if k]
            df = engine.normalise([{**c, "keywords": kw}], b["kw_vocab"])
            df["front_name"] = c["front_name"]
            m_f, s_f = S.ft.predict(df, lambda cols: engine.featurize(df, [TYPICAL], b, cols, ft=True))
            _typical_cache[oracle_id] = {"ft": float(m_f[0]), "ft_sd": float(s_f[0])}
        ft = _typical_cache[oracle_id]
        mu, sd, model = 0.5 * (mu + ft["ft"]), math.sqrt(0.5 * (sd ** 2 + ft["ft_sd"] ** 2)), "ensemble"
    ref = S.bundle["ref"][TYPICAL]
    pct = lambda v: 100 * np.searchsorted(ref, v) / len(ref)  # noqa: E731
    return {"lim_pred": clean(mu), "lim_sd": clean(sd), "lim_model": model, "lim_pct": clean(pct(mu)),
            "lim_grade": engine.grade(pct(mu)), "lim_pct_lo": clean(pct(mu - engine.Z80 * sd)),
            "lim_pct_hi": clean(pct(mu + engine.Z80 * sd)),
            "lim_dist": [round(float(x), 4) for x in ref[:: max(1, len(ref) // 400)]]}


@app.get("/api/card/{oracle_id}")
def card(oracle_id: str):
    if oracle_id not in S.by_id.index:
        raise HTTPException(404, "unknown card")
    c = {k: clean(v) for k, v in S.by_id.loc[oracle_id].items() if not k.startswith("_")}
    c["oracle_id"] = oracle_id
    apps = S.set_cards[S.set_cards["oracle_id"] == oracle_id].sort_values("set_idx", ascending=False)
    names = {s["code"]: s["name"] for s in S.meta["sets"]}
    c["appearances"] = []
    for r in records(apps):
        r["set_name"] = names.get(r["set"], r["set"])
        r["synergy"] = synergy_for(r["set"], r["name"])
        # Neighbours in the observed ranking: what this card is as good as.
        sc = S.set_cards[S.set_cards["set"] == r["set"]].sort_values("rank_obs")
        i = int(np.flatnonzero(sc["oracle_id"].to_numpy() == oracle_id)[0])
        nb = sc.iloc[max(0, i - 3): i + 4]
        r["dist"] = [round(float(x), 4) for x in np.sort(sc["theta_vs_repl"].to_numpy())]
        r["neighbours"] = records(nb[["oracle_id", "name", "rank_obs", "theta_vs_repl", "grade_obs",
                                      "image_small", "rarity"]])
        c["appearances"].append(r)
    if not c["appearances"]:
        c.update(typical_score(oracle_id, c))
    c["cmd_incl_pred"] = clean(engine.sigmoid(c["cmd_pred"])) if c.get("cmd_pred") is not None else None
    obs = c.get("edhrec_incl")
    cmd = math.log(obs / (1 - obs)) if obs is not None else c["cmd_pred"]
    c["cmd_pct"] = clean(100 * np.searchsorted(S.bundle["cmd_ref"], cmd) / len(S.bundle["cmd_ref"]))
    c["con_pct"] = clean(100 * np.searchsorted(S.bundle["con_ref"], c["con_prob"]) / len(S.bundle["con_ref"]))
    return c


@app.get("/api/set/{code}")
def set_view(code: str):
    sc = S.set_cards[S.set_cards["set"] == code.upper()]
    if sc.empty:
        raise HTTPException(404, "unknown set")
    cols = ["oracle_id", "name", "rarity", "color_identity", "mana_cost", "cmc", "type_line",
            "theta_vs_repl", "theta_hand_sd", "n_games_in_hand", "raw_wr_in_hand", "model_pred",
            "model_sd", "model_source", "rank_obs", "pct_obs", "rank_model", "pct_model",
            "grade_obs", "grade_model", "draftsim", "ata_week1", "image_small", "image"]
    return records(sc.sort_values("rank_obs")[cols])


class CustomCard(BaseModel):
    name: str = "Custom card"
    mana_cost: str = ""
    type_line: str = ""
    oracle_text: str = ""
    power: str | None = None
    toughness: str | None = None
    loyalty: str | None = None
    rarity: str = "common"
    env: str = TYPICAL
    use_ft: bool = True


@app.post("/api/predict")
def predict(card: CustomCard):
    b = S.bundle
    env = card.env.upper() if card.env.upper() in b["ref"] else TYPICAL
    df = engine.normalise([card.model_dump()], b["kw_vocab"])
    f = engine.featurize(df, [env], b)
    mu_g, sd_g = engine.limited_gbt(df, f, b)
    out = {"env": env, "gbt": clean(mu_g[0]), "gbt_sd": clean(sd_g[0]),
           "keywords": df["keywords"].iloc[0], "mana_value": clean(df["cmc"].iloc[0]),
           "colors": df["color_identity"].iloc[0], "model": "gbt"}
    mu, sd = mu_g[0], sd_g[0]
    if card.use_ft and S.ft is not None:
        t = time.time()
        m_f, s_f = S.ft.predict(df, lambda cols: engine.featurize(df, [env], b, cols, ft=True))
        out.update(ft=clean(m_f[0]), ft_sd=clean(s_f[0]), ft_seconds=round(time.time() - t, 1))
        mu, sd = 0.5 * (mu + m_f[0]), math.sqrt(0.5 * (sd ** 2 + s_f[0] ** 2))
        out["model"] = "ensemble"
    ref = b["ref"][env]
    out["dist"] = [round(float(x), 4) for x in ref[:: max(1, len(ref) // 400)]]
    pct = 100 * np.searchsorted(ref, mu) / len(ref)
    out.update(pred=clean(mu), sd=clean(sd), lo=clean(mu - engine.Z80 * sd), hi=clean(mu + engine.Z80 * sd),
               pp=clean(mu * engine.PP_PER_LOGIT), pct=clean(pct), grade=engine.grade(pct),
               pct_lo=clean(100 * np.searchsorted(ref, mu - engine.Z80 * sd) / len(ref)),
               pct_hi=clean(100 * np.searchsorted(ref, mu + engine.Z80 * sd) / len(ref)))
    if env != TYPICAL:
        n = len(ref)
        out["rank"] = int((ref > mu).sum()) + 1
        out["n_in_set"] = n + 1
        sc = S.set_cards[S.set_cards["set"] == env]
        nb = sc.iloc[np.argsort(np.abs(sc["model_pred"].to_numpy() - mu))[:6]].sort_values("model_pred", ascending=False)
        out["comparable"] = records(nb[["oracle_id", "name", "rarity", "model_pred", "theta_vs_repl",
                                        "grade_obs", "image_small"]])
    else:
        recent = S.set_cards[S.set_cards["set"].isin(S.meta["typical_sets"])]
        nb = recent.iloc[np.argsort(np.abs(recent["model_pred"].to_numpy() - mu))[:6]].sort_values("model_pred", ascending=False)
        out["comparable"] = records(nb[["oracle_id", "name", "set", "rarity", "model_pred", "theta_vs_repl",
                                        "grade_obs", "image_small"]])
    cl = engine.commander_logit(df, f, b)[0]
    out["cmd_incl"] = clean(engine.sigmoid(cl))
    out["cmd_pct"] = clean(100 * np.searchsorted(b["cmd_ref"], cl) / len(b["cmd_ref"]))
    cp = engine.constructed_prob(f, b)[0]
    out["con_prob"] = clean(cp)
    out["con_pct"] = clean(100 * np.searchsorted(b["con_ref"], cp) / len(b["con_ref"]))
    return out


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    global S
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ft-dir", default=os.environ.get("MTG_FT_DIR", os.path.join(ROOT, "models", "ft_card")),
                    help="folder with fine-tuned weights (<run>/seed*.pt); skipped if absent")
    ap.add_argument("--no-ft", action="store_true", help="never load the fine-tuned model")
    ap.add_argument("--ft-seeds", type=int, default=None,
                    help="load only the first N seeds (each adds ~1 GB of RAM; 5 seeds peak at ~10 GB)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rebuild", action="store_true", help="rebuild app/data first")
    args = ap.parse_args()
    if args.rebuild:
        import build
        build.main()
    S = State(None if args.no_ft or not os.path.isdir(args.ft_dir) else args.ft_dir, args.threads,
              args.ft_seeds)
    import uvicorn
    print(f"MTG card strength: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
