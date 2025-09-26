# scripts/export_linucb_policy.py
from __future__ import annotations
import json, sys, time, random
from pathlib import Path
import numpy as np

# --- ensure project root on sys.path ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realtime_personalization.emb_store import load_embs
from realtime_personalization.feature_joiner import make_rep
from realtime_personalization.mlp import TinyMLP
from scripts.offline_bandit import _read_events_arrow_no_pandas as read_events

ART = Path("artifacts")
DOCS = Path("docs")
ARMS_FILE = DOCS / "arms.json"
MLP_PATH  = ART / "mlp_weights.npz"
USER_EMB  = ART / "user_emb.parquet"
ITEM_EMB  = ART / "item_emb.parquet"
OUT_PATH  = ART / "heads_linucb.json"

SEED = 42
LAM  = 1e-2          # ridge prior
PRINT_EVERY = 5000

def load_arms() -> list[str]:
    with open(ARMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

class LinUCBHead:
    """Ridge-regression head with Sherman–Morrison updates.
       State: A_inv, b, theta  (all shape d or dxd)."""
    def __init__(self, d: int, lam: float = LAM):
        self.d = d
        self.A_inv = (1.0/lam) * np.eye(d, dtype=np.float32)
        self.b = np.zeros(d, dtype=np.float32)
        self.theta = np.zeros(d, dtype=np.float32)

    def update(self, x: np.ndarray, r: float):
        # Sherman–Morrison: A_inv <- A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)
        Ainv_x = self.A_inv @ x
        denom = 1.0 + float(x @ Ainv_x)
        outer = np.outer(Ainv_x, Ainv_x) / denom
        self.A_inv -= outer.astype(np.float32)
        # b, theta
        self.b += r * x
        self.theta = self.A_inv @ self.b

    def to_dict(self) -> dict:
        return {"A_inv": self.A_inv.tolist(), "theta": self.theta.tolist()}

def main():
    t0 = time.time()
    random.seed(SEED); np.random.seed(SEED)

    # ---- load artifacts ----
    arms = load_arms()
    U, V, d_emb = load_embs(str(USER_EMB), str(ITEM_EMB))
    mlp = TinyMLP.load(str(MLP_PATH)) if MLP_PATH.exists() else TinyMLP(in_dim=8+2*d_emb, hidden=64, seed=SEED)

    # feature dims
    wide_dim = 8
    hidden = 64
    d_head = hidden + 1   # +1 bias → 65

    # init heads
    heads = {a: LinUCBHead(d_head, lam=LAM) for a in arms}

    # read events (Arrow; no pandas)
    ev = read_events(ART.parent / "data/processed/events_base.parquet")
    n = ev["n"]

    # evolving user features
    store = {}  # uid -> dict(wide features)

    def rep(uid: str, page: str, device: str, item: str) -> np.ndarray:
        feats = store.get(uid, {"visitCount":0, "lastPage":"", "lastDevice":""})
        z = np.array(make_rep(feats, {"page":page, "device":device}, uid, item, U, V, d_emb), dtype=np.float32)
        h = mlp.forward(z)                  # (64,)
        h = np.concatenate([h, [1.0]], 0)   # append bias -> (65,)
        return h

    # single pass: update only the ground-truth arm per event
    updates = 0
    for i in range(n):
        uid    = ev["user_id"][i]
        page   = ev["page"][i]
        device = ev["device"][i]
        itemid = ev["item_id"][i]
        rating = float(ev["rating"][i])

        # reward proxy: click=1 if this item was shown (we're fitting from positives)
        r = 1.0  # supervised ridge; we can also weight by rating if desired

        if itemid in heads:
            x = rep(uid, page, device, itemid)
            heads[itemid].update(x, r)
            updates += 1

        # evolve user features
        feats = store.get(uid, {"visitCount":0, "lastPage":"", "lastDevice":""})
        feats["visitCount"] = feats.get("visitCount",0) + 1
        feats["lastPage"]   = page
        feats["lastDevice"] = device
        store[uid] = feats

        if (i+1) % PRINT_EVERY == 0:
            print(f"[{i+1}/{n}] updates={updates}")

    # save JSON in the same structured format {"d": d_head, "heads": {...}}
    out = {"d": d_head, "heads": {a: heads[a].to_dict() for a in arms}}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    dt = time.time() - t0
    print(f"✅ Wrote LinUCB policy snapshot → {OUT_PATH} (d={d_head}, arms={len(arms)}, updates={updates}) in {dt:.1f}s")

if __name__ == "__main__":
    main()
