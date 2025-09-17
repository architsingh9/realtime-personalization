# scripts/train_mlp.py
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

from realtime_personalization.emb_store import load_embs
from realtime_personalization.feature_joiner import make_rep
from realtime_personalization.mlp import TinyMLP

ART = Path("artifacts")
DOCS = Path("docs")
DATA = Path("data/processed")
EVENTS = DATA / "events_base.parquet"
MLP_PATH = ART / "mlp_weights.npz"
ARMS_FILE = DOCS / "arms.json"

# ---- Env-overridable hyperparams ----
EPOCHS = int(os.environ.get("EPOCHS", 8))            # default 8
NEG_PER_POS = int(os.environ.get("NEG_PER_POS", 5))  # default 5
LR = float(os.environ.get("LR", 1e-3))               # default 1e-3
WD = float(os.environ.get("WD", 5e-4))               # default 5e-4

# Fixed knobs
BATCH = 512
SEED = 42

def load_arms():
    with open(ARMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_events():
    df = pd.read_parquet(EVENTS)
    # normalize columns used downstream
    df = df.rename(columns={"userId": "user_id", "itemId": "item_id"})
    for c in ["user_id","item_id","page","device","timestamp"]:
        if c not in df.columns:
            if c in ("page","device"):
                df[c] = ""
            elif c == "timestamp":
                df[c] = np.arange(len(df))
            else:
                raise SystemExit(f"Missing required column: {c}")
    df = df.sort_values(["user_id","timestamp"]).reset_index(drop=True)
    return df

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def main():
    rng = np.random.default_rng(SEED)
    arms = load_arms()
    U,V,d = load_embs(str(ART/"user_emb.parquet"), str(ART/"item_emb.parquet"))

    wide_dim = 8
    z_dim = wide_dim + d + d
    hidden = 64

    mlp = TinyMLP(in_dim=z_dim, hidden=hidden, seed=SEED)
    if MLP_PATH.exists():
        mlp = TinyMLP.load(str(MLP_PATH))  # resume if exists

    # simple linear readout for training only
    w = rng.normal(scale=0.01, size=(hidden,)).astype(np.float32)
    wb = 0.0

    df = load_events()
    # minimal per-user store to evolve wide features chronologically
    store = {}

    buf_z, buf_y = [], []

    def z_for(uid, page, device, item):
        feats = store.get(uid, {"visitCount":0,"lastPage":"","lastDevice":""})
        z = np.array(make_rep(feats, {"page":page,"device":device}, uid, item, U, V, d), dtype=np.float32)
        return z

    def step_sgd(batch_z, batch_y):
        nonlocal w, wb, mlp
        Z = np.stack(batch_z, axis=0)
        Y = np.array(batch_y, dtype=np.float32)

        # forward
        H = mlp.forward(Z)              # [B, hidden]
        S = (H @ w) + wb                # logits
        P = sigmoid(S)

        # grads
        dS = (P - Y)                    # BCE derivative
        grad_w = H.T @ dS / len(Y)
        grad_wb = float(dS.mean())

        # backprop through MLP (two ReLU layers)
        H1 = np.maximum(0.0, Z @ mlp.W1 + mlp.b1)
        H2 = np.maximum(0.0, H1 @ mlp.W2 + mlp.b2)    # == H
        dH  = np.outer(dS, w)
        dH2 = dH * (H2 > 0)

        grad_W2 = H1.T @ dH2 / len(Y)
        grad_b2 = dH2.mean(axis=0)

        dH1 = (dH2 @ mlp.W2.T) * (H1 > 0)
        grad_W1 = Z.T @ dH1 / len(Y)
        grad_b1 = dH1.mean(axis=0)

        # optional L2 weight decay (no decay on biases)
        if WD > 0.0:
            grad_W1 += WD * mlp.W1
            grad_W2 += WD * mlp.W2
            grad_w  += WD * w

        # SGD update
        mlp.W1 -= LR * grad_W1; mlp.b1 -= LR * grad_b1
        mlp.W2 -= LR * grad_W2; mlp.b2 -= LR * grad_b2
        w      -= LR * grad_w;  wb     -= LR * grad_wb

    it = df.itertuples(index=False)
    for epoch in range(EPOCHS):
        for r in it:
            uid = str(r.user_id); pos = str(r.item_id)
            page = r.page; device = r.device

            # positive
            buf_z.append(z_for(uid, page, device, pos)); buf_y.append(1.0)
            # negatives
            neg_pool = [a for a in arms if a != pos]
            k = min(NEG_PER_POS, len(neg_pool))
            if k > 0:
                choices = neg_pool[:k] if len(neg_pool) < NEG_PER_POS else rng.choice(neg_pool, size=k, replace=False)
                for ni in choices:
                    buf_z.append(z_for(uid, page, device, str(ni))); buf_y.append(0.0)

            # evolve user features
            feats = store.get(uid, {"visitCount":0,"lastPage":"","lastDevice":""})
            feats["visitCount"] = feats.get("visitCount",0) + 1
            feats["lastPage"] = page; feats["lastDevice"] = device
            store[uid] = feats

            if len(buf_z) >= BATCH:
                step_sgd(buf_z, buf_y); buf_z.clear(); buf_y.clear()

        if buf_z: step_sgd(buf_z, buf_y); buf_z.clear(); buf_y.clear()
        it = df.itertuples(index=False)  # next epoch
        print(f"Epoch {epoch+1}/{EPOCHS} done.")

    mlp.save(str(MLP_PATH))
    print(f"✅ Saved warmed MLP → {MLP_PATH}")

if __name__ == "__main__":
    main()
