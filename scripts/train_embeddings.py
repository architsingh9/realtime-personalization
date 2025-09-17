# scripts/train_embeddings.py
import os, math, json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

DATA = Path("data/processed")
ART = Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)

EVENTS = DATA / "events_base.parquet"

EMB_DIM = int(os.environ.get("EMB_DIM", "32"))
TOP_ITEMS = int(os.environ.get("TOP_ITEMS", "2000"))   # cap for speed
WINDOW = int(os.environ.get("WINDOW", "5"))            # co-vis window per user
MIN_USER_LEN = int(os.environ.get("MIN_USER_LEN", "3"))

def load_events():
    df = pd.read_parquet(EVENTS)

    # normalize expected column names
    rename_map = {
        "userId": "user_id",
        "itemId": "item_id",
    }
    df = df.rename(columns=rename_map)

    required = {"user_id", "item_id"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise SystemExit(f"Missing required columns in {EVENTS}: {missing}. Columns present: {list(df.columns)}")

    # optional timestamp
    ts_col = "timestamp" if "timestamp" in df.columns else None
    if ts_col:
        df = df.sort_values(["user_id", ts_col])
    else:
        df = df.sort_values(["user_id"]).reset_index(drop=True)

    df = df.dropna(subset=["user_id", "item_id"])
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    return df


def build_top_vocab(df, k):
    cnt = Counter(df["item_id"].values.tolist())
    most = [iid for iid, _ in cnt.most_common(k)]
    ix = {iid:i for i, iid in enumerate(most)}
    return ix, most

def user_sequences(df):
    seqs = []
    for uid, g in df.groupby("user_id"):
        items = g["item_id"].tolist()
        if len(items) >= MIN_USER_LEN:
            seqs.append(items)
    return seqs

def build_covis_matrix(seqs, item2ix, window=5):
    M = len(item2ix)
    covis = np.zeros((M, M), dtype=np.float32)
    for items in seqs:
        idxs = [item2ix[i] for i in items if i in item2ix]
        n = len(idxs)
        for t, i in enumerate(idxs):
            # bounded window around t
            lo = max(0, t - window)
            hi = min(n, t + window + 1)
            for s in range(lo, hi):
                if s == t: continue
                j = idxs[s]
                covis[i, j] += 1.0
    # symmetrize and row-normalize
    covis = (covis + covis.T) / 2.0
    row_sums = covis.sum(axis=1, keepdims=True) + 1e-8
    covis = covis / row_sums
    return covis

def spectral_embedding(covis, dim):
    # take top-d eigenvectors (largest eigenvalues) of symmetric matrix
    # numpy.linalg.eigh returns ascending order → take last dim
    vals, vecs = np.linalg.eigh(covis.astype(np.float64))
    idx = np.argsort(vals)[-dim:]
    emb = vecs[:, idx]
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb = emb / norms
    return emb.astype(np.float32)

def write_item_embeddings(item_list, emb, path):
    df = pd.DataFrame({
        "item_id": item_list,
        **{f"e{i}": emb[:, i] for i in range(emb.shape[1])}
    })
    df.to_parquet(path, index=False)

def build_user_embeddings(df, item_emb_df, dim, last_n=20):
    item_vec = {}
    cols = [c for c in item_emb_df.columns if c.startswith("e")]
    for r in item_emb_df.itertuples(index=False):
        item_vec[getattr(r, "item_id")] = np.array([getattr(r, c) for c in cols], dtype=np.float32)

    # aggregate last N item vectors per user (mean)
    user_rows = []
    for uid, g in df.groupby("user_id"):
        items = g["item_id"].tolist()
        vecs = [item_vec[i] for i in items[-last_n:] if i in item_vec]
        if vecs:
            u = np.mean(np.stack(vecs, axis=0), axis=0)
            # normalize
            n = np.linalg.norm(u) + 1e-8
            u = (u / n).astype(np.float32)
        else:
            u = np.zeros((dim,), dtype=np.float32)
        user_rows.append((uid, *u.tolist()))
    ucols = ["user_id"] + [f"e{i}" for i in range(dim)]
    return pd.DataFrame(user_rows, columns=ucols)

def main():
    print("Loading events ...")
    df = load_events()
    print(f"Events: {len(df):,}")
    print(f"Building top-{TOP_ITEMS} item vocab ...")
    item2ix, item_list = build_top_vocab(df, TOP_ITEMS)
    print(f"Users: {df['user_id'].nunique():,}  Items (kept): {len(item2ix):,}")

    print("Making user sequences ...")
    seqs = user_sequences(df)
    print(f"Sequences: {len(seqs):,}")

    print("Building co-visitation matrix ...")
    covis = build_covis_matrix(seqs, item2ix, window=WINDOW)

    print(f"Spectral embedding (dim={EMB_DIM}) ...")
    emb = spectral_embedding(covis, EMB_DIM)

    item_emb_path = ART / "item_emb.parquet"
    print(f"Writing item embeddings → {item_emb_path}")
    write_item_embeddings(item_list, emb, item_emb_path)

    print("Building user embeddings ...")
    item_emb_df = pd.read_parquet(item_emb_path)
    user_emb_df = build_user_embeddings(df, item_emb_df, EMB_DIM, last_n=20)

    user_emb_path = ART / "user_emb.parquet"
    print(f"Writing user embeddings → {user_emb_path}")
    user_emb_df.to_parquet(user_emb_path, index=False)

    print("✅ Done.")
    print(item_emb_df.head(3))
    print(user_emb_df.head(3))

if __name__ == "__main__":
    main()
