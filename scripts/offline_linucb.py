# scripts/offline_linucb.py
import json, random, csv, time
from pathlib import Path
import numpy as np

from realtime_personalization.emb_store import load_embs
from realtime_personalization.feature_joiner import make_rep
from realtime_personalization.mlp import TinyMLP

ART = Path("artifacts")
DOCS = Path("docs")
DATA = Path("data/processed")

EVENTS    = DATA / "events_base.parquet"
ARMS_FILE = DOCS / "arms.json"
MLP_PATH  = ART / "mlp_weights.npz"

RUN_LOG = ART / "offline_ctr_linucb.csv"

SEED = 42
LAMBDA = float(json.loads(json.dumps({"v": 0.01}))["v"])  # ridge; keeps pure stdlib
ALPHA  = float(json.loads(json.dumps({"v": 0.20}))["v"])  # UCB exploration scale


def _read_events_arrow_no_pandas(path: Path):
    """
    Arrow-only reader for events_base.parquet.
    Returns dict of numpy arrays (length n): user_id, item_id, page, device, rating
    """
    import pyarrow.parquet as pq
    tbl = pq.read_table(str(path))
    n = tbl.num_rows

    def get_str_col(name: str, default: str = "") -> np.ndarray:
        if name in tbl.column_names:
            arr = tbl[name].to_numpy(zero_copy_only=False)
            out = np.empty(n, dtype=object)
            for i, v in enumerate(arr):
                out[i] = "" if v is None else str(v)
            return out
        return np.full(n, default, dtype=object)

    def get_float_col(name: str, default: float = 4.0) -> np.ndarray:
        if name in tbl.column_names:
            arr = tbl[name].to_numpy(zero_copy_only=False)
            out = np.empty(n, dtype=np.float32)
            for i, v in enumerate(arr):
                out[i] = float(default) if v is None else float(v)
            return out
        return np.full(n, float(default), dtype=np.float32)

    user_id = get_str_col("userId")
    item_id = get_str_col("itemId")
    page    = get_str_col("page",   "")
    device  = get_str_col("device", "")
    rating  = get_float_col("rating", 4.0)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)

    def take(a): return a[perm]

    return {
        "user_id": take(user_id),
        "item_id": take(item_id),
        "page":    take(page),
        "device":  take(device),
        "rating":  take(rating),
        "n":       n,
    }


def load_arms():
    with open(ARMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def reward_from_event(chosen_item: str, row_item_id: str, row_rating: float) -> int:
    base = 0.05
    if chosen_item == row_item_id:
        rating = float(row_rating if row_rating is not None else 4.0)
        prob = 0.2 + (min(max(rating, 1.0), 5.0) - 1.0) * (0.6 / 4.0)
    else:
        prob = base
    return 1 if random.random() < prob else 0


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Artifact checks
    for pth, hint in [
        (EVENTS, "Run scripts/preprocess.py first."),
        (ARMS_FILE, "Run scripts/select_arms.py to produce docs/arms.json."),
        (ART / "user_emb.parquet", "Run scripts/train_embeddings.py first."),
        (ART / "item_emb.parquet", "Run scripts/train_embeddings.py first."),
        (MLP_PATH, "Run scripts/train_mlp.py to warm-start the MLP."),
    ]:
        if not Path(pth).exists():
            raise SystemExit(f"Missing required artifact: {pth}\nHint: {hint}")

    arms = load_arms()
    U, V, emb_dim = load_embs(str(ART / "user_emb.parquet"), str(ART / "item_emb.parquet"))
    mlp = TinyMLP.load(str(MLP_PATH))  # we still use the same representation
    
    # LinUCB per-arm parameters over the hidden rep + bias (65-dim)
    HIDDEN = 64
    D = HIDDEN + 1  # +1 bias
    A = {a: (LAMBDA * np.eye(D, dtype=np.float32)) for a in arms}   # dxd
    A_inv = {a: np.linalg.inv(A[a]) for a in arms}                  # dxd
    b = {a: np.zeros(D, dtype=np.float32) for a in arms}            # d

    def rep(uid, page, device, item):
        feats = store.get(uid, {"visitCount": 0, "lastPage": "", "lastDevice": ""})
        z = np.array(make_rep(feats, {"page": page, "device": device}, uid, item, U, V, emb_dim), dtype=np.float32)
        h = mlp.forward(z)  # (64,)
        return np.concatenate([h, np.array([1.0], dtype=np.float32)], axis=0)  # (65,)

    ev = _read_events_arrow_no_pandas(EVENTS)
    n = ev["n"]
    store = {}

    clicks = 0
    imps = 0
    hist = []
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        uid    = ev["user_id"][i]
        page   = ev["page"][i]
        device = ev["device"][i]
        itemid = ev["item_id"][i]
        rating = float(ev["rating"][i])

        # UCB selection
        x_by_arm = {}
        scores = []
        for a in arms:
            x = rep(uid, page, device, a)           # (65,)
            Ainv = A_inv[a]
            theta = Ainv @ b[a]                     # (65,)
            mean = float(theta @ x)
            var  = float(np.sqrt(np.maximum(1e-12, x @ (Ainv @ x))))
            ucb  = mean + ALPHA * var
            x_by_arm[a] = x
            scores.append((ucb, a))
        chosen = max(scores, key=lambda t: t[0])[1]

        # reward + update
        r = reward_from_event(chosen, itemid, rating)
        clicks += r
        imps += 1

        x = x_by_arm[chosen]
        A[chosen] += np.outer(x, x)
        b[chosen] += r * x
        # keep inverse in sync (recompute; D=65, cheap)
        A_inv[chosen] = np.linalg.inv(A[chosen])

        # evolve user features
        feats = store.get(uid, {"visitCount": 0, "lastPage": "", "lastDevice": ""})
        feats["visitCount"] = feats.get("visitCount", 0) + 1
        feats["lastPage"] = page
        feats["lastDevice"] = device
        store[uid] = feats

        if imps % 1000 == 0:
            ctr = clicks / imps
            hist.append((imps, clicks, ctr))
            print(f"[{imps}] CTR={ctr:.3f}")

    final_ctr = clicks / imps if imps else 0.0
    print(f"Final CTR={final_ctr:.3f}  (clicks={clicks}, imps={imps})")

    with open(RUN_LOG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impressions", "clicks", "ctr"])
        w.writerows(hist)
    print(f"📈 Wrote CTR curve to {RUN_LOG} at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
