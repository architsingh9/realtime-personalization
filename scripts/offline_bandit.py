# scripts/offline_bandit.py
import json, random, csv, time
from pathlib import Path
import pandas as pd
import numpy as np

# Where to write the CTR curve sampled every 1,000 impressions
RUN_LOG = Path("artifacts/offline_ctr_run.csv")

from realtime_personalization.emb_store import load_embs
from realtime_personalization.feature_joiner import make_rep
from realtime_personalization.mlp import TinyMLP
from realtime_personalization.neurallinear import NeuralLinearModel

ART = Path("artifacts")
DOCS = Path("docs")
DATA = Path("data/processed")

EVENTS     = DATA / "events_base.parquet"
ARMS_FILE  = DOCS / "arms.json"
MLP_PATH   = ART / "mlp_weights.npz"
HEADS_PATH = ART / "bandit_heads.json"
USER_EMB   = ART / "user_emb.parquet"
ITEM_EMB   = ART / "item_emb.parquet"

SEED = 42  # reproducibility for offline replay

def load_arms():
    with open(ARMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_events():
    df = pd.read_parquet(EVENTS)
    # normalize names expected downstream
    df = df.rename(columns={"userId":"user_id", "itemId":"item_id"})
    for c in ["user_id","item_id","page","device"]:
        if c not in df.columns:
            df[c] = "" if c in ("page","device") else ""
    if "rating" not in df.columns:
        df["rating"] = 4.0
    # shuffle with fixed seed for deterministic replay
    return df[["user_id","item_id","page","device","rating"]].sample(frac=1.0, random_state=SEED).reset_index(drop=True)

def reward_from_event(chosen_item: str, event_row) -> int:
    """Click simulator used for offline replay."""
    base = 0.05
    if chosen_item == event_row.item_id:
        rating = float(getattr(event_row, "rating", 4.0))
        prob = 0.2 + (min(max(rating, 1.0), 5.0) - 1.0) * (0.6 / 4.0)
    else:
        prob = base
    return 1 if random.random() < prob else 0

def main():
    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    # Sanity checks for artifacts
    for pth, msg in [
        (EVENTS,     "Run scripts/preprocess.py first to produce events_base.parquet."),
        (ARMS_FILE,  "Run scripts/select_arms.py to produce docs/arms.json."),
        (USER_EMB,   "Run scripts/train_embeddings.py to produce user/item embeddings."),
        (ITEM_EMB,   "Run scripts/train_embeddings.py to produce user/item embeddings."),
        (MLP_PATH,   "Run scripts/train_mlp.py to warm-start the MLP (mlp_weights.npz)."),
        (HEADS_PATH, "Run scripts/init_mlp_and_heads.py to init bandit heads."),
    ]:
        if not Path(pth).exists():
            raise SystemExit(f"Missing required artifact: {pth}\nHint: {msg}")

    # Load artifacts
    arms = load_arms()
    U, V, emb_dim = load_embs(str(USER_EMB), str(ITEM_EMB))
    mlp = TinyMLP.load(str(MLP_PATH))
    model = NeuralLinearModel.load_json(str(HEADS_PATH))

    # Lightweight per-user store to evolve wide features
    user_feats = {}
    df = load_events()

    clicks = 0
    imps = 0
    hist = []  # (impressions, clicks, ctr)

    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        uid = str(row.user_id)
        feats = user_feats.get(uid, {"visitCount": 0, "lastPage": "", "lastDevice": ""})

        # Build per-arm hidden reps for Thompson decision
        h_by_arm = {}
        for a in arms:
            z = np.array(
                make_rep(feats, {"page": row.page, "device": row.device}, uid, a, U, V, emb_dim),
                dtype=np.float32,
            )
            h = mlp.forward(z)                           # (HIDDEN,)
            h = np.concatenate([h, np.array([1.0], dtype=np.float32)])  # append bias → (HIDDEN+1,)
            h_by_arm[a] = h

        # Choose arm via Thompson sampling over heads
        chosen = model.choose_thompson(h_by_arm)

        # Simulate feedback and update
        r = reward_from_event(chosen, row)
        clicks += r
        imps += 1
        model.update(chosen, h_by_arm[chosen], r)

        # Evolve user features (as if maintained in an online store)
        feats["visitCount"] = feats.get("visitCount", 0) + 1
        feats["lastPage"]   = row.page
        feats["lastDevice"] = row.device
        user_feats[uid]     = feats

        if imps % 1000 == 0:
            ctr = clicks / imps
            hist.append((imps, clicks, ctr))
            print(f"[{imps}] CTR={ctr:.3f}")

    final_ctr = clicks / imps if imps else 0.0
    print(f"Final CTR={final_ctr:.3f}  (clicks={clicks}, imps={imps})")

    # Persist updated heads and the CTR curve
    model.save_json(str(HEADS_PATH))
    print(f"Saved updated heads to {HEADS_PATH}")

    with open(RUN_LOG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impressions", "clicks", "ctr"])
        w.writerows(hist)
    print(f"📈 Wrote CTR curve to {RUN_LOG} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
