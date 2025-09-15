import json, os, random
import pandas as pd
from pathlib import Path

from realtime_personalization.feature_vector import build_x
from realtime_personalization.linucb import LinUCBModel

ARTIFACTS = Path("artifacts")
DOCS = Path("docs")
DATA = Path("data/processed")

EVENTS = DATA / "events_base.parquet"
ARMS_FILE = DOCS / "arms.json"
MODEL_FILE = ARTIFACTS / "model.json"

ALPHA = 1.0  # exploration strength

def load_arms():
    with open(ARMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)  # list of itemIds/ASINs

def load_events():
    # expected columns from preprocess.py:
    # user_id, item_id, timestamp, page, device, rating (if available)
    df = pd.read_parquet(EVENTS)
    # safety: fill minimal columns
    for c in ["user_id","item_id","page","device"]:
        if c not in df.columns: df[c] = ""
    if "rating" not in df.columns:
        df["rating"] = 4.0  # neutral
    return df[["user_id","item_id","page","device","rating"]].sample(frac=1.0, random_state=42).reset_index(drop=True)

def reward_from_event(chosen_item: str, event_row) -> int:
    """
    Click simulator with structure:
      - Base chance:        0.02
      - Category match:     +0.08  (arm category matches current page)
      - Exact item match:   +0.20  (very strong signal)
      - Rating uplift:      +0.02 * (rating - 3) clipped to [0, +0.04]
    """
    base = 0.02

    # deduce simple page category bucket (same as feature_vector.bucket_page)
    p = (str(event_row.page) or "").lower()
    if "camera" in p:
        page_cat = "cameras"
    elif "computer" in p or "laptop" in p or "pc" in p:
        page_cat = "computers"
    elif "electronic" in p or "audio" in p or "tv" in p:
        page_cat = "electronics"
    else:
        page_cat = "other"

    # map chosen arm (ASIN) to a pseudo category using a simple heuristic:
    # if you have real item->category mapping, use it; otherwise infer from the chosen item's ID seen in events
    # For a simple approximation, use the current page as the arm's "intended" category:
    arm_cat = page_cat

    prob = base
    if arm_cat == page_cat:
        prob += 0.08
    if chosen_item == event_row.item_id:
        prob += 0.20

    try:
        rating = float(event_row.rating)
    except Exception:
        rating = 3.0
    prob += max(0.0, min(0.04, 0.02 * (rating - 3.0)))  # small uplift for high ratings

    # clamp to [0, 0.95] just in case
    prob = max(0.0, min(0.95, prob))
    import random
    return 1 if random.random() < prob else 0


def main():
    arms = load_arms()
    if not MODEL_FILE.exists():
        raise SystemExit("Seed the model first with scripts/init_model.py so artifacts/model.json exists.")

    model = LinUCBModel.load_json_file(str(MODEL_FILE))

    # per-user lightweight feature store for offline sim
    # we track visitCount/lastPage/lastDevice in memory to mimic DynamoDB
    user_feats = {}

    df = load_events()
    clicks = 0
    imps = 0

    for _, row in df.iterrows():
        uid = str(row.user_id)
        # default user feats if cold start
        feats = user_feats.get(uid, {"visitCount":0, "lastPage":"", "lastDevice":""})

        x = build_x(feats, {"page": row.page, "device": row.device})
        # choose arm
        best_arm, _ = model.choose(x, alpha=ALPHA)

        # observe reward
        r = reward_from_event(best_arm, row)
        clicks += r; imps += 1

        # update model
        model.update(best_arm, x, r)

        # update user feats (what Glue would do)
        feats["visitCount"] = feats.get("visitCount",0) + 1
        feats["lastPage"] = row.page
        feats["lastDevice"] = row.device
        user_feats[uid] = feats

        # occasionally print CTR
        if imps % 1000 == 0:
            print(f"[{imps}] CTR={clicks/imps:.3f}")

    print(f"Final CTR={clicks/imps:.3f}  (clicks={clicks}, imps={imps})")
    model.save_json_file(str(MODEL_FILE))
    print(f"Saved updated model to {MODEL_FILE}")

if __name__ == "__main__":
    main()
