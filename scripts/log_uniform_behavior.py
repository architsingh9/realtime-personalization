# scripts/log_uniform_behavior.py
from __future__ import annotations
import csv, json, random, time
from pathlib import Path
import numpy as np

from realtime_personalization.emb_store import load_embs  # just to ensure artifacts exist; not needed for uniform
# We will reuse the same reward model used in offline_bandit.py
# (clicked if chosen == event's item_id, with rating-shaped prob)

ART = Path("artifacts")
DOCS = Path("docs")
DATA = Path("data/processed")

EVENTS    = DATA / "events_base.parquet"
ARMS_FILE = DOCS / "arms.json"
LOG_PATH  = ART / "logged_uniform.csv"

SEED = 123

def _read_events_arrow_no_pandas(path: Path):
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
        else:
            return np.full(n, default, dtype=object)

    def get_float_col(name: str, default: float = 4.0) -> np.ndarray:
        if name in tbl.column_names:
            arr = tbl[name].to_numpy(zero_copy_only=False)
            out = np.empty(n, dtype=np.float32)
            for i, v in enumerate(arr):
                out[i] = float(default) if v is None else float(v)
            return out
        else:
            return np.full(n, float(default), dtype=np.float32)

    user_id = get_str_col("userId")
    item_id = get_str_col("itemId")
    page    = get_str_col("page",   "")
    device  = get_str_col("device", "")
    rating  = get_float_col("rating", 4.0)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(user_id))
    def take(a): return a[perm]

    return {
        "user_id": take(user_id),
        "item_id": take(item_id),
        "page":    take(page),
        "device":  take(device),
        "rating":  take(rating),
        "n":       len(user_id),
    }

def reward_from_event(chosen_item: str, row_item_id: str, row_rating: float) -> int:
    base = 0.05
    if chosen_item == row_item_id:
        rating = float(row_rating if row_rating is not None else 4.0)
        prob = 0.2 + (min(max(rating, 1.0), 5.0) - 1.0) * (0.6 / 4.0)
    else:
        prob = base
    return 1 if random.random() < prob else 0

def load_arms():
    with open(ARMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Sanity checks
    for pth, msg in [
        (EVENTS,    "Run scripts/preprocess.py first to produce events_base.parquet."),
        (ARMS_FILE, "Run scripts/select_arms.py to produce docs/arms.json."),
    ]:
        if not Path(pth).exists():
            raise SystemExit(f"Missing required artifact: {pth}\nHint: {msg}")

    arms = load_arms()
    if not arms:
        raise SystemExit("docs/arms.json is empty.")
    p_b = 1.0 / float(len(arms))  # uniform propensity

    ev = _read_events_arrow_no_pandas(EVENTS)
    n = ev["n"]

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        # Minimal OPE log schema
        w.writerow(["user_id","page","device","gt_item","rating","action","reward","p_b"])
        clicks = 0
        for i in range(n):
            uid    = ev["user_id"][i]
            page   = ev["page"][i]
            device = ev["device"][i]
            gt     = ev["item_id"][i]
            rating = float(ev["rating"][i])

            action = random.choice(arms)  # uniform policy
            r = reward_from_event(action, gt, rating)
            clicks += r

            w.writerow([uid, page, device, gt, rating, action, r, p_b])

            if (i+1) % 1000 == 0:
                ctr = clicks / float(i+1)
                print(f"[{i+1}] logging… CTR={ctr:.3f}")

    print(f"✅ Wrote uniform logged data → {LOG_PATH} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
