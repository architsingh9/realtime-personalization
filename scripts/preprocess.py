import os
import json
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable

RAW_REV = Path("data/raw/Electronics.jsonl.gz")
RAW_META = Path("data/raw/meta_Electronics.jsonl.gz")
OUT     = Path("data/processed/events_base.parquet")

# ---- Config (tweak via CLI args if you like)
REV_LIMIT  = int(os.getenv("REV_LIMIT", "100000"))   # set to 0 for full
META_LIMIT = int(os.getenv("META_LIMIT", "50000"))   # set to 0 for full
RNG_SEED   = int(os.getenv("RNG_SEED", "42"))

def read_jsonl_gz(path: Path, limit: int = 0) -> Iterable[dict]:
    """Yield JSON objects from a .jsonl.gz file, optionally capped by limit."""
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def to_route(categories):
    """
    Derive a top-level route like '/electronics' from 'categories'.
    The field can be a list, nested lists, or a string.
    """
    # Typical form is a flat list: ['Electronics','Television & Video','Accessories',...]
    if isinstance(categories, list) and categories:
        top = categories[0]
        if isinstance(top, list) and top:
            top = top[0]
        if isinstance(top, str) and top.strip():
            return "/" + top.strip().lower().replace(" ", "_")
    if isinstance(categories, str) and categories.strip():
        return "/" + categories.strip().lower().replace(" ", "_")
    return "/other"

def load_reviews(limit: int) -> pd.DataFrame:
    """
    Load reviews with the schema observed in EDA:
      rating, title (review title), asin, user_id, timestamp (ms), verified_purchase, helpful_vote
    """
    rows = []
    for obj in read_jsonl_gz(RAW_REV, limit):
        row = {
            "userId":  obj.get("user_id"),
            "itemId":  obj.get("asin"),
            "rating":  obj.get("rating"),
            "ts_ms":   obj.get("timestamp"),
            "verified": obj.get("verified_purchase"),
            "helpful":  obj.get("helpful_vote"),
            "review_title": obj.get("title"),
        }
        # minimal sanity: require userId + itemId
        if row["userId"] and row["itemId"]:
            rows.append(row)
    df = pd.DataFrame(rows)

    # Convert ms → seconds (int)
    df["timestamp"] = pd.to_numeric(df["ts_ms"], errors="coerce").fillna(0).astype("int64") // 1000
    df.drop(columns=["ts_ms"], inplace=True)

    return df

def load_meta(limit: int) -> pd.DataFrame:
    """
    Load metadata with the schema observed in EDA:
      parent_asin (join key), title (product title), store (brand-ish), categories (list)
    """
    rows = []
    for obj in read_jsonl_gz(RAW_META, limit):
        rows.append({
            "itemId":      obj.get("parent_asin"),
            "product_title": obj.get("title"),
            "brand":       obj.get("store"),
            "categories":  obj.get("categories"),
        })
    return pd.DataFrame(rows)

def main():
    print(f"Reading reviews from {RAW_REV} (limit={REV_LIMIT or 'ALL'}) ...")
    reviews = load_reviews(REV_LIMIT)

    print(f"Reading metadata from {RAW_META} (limit={META_LIMIT or 'ALL'}) ...")
    meta = load_meta(META_LIMIT)

    # Join on itemId (reviews.asin == meta.parent_asin)
    print("Joining reviews ↔ metadata on itemId ...")
    df = reviews.merge(meta, on="itemId", how="left")

    # Synthetic context for the pipeline
    rng = np.random.default_rng(RNG_SEED)
    df["device"] = rng.choice(["mobile", "desktop"], size=len(df), p=[0.6, 0.4])
    df["page"] = df["categories"].apply(to_route)

    # Tidy final columns
    keep = [
        "userId", "itemId", "rating", "timestamp",
        "brand", "page", "device",
        "review_title", "product_title", "categories",
        "verified", "helpful",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[keep]

    # Write Parquet
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"✅ Wrote {OUT} with {len(df):,} rows")

if __name__ == "__main__":
    main()
