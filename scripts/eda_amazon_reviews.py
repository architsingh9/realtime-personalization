"""
EDA for Amazon Reviews (Electronics) Dataset
---------------------------------------------
This script inspects the raw Amazon Review Electronics dataset before preprocessing.
It:
  - Loads a small sample of the JSONL.GZ files
  - Prints out available fields
  - Summarizes record counts
  - Shows rating distributions
  - Checks unique users/items
  - Explores categories structure
"""

import gzip
import json
import pandas as pd
from collections import Counter
from pathlib import Path

RAW_REV = Path("data/raw/Electronics.jsonl.gz")
RAW_META = Path("data/raw/meta_Electronics.jsonl.gz")

def sample_jsonl(path, n=5):
    """Print a few raw records for inspection"""
    print(f"\n=== Sample from {path.name} ===")
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= n: break
            print(json.loads(line))

def field_distribution(path, max_lines=50_000):
    """Count fields appearing in the dataset"""
    keys = Counter()
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            obj = json.loads(line)
            keys.update(obj.keys())
    print(f"\n=== Field frequency in {path.name} (first {max_lines} rows) ===")
    for k, v in keys.most_common(20):
        print(f"{k:20} {v}")

def quick_stats_reviews(path, max_lines=100_000):
    """Basic EDA on reviews"""
    rows = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            obj = json.loads(line)
            rows.append(obj)
    df = pd.DataFrame(rows)

    print("\n=== Review dataset stats ===")
    print(f"Total rows sampled: {len(df):,}")
    if "overall" in df.columns:
        print("Rating distribution (overall):")
        print(df["overall"].value_counts())
    elif "rating" in df.columns:
        print("Rating distribution (rating):")
        print(df["rating"].value_counts())

    if "reviewerID" in df.columns:
        user_col = "reviewerID"
    elif "user_id" in df.columns:
        user_col = "user_id"
    else:
        user_col = None

    if "asin" in df.columns:
        item_col = "asin"
    elif "item_id" in df.columns:
        item_col = "item_id"
    else:
        item_col = None

    if user_col and item_col:
        print(f"Unique users: {df[user_col].nunique():,}")
        print(f"Unique items: {df[item_col].nunique():,}")

def quick_stats_meta(path, max_lines=50_000):
    """Basic EDA on metadata"""
    rows = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    print("\n=== Metadata dataset stats ===")
    print(f"Total rows sampled: {len(df):,}")
    if "category" in df.columns:
        print("Example categories:", df["category"].head(5).tolist())
    if "title" in df.columns:
        print("Example titles:", df["title"].head(5).tolist())

def main():
    # Inspect raw samples
    sample_jsonl(RAW_REV, n=3)
    sample_jsonl(RAW_META, n=3)

    # Field distributions
    field_distribution(RAW_REV)
    field_distribution(RAW_META)

    # Stats
    quick_stats_reviews(RAW_REV)
    quick_stats_meta(RAW_META)

if __name__ == "__main__":
    main()
