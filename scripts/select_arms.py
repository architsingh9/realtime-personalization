# scripts/select_arms.py
import os, json, argparse
from pathlib import Path
import pandas as pd

PARQ = Path("data/processed/events_base.parquet")
OUT  = Path("docs/arms.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=None,
                        help="Number of top items to keep as arms.")
    args = parser.parse_args()

    # precedence: CLI --top_k > env TOP_N > default 8
    top_n = args.top_k if args.top_k is not None else int(os.getenv("TOP_N", "8"))

    df = pd.read_parquet(PARQ, columns=["itemId"])
    top_items = (
        df.value_counts("itemId")
          .head(top_n)
          .index.astype(str).tolist()
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(top_items, f, indent=2)

    print(f"✅ Saved top {top_n} arms to {OUT}:\n{top_items[:8]}{' ...' if len(top_items)>8 else ''}")

if __name__ == "__main__":
    main()
