# scripts/select_arms.py
import os, json
from pathlib import Path
from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

PARQ = Path("data/processed/events_base.parquet")
DOCS = Path("docs"); DOCS.mkdir(parents=True, exist_ok=True)
ARMS_FILE = DOCS / "arms.json"

def main():
    top_n = int(os.getenv("TOP_N", "256"))

    table = pq.read_table(PARQ)

    # normalize column name
    cols = {c.lower(): c for c in table.column_names}
    if "itemid" in cols:
        colname = cols["itemid"]
    elif "item_id" in cols:
        colname = cols["item_id"]
    else:
        raise SystemExit("events_base.parquet missing 'itemId'/'item_id' column")

    # cast to string to avoid mixed types, then to Python list
    item_col = pc.cast(table[colname], pa.string())
    item_ids = item_col.to_pylist()

    counts = Counter(item_ids)
    arms = [k for k, _ in counts.most_common(top_n)]

    with open(ARMS_FILE, "w", encoding="utf-8") as f:
        json.dump(arms, f, indent=2)

    print(f"✅ Saved top {top_n} arms to {ARMS_FILE}:")
    print(arms[:8] if len(arms) >= 8 else arms)

if __name__ == "__main__":
    main()
