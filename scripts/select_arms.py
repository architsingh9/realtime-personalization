import pandas as pd, json, os
from pathlib import Path

PARQ = Path("data/processed/events_base.parquet")
OUT  = Path("docs/arms.json")

TOP_N = int(os.getenv("TOP_N", "8"))  # choose 5–10 arms

df = pd.read_parquet(PARQ, columns=["itemId"])
top_items = (
    df.value_counts("itemId")
      .head(TOP_N)
      .index.tolist()
)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(top_items, f, indent=2)

print(f"✅ Saved top {TOP_N} arms to {OUT}:\n{top_items}")
