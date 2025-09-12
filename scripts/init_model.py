import json, os
from pathlib import Path

ARMS = Path("docs/arms.json")
OUT  = Path("artifacts/model.json")

# Feature dimension (must match what the Lambda uses)
# Example: [bias, log(visitCount), device_mobile, device_desktop, p_electronics, p_cameras, p_computers, p_other]
D = int(os.getenv("FEAT_DIM", "8"))

with open(ARMS) as f:
    arms = json.load(f)

model = {}
for a in arms:
    model[a] = {
        "A_inv": [[1.0 if i==j else 0.0 for j in range(D)] for i in range(D)],
        "theta": [0.0] * D
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(model, f)

print(f"✅ Wrote {OUT} with {len(arms)} arms and D={D}")
