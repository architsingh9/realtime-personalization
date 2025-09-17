# scripts/init_mlp_and_heads.py
import json
from pathlib import Path

from realtime_personalization.mlp import TinyMLP
from realtime_personalization.neurallinear import NeuralLinearModel

ART = Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)
DOCS = Path("docs")

EMB_DIM   = 32                     # from train_embeddings.py
WIDE_DIM  = 8
Z_DIM     = WIDE_DIM + EMB_DIM + EMB_DIM   # 72
HIDDEN    = 64
D         = HIDDEN + 1             # +1 for bias feature we’ll append to h

MLP_PATH   = ART / "mlp_weights.npz"
HEADS_PATH = ART / "bandit_heads.json"
ARMS_FILE  = DOCS / "arms.json"

def load_arms():
    with open(ARMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    arms = load_arms()
    print(f"Arms: {len(arms)} → {arms[:5]} ...")

    # MLP: keep if a warmed file already exists (train_mlp.py)
    if MLP_PATH.exists():
        print("ℹ️  Kept existing artifacts/mlp_weights.npz (not overwriting)")
    else:
        mlp = TinyMLP(in_dim=Z_DIM, hidden=HIDDEN, seed=42)
        mlp.save(str(MLP_PATH))
        print("✅ Wrote artifacts/mlp_weights.npz")

    # Initialize fresh Bayesian heads with dimension = HIDDEN + 1 (bias)
    nl = NeuralLinearModel.init_blank(arms=arms, d=D, lam=1e-3)
    nl.save_json(str(HEADS_PATH))
    print(f"✅ Wrote {HEADS_PATH} (d={D})")

if __name__ == "__main__":
    main()
