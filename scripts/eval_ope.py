# scripts/eval_ope.py
from __future__ import annotations
import json, math
from pathlib import Path
import csv
import numpy as np

from realtime_personalization.emb_store import load_embs
from realtime_personalization.feature_joiner import make_rep
from realtime_personalization.mlp import TinyMLP
from realtime_personalization.neurallinear import NeuralLinearModel

ART = Path("artifacts")
DOCS = Path("docs")

LOG_PATH   = ART / "logged_uniform.csv"   # created by your uniform logger
MLP_PATH   = ART / "mlp_weights.npz"
HEADS_PATH = ART / "bandit_heads.json"
ARMS_FILE  = DOCS / "arms.json"
USER_EMB   = ART / "user_emb.parquet"
ITEM_EMB   = ART / "item_emb.parquet"

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def softmax(scores: list[float]) -> np.ndarray:
    a = np.asarray(scores, dtype=np.float64)
    a = a - np.max(a)
    ea = np.exp(a)
    return ea / np.sum(ea)

def read_log_csv(path: Path):
    """CSV reader without pandas. Returns dict of numpy arrays."""
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the uniform logger step first.")
    user_id, page, device, action = [], [], [], []
    reward, p_b = [], []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # expected columns (extra columns are fine):
        # user_id, item_id (gt), action, reward, p_b, page, device
        for row in r:
            user_id.append(str(row.get("user_id", "")))
            page.append(str(row.get("page", "")))
            device.append(str(row.get("device", "")))
            action.append(str(row.get("action", "")))
            # reward & behavior prob
            rw = row.get("reward", "0")
            pb = row.get("p_b", "")
            reward.append(float(rw))
            p_b.append(float(pb) if pb != "" else np.nan)
    return {
        "user_id": np.array(user_id, dtype=object),
        "page":    np.array(page, dtype=object),
        "device":  np.array(device, dtype=object),
        "action":  np.array(action, dtype=object),
        "reward":  np.array(reward, dtype=np.float64),
        "p_b":     np.array(p_b, dtype=np.float64),
        "n":       len(user_id),
    }

def main():
    # Load artifacts
    arms = json.load(open(ARMS_FILE, "r", encoding="utf-8"))
    U, V, d = load_embs(str(USER_EMB), str(ITEM_EMB))
    mlp = TinyMLP.load(str(MLP_PATH))
    model = NeuralLinearModel.load_json(str(HEADS_PATH))

    log = read_log_csv(LOG_PATH)
    n = log["n"]
    if n == 0:
        raise SystemExit(f"{LOG_PATH} is empty.")

    # Default behavior prob if missing = uniform
    default_pb = 1.0 / max(len(arms), 1)

    pi_probs = np.zeros(n, dtype=np.float64)  # policy prob of logged action
    q_hats   = np.zeros(n, dtype=np.float64)  # model’s reward estimate for logged action
    rewards  = log["reward"].astype(np.float64)
    p_b      = log["p_b"].astype(np.float64)

    # Compute per-row quantities
    for i in range(n):
        uid    = str(log["user_id"][i])
        page   = str(log["page"][i])
        device = str(log["device"][i])
        a_log  = str(log["action"][i])

        feats_base = {"visitCount": 0, "lastPage": "", "lastDevice": ""}

        # Build hidden reps for all arms
        h_by_arm = {}
        scores = []
        for a in arms:
            z = np.array(
                make_rep(feats_base, {"page": page, "device": device}, uid, a, U, V, d),
                dtype=np.float32,
            )
            h = mlp.forward(z)  # (64,)
            if h.shape[0] == 64:  # append bias to match d=65 heads
                h = np.concatenate([h, np.array([1.0], dtype=np.float32)], axis=0)
            h_by_arm[a] = h
            scores.append(float(model.heads[a].theta @ h))

        # policy π: softmax over linear scores
        probs = softmax(scores)
        pi_map = {arms[j]: float(probs[j]) for j in range(len(arms))}
        pi_probs[i] = pi_map.get(a_log, 0.0)

        # q_hat: model’s click prob estimate for the logged action
        if a_log in h_by_arm:
            s = float(model.heads[a_log].theta @ h_by_arm[a_log])
            q_hats[i] = sigmoid(s)
        else:
            q_hats[i] = 0.5  # fallback

        # behavior prob fallback if missing
        if not np.isfinite(p_b[i]) or p_b[i] <= 0.0:
            p_b[i] = default_pb

    # IPS and DR
    w = pi_probs / p_b
    ips_est = float(np.mean(w * rewards))
    dr_est  = float(np.mean(q_hats + w * (rewards - q_hats)))

    print(f"Loaded {n} rows from {LOG_PATH}")
    print(f"IPS estimate: {ips_est:.4f}")
    print(f"DR  estimate: {dr_est:.4f}")

if __name__ == "__main__":
    main()
