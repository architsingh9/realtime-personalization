# src/realtime_personalization/neurallinear.py
from __future__ import annotations
import os, json
from typing import Dict
import numpy as np

# defaults controlled via env; safe values for offline ablations
DEFAULT_SIGMA = float(os.getenv("SIGMA", "0.05"))    # Thompson exploration scale
DEFAULT_LAMBDA = float(os.getenv("LAMBDA", "0.01"))  # ridge prior for A init

# ---------- numeric helpers ----------
def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

def _chol_with_dynamic_jitter(cov: np.ndarray, base_eps: float = 1e-8) -> np.ndarray:
    """
    Return lower-triangular 'chol-like' factor L such that (cov + jitter*I) ≈ L @ L.T.
    Jitter is chosen adaptively from spectrum/condition number.
    """
    cov = _symmetrize(cov)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        evals = np.linalg.eigvalsh(cov)
        min_eval = float(evals[0])
        jitter = 0.0
        if min_eval < 0:
            jitter = (-min_eval) + base_eps
        max_eval = float(evals[-1])
        if max_eval > 0 and min_eval > 0:
            cond = max_eval / min_eval
            if cond > 1e8:
                jitter = max(jitter, base_eps * cond)
        cov2 = cov + jitter * np.eye(cov.shape[0], dtype=cov.dtype)
        try:
            return np.linalg.cholesky(cov2)
        except np.linalg.LinAlgError:
            # final fallback: PSD square-root via SVD
            U, S, _ = np.linalg.svd(_symmetrize(cov2), full_matrices=False)
            S = np.clip(S, 0.0, None)
            return (U * np.sqrt(S)) @ U.T  # symmetric factor

# ---------- Bayesian head ----------
class BayesianHead:
    def __init__(self, d: int, lam: float = DEFAULT_LAMBDA):
        self.d = int(d)
        self.A = lam * np.eye(self.d, dtype=np.float32)   # d x d
        self.b = np.zeros(self.d, dtype=np.float32)       # d
        self.theta = np.zeros(self.d, dtype=np.float32)   # d
        self.A_inv = np.eye(self.d, dtype=np.float32)     # d x d

    def update(self, x: np.ndarray, r: float):
        # A ← A + x x^T ; b ← b + r x ; then recompute inverse & theta
        xxT = np.outer(x, x)
        self.A += xxT
        self.b += r * x
        self.A_inv = np.linalg.inv(self.A)
        self.theta = self.A_inv @ self.b

    def sample_theta(self, sigma: float = DEFAULT_SIGMA, eps: float = 1e-8) -> np.ndarray:
        # posterior covariance: sigma^2 * A_inv
        cov = (sigma ** 2) * self.A_inv.astype(np.float64)
        L = _chol_with_dynamic_jitter(cov, base_eps=eps)
        z = np.random.normal(size=(self.d,)).astype(np.float64)
        return (self.theta.astype(np.float64) + L @ z).astype(np.float32)

    # ---- IO helpers for JSON persistence ----
    def to_dict(self) -> Dict:
        # store the minimum needed (A_inv, theta); A/b are re-derived/update-only
        return {
            "A_inv": self.A_inv.tolist(),
            "theta": self.theta.tolist(),
        }

    @classmethod
    def from_dict(cls, obj: Dict, d: int, lam: float = DEFAULT_LAMBDA) -> "BayesianHead":
        h = cls(d=d, lam=lam)
        A_inv = np.array(obj.get("A_inv", np.eye(d)), dtype=np.float32)
        theta = np.array(obj.get("theta", np.zeros(d)), dtype=np.float32)
        # ensure shapes match
        assert A_inv.shape == (d, d), f"A_inv shape {A_inv.shape} != {(d,d)}"
        assert theta.shape == (d,),   f"theta shape {theta.shape} != {(d,)}"
        h.A_inv = A_inv
        # reconstruct A from A_inv for online updates
        try:
            h.A = np.linalg.inv(h.A_inv)
        except np.linalg.LinAlgError:
            # if inversion is unstable, re-init A with ridge prior
            h.A = DEFAULT_LAMBDA * np.eye(d, dtype=np.float32)
            h.A_inv = np.linalg.inv(h.A)
        h.theta = theta
        h.b = h.A @ h.theta  # consistent with theta = A_inv b
        return h

# ---------- Model over arms ----------
class NeuralLinearModel:
    def __init__(self, heads: Dict[str, BayesianHead]):
        self.heads = heads  # arm_id -> head

    # selection
    def choose_thompson(self, h_by_arm: Dict[str, np.ndarray], sigma: float = DEFAULT_SIGMA) -> str:
        best_arm, best_score = None, -1e18
        for arm, h in h_by_arm.items():
            theta_tilde = self.heads[arm].sample_theta(sigma=sigma)
            s = float(theta_tilde @ h)  # assume h already contains bias if used
            if s > best_score:
                best_arm, best_score = arm, s
        return best_arm

    def choose_greedy(self, h_by_arm: Dict[str, np.ndarray]) -> str:
        best_arm, best_score = None, -1e18
        for arm, h in h_by_arm.items():
            s = float(self.heads[arm].theta @ h)
            if s > best_score:
                best_arm, best_score = arm, s
        return best_arm

    def update(self, arm: str, h: np.ndarray, r: float):
        self.heads[arm].update(h, r)

    # IO
    @classmethod
    def load_json(cls, path: str) -> "NeuralLinearModel":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # support two formats:
        # 1) {"d": 65, "heads": {"ARMID": {"A_inv": [...], "theta":[...]}, ...}}
        # 2) {"ARMID": {...}, ...}  (no top-level "d": infer from first theta)
        if "heads" in data:
            d = int(data["d"])
            raw_heads = data["heads"]
        else:
            # old flat format
            any_arm = next(iter(data.values()))
            d = len(any_arm["theta"])
            raw_heads = data
        heads = {}
        for arm, obj in raw_heads.items():
            heads[arm] = BayesianHead.from_dict(obj, d=d, lam=DEFAULT_LAMBDA)
        return cls(heads=heads)

    def save_json(self, path: str):
        # persist in the structured format with explicit d
        # all heads must have same d
        any_head = next(iter(self.heads.values()))
        d = int(any_head.d)
        out = {"d": d, "heads": {arm: head.to_dict() for arm, head in self.heads.items()}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    @classmethod
    def init_blank(cls, arms: list[str], d: int, lam: float = DEFAULT_LAMBDA) -> "NeuralLinearModel":
        """
        Create a fresh model with one BayesianHead per arm, each of size d.
        Use d = hidden_dim + 1 if you append a bias feature to h.
        """
        heads = {arm: BayesianHead(d=d, lam=lam) for arm in arms}
        return cls(heads=heads)
