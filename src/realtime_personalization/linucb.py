# src/realtime_personalization/linucb.py
from __future__ import annotations
from typing import Dict
import os
import numpy as np

DEFAULT_ALPHA = float(os.getenv("ALPHA", "1.0"))   # exploration strength c in LinUCB
DEFAULT_LAMBDA = float(os.getenv("LAMBDA", "1e-3"))  # ridge prior

class LinUCBHead:
    def __init__(self, d: int, lam: float = DEFAULT_LAMBDA):
        self.d = int(d)
        self.A = lam * np.eye(self.d, dtype=np.float32)  # d x d
        self.b = np.zeros(self.d, dtype=np.float32)      # d
        self.A_inv = np.eye(self.d, dtype=np.float32)    # keep inverse cached
        self.theta = np.zeros(self.d, dtype=np.float32)

    def score_ucb(self, x: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
        # p = theta^T x + alpha * sqrt(x^T A^{-1} x)
        mean = float(self.theta @ x)
        var = float(x.T @ self.A_inv @ x)
        return mean + alpha * (var ** 0.5)

    def update(self, x: np.ndarray, r: float):
        # standard ridge update
        self.A += np.outer(x, x)
        self.b += r * x
        self.A_inv = np.linalg.inv(self.A)
        self.theta = self.A_inv @ self.b

class LinUCBModel:
    def __init__(self, heads: Dict[str, LinUCBHead], alpha: float = DEFAULT_ALPHA):
        self.heads = heads
        self.alpha = float(alpha)

    @classmethod
    def init_blank(cls, arms, d: int, lam: float = DEFAULT_LAMBDA, alpha: float = DEFAULT_ALPHA):
        heads = {a: LinUCBHead(d=d, lam=lam) for a in arms}
        return cls(heads=heads, alpha=alpha)

    def choose(self, h_by_arm: Dict[str, np.ndarray]) -> str:
        best_arm, best_score = None, -1e18
        for arm, h in h_by_arm.items():
            s = self.heads[arm].score_ucb(h, self.alpha)
            if s > best_score:
                best_arm, best_score = arm, s
        return best_arm

    def update(self, arm: str, h: np.ndarray, r: float):
        self.heads[arm].update(h, r)
