# src/realtime_personalization/mlp.py
from __future__ import annotations
import os
import numpy as np

def xavier_init(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    W = rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)
    b = np.zeros((fan_out,), dtype=np.float32)
    return W, b

class TinyMLP:
    def __init__(self, in_dim: int, hidden: int = 64, seed: int = 42):
        self.in_dim = in_dim
        self.hidden = hidden
        self.rng = np.random.default_rng(seed)
        self.W1, self.b1 = xavier_init(in_dim, hidden, self.rng)
        self.W2, self.b2 = xavier_init(hidden, hidden, self.rng)  # optional second layer (keeps same dim)

    @staticmethod
    def relu(x): return np.maximum(0.0, x)

    def forward(self, z: np.ndarray) -> np.ndarray:
        # z: (..., in_dim)
        h1 = self.relu(z @ self.W1 + self.b1)
        h2 = self.relu(h1 @ self.W2 + self.b2)
        return h2  # shape (..., hidden)

    def save(self, path: str):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 in_dim=self.in_dim, hidden=self.hidden)

    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        mlp = cls(int(data["in_dim"]), int(data["hidden"]))
        mlp.W1 = data["W1"]; mlp.b1 = data["b1"]
        mlp.W2 = data["W2"]; mlp.b2 = data["b2"]
        return mlp
