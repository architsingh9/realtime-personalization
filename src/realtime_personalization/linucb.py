from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math

@dataclass
class ArmParams:
    # Stored terms
    A_inv: List[List[float]]          # d x d
    theta: List[float]                # d
    # Lazily added when training (if not present in JSON)
    A: List[List[float]] = field(default_factory=list)  # d x d
    b: List[float] = field(default_factory=list)        # d

@dataclass
class LinUCBModel:
    d: int
    arms: Dict[str, ArmParams]  # itemId -> params

    @staticmethod
    def identity(d: int) -> List[List[float]]:
        return [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]

    @classmethod
    def from_json(cls, data: Dict) -> "LinUCBModel":
        # Infer d from first arm's theta length
        any_arm = next(iter(data.values()))
        d = len(any_arm["theta"])
        arms = {}
        for arm, params in data.items():
            A_inv = params.get("A_inv") or cls.identity(d)
            theta = params.get("theta") or [0.0] * d
            A = params.get("A", [])
            b = params.get("b", [])
            arms[arm] = ArmParams(A_inv=A_inv, theta=theta, A=A, b=b)
        return cls(d=d, arms=arms)

    def to_json(self) -> Dict:
        out = {}
        for arm, p in self.arms.items():
            out[arm] = {
                "A_inv": p.A_inv,
                "theta": p.theta,
            }
            # persist training terms if available
            if p.A and p.b:
                out[arm]["A"] = p.A
                out[arm]["b"] = p.b
        return out

    # --------- Linear algebra helpers (pure Python; d=8 is tiny) ---------
    @staticmethod
    def dot(a: List[float], b: List[float]) -> float:
        return sum(x*y for x, y in zip(a, b))

    @staticmethod
    def matvec(M: List[List[float]], v: List[float]) -> List[float]:
        return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

    @staticmethod
    def outer(u: List[float], v: List[float]) -> List[List[float]]:
        return [[ui*vj for vj in v] for ui in u]

    @staticmethod
    def matadd(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def vecadd(a: List[float], b: List[float]) -> List[float]:
        return [ai + bi for ai, bi in zip(a, b)]

    @staticmethod
    def inv_2x2(M: List[List[float]]) -> List[List[float]]:
        # Not used; kept for illustration. We'll use generic Gauss-Jordan below.

        det = M[0][0]*M[1][1] - M[0][1]*M[1][0]
        assert abs(det) > 1e-12, "Singular"
        inv = [[ M[1][1]/det, -M[0][1]/det],
               [-M[1][0]/det,  M[0][0]/det]]
        return inv

    @staticmethod
    def invert(M: List[List[float]]) -> List[List[float]]:
        # Gauss-Jordan for small d (d<=32 is fine)
        n = len(M)
        A = [row[:] for row in M]
        I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for col in range(n):
            # pivot
            pivot = col
            for r in range(col, n):
                if abs(A[r][col]) > abs(A[pivot][col]):
                    pivot = r
            A[col], A[pivot] = A[pivot], A[col]
            I[col], I[pivot] = I[pivot], I[col]
            piv = A[col][col]
            assert abs(piv) > 1e-12, "Singular"
            # normalize
            for j in range(n):
                A[col][j] /= piv
                I[col][j] /= piv
            # eliminate
            for r in range(n):
                if r == col: continue
                factor = A[r][col]
                for j in range(n):
                    A[r][j] -= factor * A[col][j]
                    I[r][j] -= factor * I[col][j]
        return I

    # --------- Scoring & updates ----------
    def score(self, arm: str, x: List[float], alpha: float) -> float:
        p = self.arms[arm]
        exploit = self.dot(p.theta, x)
        Ax = self.matvec(p.A_inv, x)
        explore = math.sqrt(max(0.0, self.dot(x, Ax))) * alpha
        return exploit + explore

    def choose(self, x: List[float], alpha: float) -> Tuple[str, float]:
        best_arm, best_score = None, -1e18
        for arm in self.arms:
            s = self.score(arm, x, alpha)
            if s > best_score:
                best_arm, best_score = arm, s
        return best_arm, best_score

    def ensure_training_terms(self, arm: str):
        p = self.arms[arm]
        if not p.A:
            p.A = self.invert(p.A_inv)  # A_inv given by seeder -> reconstruct A
        if not p.b:
            p.b = [0.0] * self.d

    def update(self, arm: str, x: List[float], reward: float, ridge_lambda: float = 0.0):
        """A <- A + x x^T ; b <- b + r x ; then A_inv, theta updated."""
        p = self.arms[arm]
        self.ensure_training_terms(arm)

        # A += x x^T (+ ridge correction only at init time if wanted)
        xxT = self.outer(x, x)
        p.A = self.matadd(p.A, xxT)
        # b += r x
        p.b = self.vecadd(p.b, [reward * xi for xi in x])

        # Recompute inverse and theta
        p.A_inv = self.invert(p.A)
        p.theta = self.matvec(p.A_inv, p.b)

    # --------- IO helpers ----------
    @classmethod
    def load_json_file(cls, path: str) -> "LinUCBModel":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json(data)

    def save_json_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)
