# src/realtime_personalization/emb_store.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

def _df_to_map(df: pd.DataFrame, key_col: str) -> Dict[str, np.ndarray]:
    cols = [c for c in df.columns if c.startswith("e")]
    out: Dict[str, np.ndarray] = {}
    for r in df.itertuples(index=False):
        k = getattr(r, key_col)
        v = np.array([getattr(r, c) for c in cols], dtype=np.float32)
        out[str(k)] = v
    return out

def load_embs(user_emb_path: str, item_emb_path: str):
    U_df = pd.read_parquet(user_emb_path)
    V_df = pd.read_parquet(item_emb_path)
    U = _df_to_map(U_df, "user_id")
    V = _df_to_map(V_df, "item_id")
    dim = len([c for c in U_df.columns if c.startswith("e")])
    return U, V, dim

def user_vec(U: Dict[str, np.ndarray], user_id: str, dim: int) -> List[float]:
    v = U.get(str(user_id))
    if v is None:
        return [0.0]*dim
    return v.tolist()

def item_vec(V: Dict[str, np.ndarray], item_id: str, dim: int) -> List[float]:
    v = V.get(str(item_id))
    if v is None:
        return [0.0]*dim
    return v.tolist()
