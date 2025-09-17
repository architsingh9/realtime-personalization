# src/realtime_personalization/feature_joiner.py
from __future__ import annotations
from typing import Dict, List, Tuple

from .feature_vector import build_x
from .emb_store import user_vec, item_vec

def make_rep(
    user_features: Dict[str, object],
    context: Dict[str, object],
    user_id: str,
    cand_item_id: str,
    U, V, emb_dim: int
) -> List[float]:
    x_wide = build_x(user_features, context)          # 8-d
    u = user_vec(U, user_id, emb_dim)                 # d_u
    v = item_vec(V, cand_item_id, emb_dim)            # d_v
    return x_wide + u + v
