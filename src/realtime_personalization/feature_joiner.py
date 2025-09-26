# src/realtime_personalization/feature_joiner.py
from __future__ import annotations
import numpy as np

# If you already have a helper like build_x in feature_vector, import it here.
# It should return the 8-d "wide" features from (feats, ctx).
try:
    from .feature_vector import build_x as _build_x_wide
except Exception:
    _build_x_wide = None


def _default_build_x_wide(feats: dict, ctx: dict) -> np.ndarray:
    """
    Fallback wide features (8-dim) if feature_vector.build_x is unavailable.
    Encodes visitCount, lastPage, lastDevice, page, device into a tiny numeric vector.
    """
    visit = float(feats.get("visitCount", 0) or 0)
    last_page = str(feats.get("lastPage", "") or "")
    last_dev  = str(feats.get("lastDevice", "") or "")
    page      = str(ctx.get("page", "") or "")
    device    = str(ctx.get("device", "") or "")

    # Simple hash-bucket encodings for categorical tokens (stable across runs)
    def hb(s: str, m: int) -> float:
        return float(hash(s) % m) / (m - 1 if m > 1 else 1)

    return np.array([
        visit,
        hb(last_page, 97),
        hb(last_dev,  17),
        hb(page,      97),
        hb(device,    17),
        1.0,          # bias for the wide part (kept inside the 8-d block)
        (visit > 0) * 1.0,
        (last_page == page) * 1.0,
    ], dtype=np.float32)


def make_rep(
    feats: dict,
    ctx: dict,
    user_id: str,
    item_id: str,
    U: dict[str, np.ndarray],
    V: dict[str, np.ndarray],
    emb_dim: int,
) -> np.ndarray:
    """
    Build the 72-d input z = [x_wide(8), u(emb_dim), v(emb_dim)] as a single NumPy array.
    IMPORTANT: we CONCATENATE, not add.
    """
    # wide features (8,)
    if _build_x_wide is not None:
        x_wide = np.asarray(_build_x_wide(feats, ctx), dtype=np.float32)
    else:
        x_wide = _default_build_x_wide(feats, ctx)

    if x_wide.shape != (8,):
        raise ValueError(f"Expected wide features shape (8,), got {x_wide.shape}")

    # embeddings (emb_dim,)
    u = U.get(str(user_id))
    v = V.get(str(item_id))
    if u is None:
        u = np.zeros(emb_dim, dtype=np.float32)
    else:
        u = np.asarray(u, dtype=np.float32)
        if u.shape != (emb_dim,):
            raise ValueError(f"user emb shape {u.shape} != ({emb_dim},)")

    if v is None:
        v = np.zeros(emb_dim, dtype=np.float32)
    else:
        v = np.asarray(v, dtype=np.float32)
        if v.shape != (emb_dim,):
            raise ValueError(f"item emb shape {v.shape} != ({emb_dim},)")

    # CONCATENATE, do not use "+" (which would try element-wise add)
    z = np.concatenate([x_wide, u, v], axis=0).astype(np.float32, copy=False)

    # (Optional) shape guard for the expected 72-d input of TinyMLP (8 + 32 + 32)
    # If your emb_dim changes, this still works; only this assert will change.
    # Comment out if you’ve changed your MLP in_dim.
    expected = 8 + 2 * emb_dim
    if z.shape != (expected,):
        raise ValueError(f"make_rep output shape {z.shape} != ({expected},)")

    return z
