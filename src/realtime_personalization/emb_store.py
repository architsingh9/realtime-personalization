# src/realtime_personalization/emb_store.py
# Pure-PyArrow loader to avoid Pandas/Arrow ABI issues entirely.
import numpy as np

def _load_emb_table(path: str, id_candidates=("user_id", "item_id", "id")):
    """
    Read a Parquet table with PyArrow and return (ids:list[str], X:float32[n,d]).
    Assumes exactly one id column + D numeric embedding columns (e0..e{d-1} or similar).
    """
    import pyarrow.parquet as pq
    import pyarrow as pa

    tbl = pq.read_table(path)
    names = list(tbl.column_names)

    # Pick id column
    id_col = next((c for c in id_candidates if c in names), None) or names[0]

    # Candidate embedding columns = all non-id columns
    emb_cols = [c for c in names if c != id_col]

    # Prefer ordering by suffix integer if columns look like e0, e1, ...
    def _emb_sort_key(c: str):
        if c.startswith("e"):
            try:
                return (0, int(c[1:]))
            except Exception:
                return (0, 10**9)
        return (1, c)

    emb_cols.sort(key=_emb_sort_key)

    # Keep only numeric columns (defensive)
    schema = tbl.schema
    numeric_cols = []
    for c in emb_cols:
        t = schema.field(c).type
        if pa.types.is_integer(t) or pa.types.is_floating(t):
            numeric_cols.append(c)
    emb_cols = numeric_cols
    if len(emb_cols) == 0:
        raise ValueError(f"No numeric embedding columns found in {path} (id={id_col}).")

    # Materialize to NumPy (no pandas involved)
    cols_np = []
    for c in emb_cols:
        # zero_copy_only=False allows conversion for non-contiguous buffers
        arr = tbl[c].to_numpy(zero_copy_only=False)
        # Ensure 1-D
        arr = np.asarray(arr).reshape(-1)
        cols_np.append(arr)

    X = np.column_stack(cols_np).astype(np.float32, copy=False)

    # IDs as strings
    ids_py = [str(x) for x in tbl[id_col].to_pylist()]

    return ids_py, X

def load_embs(user_emb_path: str, item_emb_path: str):
    """
    Returns:
      U_map: dict[user_id -> float32[d]]
      V_map: dict[item_id -> float32[d]]
      d: int
    """
    u_ids, U = _load_emb_table(user_emb_path, id_candidates=("user_id", "id"))
    i_ids, V = _load_emb_table(item_emb_path, id_candidates=("item_id", "id"))

    if U.shape[1] != V.shape[1]:
        raise ValueError(f"user/item embedding dim mismatch: {U.shape[1]} vs {V.shape[1]}")
    d = int(U.shape[1])

    U_map = {u_ids[i]: U[i] for i in range(len(u_ids))}
    V_map = {i_ids[i]: V[i] for i in range(len(i_ids))}
    return U_map, V_map, d

# --------- Back-compat helpers used by feature_joiner ---------
def user_vec(U_map, user_id, d):
    v = U_map.get(str(user_id))
    return np.asarray(v, dtype=np.float32) if v is not None else np.zeros(d, dtype=np.float32)

def item_vec(V_map, item_id, d):
    v = V_map.get(str(item_id))
    return np.asarray(v, dtype=np.float32) if v is not None else np.zeros(d, dtype=np.float32)
