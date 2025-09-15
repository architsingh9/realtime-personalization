from __future__ import annotations
import math
from typing import Dict, List, Tuple

# ===== FEATURE CONTRACT (D = 8) =====
# x = [
#   0) bias = 1.0,
#   1) log1p(visitCount),
#   2) isMobile,
#   3) isDesktop,
#   4) p_electronics,
#   5) p_cameras,
#   6) p_computers,
#   7) p_other,
# ]

PAGE_BUCKETS: Tuple[str, ...] = ("electronics", "cameras", "computers", "other")

def bucket_page(page: str) -> List[float]:
    p = (page or "").lower()
    onehot = [0.0, 0.0, 0.0, 0.0]
    if "camera" in p:
        onehot[1] = 1.0
    elif "computer" in p or "laptop" in p or "pc" in p:
        onehot[2] = 1.0
    elif "electronic" in p or "audio" in p or "tv" in p:
        onehot[0] = 1.0
    else:
        onehot[3] = 1.0
    return onehot

def build_x(user_features: Dict[str, object], context: Dict[str, object]) -> List[float]:
    """Return the 8-dim feature vector used everywhere."""
    visit_count = 0
    last_device = ""
    # User features may not exist yet (cold-start); be robust
    if user_features:
        visit_count = int(user_features.get("visitCount", 0))
        last_device = str(user_features.get("lastDevice", "")) or ""

    # Context can override
    device = (context.get("device") or last_device or "").lower()
    page = str(context.get("page") or user_features.get("lastPage", "") or "")

    x = [0.0] * 8
    x[0] = 1.0  # bias
    x[1] = math.log1p(max(0, visit_count))
    x[2] = 1.0 if device == "mobile" else 0.0
    x[3] = 1.0 if device == "desktop" else 0.0
    x[4:8] = bucket_page(page)
    return x
