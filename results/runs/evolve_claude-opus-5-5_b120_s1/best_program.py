import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    cap = bins.max()
    r = (bins - item).astype(float)
    score = -r
    frac = r / cap
    # exact/near-exact fits are great
    score = score + np.where(frac < 1e-9, 0.05 * cap, 0.0)
    # tiny leftover slivers are mildly penalised
    sliver = (frac > 1e-9) & (frac < 0.02)
    score = score - np.where(sliver, 0.015 * cap, 0.0)
    # avoid opening new bin unless necessary (keep empty bin last)
    score = score - np.where(bins >= cap, 0.5 * cap, 0.0)
    return score
