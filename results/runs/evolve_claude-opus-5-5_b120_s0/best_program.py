import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    cap = bins.max()
    r = (bins - item).astype(float)
    score = -r
    # near-perfect fits are always best
    score += np.where(r <= 0.002 * cap, cap, 0.0)
    # penalize leaving awkward small gaps (smaller than typical items) in used bins
    used = bins < cap
    awkward = used & (r > 0.002 * cap) & (r < 0.5 * min(item, 0.2 * cap))
    score -= np.where(awkward, 0.05 * cap, 0.0)
    return score
