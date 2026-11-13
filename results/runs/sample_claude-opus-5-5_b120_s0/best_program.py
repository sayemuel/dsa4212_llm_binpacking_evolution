import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    cap = float(np.max(bins))
    rem = bins - item
    r = rem / cap
    score = -r
    # bonus for near-perfect fit
    score = score + np.where(r < 0.02, 1.0, 0.0)
    # penalise awkward small leftovers that few items could fill
    small = (r >= 0.02) & (r < 0.5 * item / cap + 0.05)
    score = score - np.where(small, 0.15 * (1.0 - r), 0.0)
    # opening a new bin is the last resort
    score = np.where(bins >= cap, score - 1.0, score)
    return score
