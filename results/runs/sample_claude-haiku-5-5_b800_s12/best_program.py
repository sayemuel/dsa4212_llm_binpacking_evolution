import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best fit with a bonus for near-exact fits and a penalty for awkward leftover gaps."""
    bins = np.asarray(bins, dtype=float)
    cap = np.max(bins)                  # an empty bin is always present
    r = bins - item                     # leftover space after placing item
    score = -r                          # best-fit base: smaller leftover is better
    # Strong bonus for (near-)perfect fits: closes a bin with no waste.
    score = score + 2.0 * cap * (r < 1e-9)
    # Mild penalty for leftovers in the mid band that are hard to reuse.
    awkward = (r > 0.05 * cap) & (r < 0.2 * cap)
    score = score - 0.5 * cap * awkward
    return score
