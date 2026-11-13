import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best Fit base with an exact-fit bonus, a soft dead-space penalty for
    leftovers smaller than the item, a mild penalty for awkward mid-sized
    leftovers, and empty bins as a last resort."""
    bins = np.asarray(bins, dtype=float)
    cap = float(np.max(bins))  # an empty bin is always present -> capacity
    if cap <= 0 or item <= 0:
        return np.zeros_like(bins)

    u = (bins - item) / cap      # normalised leftover after placing (>= 0)
    s = item / cap               # normalised item size

    score = -u                   # Best Fit ordering

    exact = u <= 1e-9
    score = score + 0.8 * exact  # exact fits dominate

    # dead space: small positive leftover that cannot host a typical item
    dead = (~exact) & (u < 0.5 * s)
    score = score - 0.3 * s * dead

    # mild penalty on mid-sized leftovers that are awkward to fill later
    middle = (~exact) & (u > 0.05) & (u < 0.2)
    score = score - 0.05 * middle

    # untouched empty bins are a last resort
    empty = bins >= cap - 1e-12
    score = score - 0.5 * empty

    # tiny tie-break toward fuller bins
    score = score + 1e-6 * (1.0 - u)
    return score
