import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best fit with a penalty on awkward (medium-sized) leftover gaps."""
    bins = np.asarray(bins, dtype=float)
    C = float(np.max(bins))              # an empty bin is always present
    leftover = bins - item               # space left after placing item
    score = -leftover                    # best-fit core: tighter is better
    # Medium leftovers are hard to reuse; very small ones are near-perfect fits.
    awkward = (leftover >= 0.05 * C) & (leftover <= 0.25 * C)
    score = score - 0.5 * C * awkward
    # Mild bonus for nearly perfect fits (almost no waste).
    score = score + 0.1 * C * (leftover < 0.01 * C)
    return score
