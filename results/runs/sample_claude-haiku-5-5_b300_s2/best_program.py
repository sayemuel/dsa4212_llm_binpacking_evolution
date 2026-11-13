import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best fit with penalties on awkward leftover gaps.

    Leftover fraction x = (remaining - item) / capacity.
    - Tiny leftovers (nearly full bins) get a bonus.
    - Mid-range leftovers (too big to waste, too small to host a typical
      item) get a penalty, steering items toward bins with either a
      near-perfect fit or a comfortably large remainder.
    - Empty bin (x close to 1) is ranked last, so new bins are opened only
      when necessary.
    """
    bins = np.asarray(bins, dtype=float)
    cap = float(np.max(bins))
    if cap <= 0.0:
        return np.zeros_like(bins)
    x = (bins - item) / cap
    score = -x
    awkward = (x > 0.02) & (x < 0.25)
    tiny = x <= 0.02
    score = score - 0.5 * awkward + 1.0 * tiny
    return score
