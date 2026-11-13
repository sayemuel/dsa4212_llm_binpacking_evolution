import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best fit with a penalty on awkward mid-sized leftover gaps.

    Leftover r = bin_remaining - item. Tight fits (r near 0) are rewarded as in
    best fit. Leftovers in a mid-range dead zone (too big to ignore, too small to
    host a typical item) are penalised, so the algorithm prefers either a nearly
    full bin or a bin that keeps enough room for future items.
    """
    bins = np.asarray(bins, dtype=float)
    cap = np.max(bins)                  # the empty bin always present -> full capacity
    r = bins - item                     # leftover after placing the item
    rn = r / cap                        # normalised leftover in [0, 1]

    # Base: best fit (smaller leftover is better).
    score = -rn

    # Dead zone penalty: leftovers roughly in (0.05, 0.25) of capacity are hard to reuse.
    dead = (rn > 0.05) & (rn < 0.25)
    score = score - 0.5 * dead

    # Mildly prefer bins that can accept a typical small item afterwards.
    score = score + 0.1 * (rn >= 0.25)

    return score
