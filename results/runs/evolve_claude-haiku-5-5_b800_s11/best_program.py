import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best fit with quadratic dead-space and sliver penalties, a Gaussian
    reuse bonus, a mid-range leftover bonus, exact/near-fit bonuses, a
    size-scaled fullness preference, and a firmer reluctance to open bins."""
    bins = np.asarray(bins, dtype=float)
    cap = bins.max()
    gap = bins - item
    exact = gap <= 1e-9
    open_gap = ~exact
    near = open_gap & (gap <= 0.02 * cap)
    empty = bins >= cap - 1e-12
    scale = max(item, 1e-12)
    rel = np.clip(item / cap, 0.0, 1.0)

    # Base: best fit (tighter is better).
    score = -gap

    # Dead-space penalty: leftovers smaller than the item are hard to reuse.
    frac = np.clip(gap / scale, 0.0, 1.0)
    dead = open_gap & (gap < item)
    score = score - cap * dead * (0.6 - 0.3 * frac * frac)

    # Sliver penalty: tiny non-near leftovers are almost surely wasted.
    sliver = open_gap & (gap < 0.045 * cap) & (~near)
    score = score - 0.09 * cap * sliver.astype(float)

    # Reuse bonus: Gaussian bump centred at leftover = item.
    z = (gap - item) / (0.5 * scale)
    score = score + 0.05 * cap * np.exp(-0.5 * z * z) * open_gap

    # Mid-range leftovers can still host a typical item later.
    usable = open_gap & (gap >= 0.15 * cap) & (gap <= 0.5 * cap)
    score = score + 0.012 * cap * usable.astype(float)

    # Near-exact fits are nearly as good as exact fits.
    score = score + 1.0 * cap * near.astype(float)

    # Fullness preference, scaled by item size.
    fill = (cap - bins) / cap
    score = score + 0.03 * cap * fill * rel

    # Exact fits are strictly best.
    score = score + 2.0 * cap * exact.astype(float)

    # Opening an empty bin is a last resort; small items are more reluctant.
    score = score - (0.095 - 0.03 * rel) * cap * empty.astype(float)

    # Tiny tie-breaker favouring earlier bins.
    score = score - 1e-9 * np.arange(len(bins))
    return score
