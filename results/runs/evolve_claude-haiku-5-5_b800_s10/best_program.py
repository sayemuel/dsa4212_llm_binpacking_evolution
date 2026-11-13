import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best fit backbone with exact-fit bonus, dead-zone penalty for unusable
    leftovers, windowed reuse reward, Gaussian bonus near s, tiny-gap penalty,
    a bonus for bins nearly full after placement, and fill-level consolidation."""
    cap = np.max(bins)
    r = bins - item
    x = r / cap                          # normalised leftover after placing
    s = item / cap                       # normalised item size
    exact = x < 1e-9

    # Dead-zone penalty for leftovers too small to be reused.
    dead_limit = np.minimum(s, 0.2)
    t = np.clip((dead_limit * 1.3 - x) / (0.6 * dead_limit + 1e-12), 0.0, 1.0)
    dead_pen = t * t * (3.0 - 2.0 * t)
    dead_pen = np.where(exact, 0.0, dead_pen)

    # Reward leftovers in a window that can host an item of size ~ s.
    lo = np.clip((x - 0.5 * s) / (0.5 * s + 1e-12), 0.0, 1.0)
    hi = np.clip((1.5 * s + 0.1 - x) / (0.1 + 1e-12), 0.0, 1.0)
    usable = np.where(exact, 0.0, lo * hi)

    # Gaussian bonus for leftovers close to s.
    near_s = np.exp(-((x - s) / (0.25 * s + 1e-12)) ** 2)
    near_s = np.where(exact, 0.0, near_s)

    # Penalty for tiny nonzero gaps (almost never fillable).
    frag = np.clip((0.015 - x) / 0.015, 0.0, 1.0)
    frag = np.where(exact, 0.0, frag)

    # Small bonus for bins that end up nearly full.
    nearly_full = np.clip((0.05 - x) / 0.05, 0.0, 1.0)
    nearly_full = np.where(exact, 0.0, nearly_full)

    fill_now = (cap - bins) / cap        # fill level before placing
    return (-x
            + 1.0 * exact
            - 0.4 * dead_pen
            + 0.05 * usable
            + 0.02 * near_s
            - 0.03 * frag
            + 0.02 * nearly_full
            + 0.1 * fill_now ** 3
            + 0.03 * fill_now ** 2
            + 0.01 * fill_now ** 4)
