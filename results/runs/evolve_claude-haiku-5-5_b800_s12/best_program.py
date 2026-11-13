import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best fit core with exact-fit and near-fit bonuses, dead-gap, dead2 and
    sliver penalties, a usable-residual bonus, a large-leftover penalty, an
    item-scaled fullness term, and a fill-ratio term with a sharp tail."""
    bins = np.asarray(bins, dtype=float)
    C = float(np.max(bins))
    if C <= 0:
        C = 1.0
    left = bins - item
    g = np.clip(left / C, 0.0, None)

    exact = g <= 1e-7
    not_exact = ~exact

    score = -g
    score = score + 2.2 * exact
    score = score + 0.65 * np.exp(-g / 0.011) * not_exact

    # mid-small residuals are hard to refill later
    dead = not_exact * np.exp(-((g - 0.12) / 0.07) ** 2)
    score = score - 0.95 * dead

    # residuals just above the dead band
    dead2 = not_exact * np.exp(-((g - 0.2) / 0.04) ** 2)
    score = score - 0.14 * dead2

    # tiny slivers that almost nothing can refill
    sliver = not_exact * np.exp(-(g / 0.02) ** 2)
    score = score - 0.2 * sliver

    # residuals that can still host a moderate future item
    usable = not_exact * np.exp(-((g - 0.3) / 0.15) ** 2)
    score = score + 0.11 * usable

    # large leftovers waste space
    big_left = not_exact * np.clip((g - 0.4) / 0.6, 0.0, 1.0)
    score = score - 0.15 * big_left

    # consolidation: prefer fuller bins, more strongly for large items
    fullness = np.clip(bins / C, 0.0, 1.0)
    weight = 0.3 + 0.22 * min(item / C, 1.0)
    score = score + weight * fullness ** 2

    # fill-ratio bonus with a sharper tail for very tight fits
    fill_ratio = np.clip(item / np.maximum(bins, 1e-12), 0.0, 1.0)
    score = score + 0.11 * fill_ratio + 0.03 * fill_ratio ** 4

    return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
