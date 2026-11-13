import numpy as np

def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best fit with an exact-fit bonus, a smooth penalty on awkward
    mid-sized leftovers, a penalty on leftovers smaller than the item,
    a mild penalty on leftovers tiny relative to the item, and an
    earliest-bin tie-break."""
    bins = np.asarray(bins, dtype=float)
    cap = float(np.max(bins))
    leftover = bins - item
    rel = leftover / cap
    item_rel = item / cap

    # Base: best fit
    score = -leftover

    # Exact / near-exact fit bonus
    score = score + 0.25 * cap * np.exp(-np.clip(rel / 0.002, 0.0, 700.0))

    # Awkward middle band penalty (~4%-22% of capacity)
    band = _sig(80.0 * (rel - 0.04)) * (1.0 - _sig(80.0 * (rel - 0.22)))
    score = score - 0.55 * cap * band

    # Leftovers smaller than the item are hard to refill
    small_gap = _sig(60.0 * (item_rel - rel - 0.02))
    score = score - 0.13 * cap * small_gap

    # Mild penalty on leftovers tiny relative to the item (not exact fits)
    ratio = leftover / max(item, 1e-12)
    ratio_pen = _sig(20.0 * (0.5 - ratio))
    score = score - 0.02 * cap * ratio_pen * (rel > 1e-3)

    # Earliest-bin tie-break
    idx = np.arange(bins.shape[0], dtype=float)
    score = score - 1e-9 * cap * idx

    return score
