import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best fit with a sharper exact-fit reward, smooth penalties for
    leftovers that cannot host another item of this size (slivers and
    near-misses), a penalty for leaving roughly half a bin, a fill
    tie-breaker, and a new-bin discouragement."""
    bins = np.asarray(bins, dtype=float)
    cap = float(np.max(bins))  # an empty bin is always present
    l = bins - item
    x = l / cap
    s = item / cap
    w = max(0.05, min(0.15, s))
    eps = 1e-9

    def sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    score = -x
    score = score + 0.8 * np.exp(-x / 0.008)                        # sharper exact-fit reward
    score = score - 0.3 * np.exp(-((x - 0.5 * w) / w) ** 2)         # sliver penalty
    score = score - 0.06 * sig((s - x) / (0.1 * s + eps))           # leftover too small for same-size item
    score = score - 0.03 * sig((0.5 * s - x) / (0.1 * s + eps))     # tiny leftover
    score = score - 0.04 * sig((x - 0.5) / 0.02)                    # half-bin leftover
    score = score + 0.02 * sig((x - s) / (0.1 * s + eps)) * sig((2.0 * s - x) / (0.1 * s + eps))
    score = score + 0.01 * (1.0 - bins / cap)                       # fuller bins win ties
    score = score - 0.06 * (bins >= cap - 1e-12)                    # discourage opening a new bin
    return score
