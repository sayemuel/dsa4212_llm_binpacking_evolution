import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best Fit core with an exact-fit bonus, graded penalties on
    hard-to-fill gaps (0.08 to 0.40 of capacity), a sliver penalty,
    a half-bin bonus, and a tight near-full bonus."""
    bins = np.asarray(bins, dtype=float)
    cap = np.max(bins)  # an empty bin is always present
    rel = (bins - item) / cap
    score = -rel
    # sharp bonus for (near-)perfect fits
    score = score + 0.40 * np.exp(-(rel / 0.006) ** 2)
    # graded penalties on gaps that are hard to fill later
    score = score - 0.50 * np.exp(-((rel - 0.10) / 0.063) ** 2)
    score = score - 0.12 * np.exp(-((rel - 0.15) / 0.045) ** 2)
    score = score - 0.10 * np.exp(-((rel - 0.20) / 0.040) ** 2)
    score = score - 0.15 * np.exp(-((rel - 0.25) / 0.085) ** 2)
    score = score - 0.05 * np.exp(-((rel - 0.35) / 0.065) ** 2)
    # small extra penalty on gaps around 0.30 of capacity
    score = score - 0.03 * np.exp(-((rel - 0.30) / 0.030) ** 2)
    # small penalty on gaps around 0.40 of capacity
    score = score - 0.015 * np.exp(-((rel - 0.40) / 0.040) ** 2)
    # small penalty on large leftover gaps
    score = score - 0.02 * np.exp(-((rel - 0.45) / 0.10) ** 2)
    # sliver penalty
    score = score - 0.035 * np.exp(-((rel - 0.05) / 0.025) ** 2)
    # mild bonus for gaps around half a bin
    score = score + 0.02 * np.exp(-((rel - 0.5) / 0.15) ** 2)
    # tight preference for bins that end up nearly full
    score = score + 0.05 * np.exp(-(rel / 0.015) ** 2)
    return score
