import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best fit with near-perfect-fit bonus and penalty for awkward mid-size gaps."""
    bins = np.asarray(bins, dtype=float)
    cap = float(np.max(bins))  # an empty bin is always present, so this is the capacity
    if cap <= 0:
        cap = 1.0
    r = (bins - item) / cap  # normalised residual space after placing the item

    score = -r
    # Near-perfect fits: almost no waste, strongly preferred.
    score = score + 1.0 * (r < 1e-3)
    # Awkward residuals (too big to waste, too small to hold a typical item): discouraged.
    awkward = (r >= 0.05) & (r < 0.25)
    score = score - 0.5 * awkward
    return score
