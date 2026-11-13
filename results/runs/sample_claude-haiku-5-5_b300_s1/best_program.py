import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best fit with a penalty for leaving awkward mid-sized gaps.

    Residual r = remaining space after placing the item. Tight fits (r near 0)
    and large leftovers (which can still take future items) are preferred;
    leftovers in the middle band are penalised since few items fit them.
    """
    bins = np.asarray(bins, dtype=float)
    cap = float(np.max(bins))
    if cap <= 0:
        return np.zeros_like(bins)
    r = (bins - item) / cap  # normalised residual, in [0, 1]

    score = -r
    # Penalise awkward mid-size gaps (too big to waste, too small to reuse easily).
    awkward = (r > 0.02) & (r < 0.3)
    score = score - 0.5 * awkward.astype(float)
    # Strongly reward near-perfect fits.
    score = score + 0.25 * (r <= 0.01)
    return score
