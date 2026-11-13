import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    cap = float(np.max(bins))
    if cap <= 0:
        return np.zeros_like(bins, dtype=float)
    scale = 100.0 / cap
    b = bins.astype(float) * scale
    it = max(float(item) * scale, 1e-6)

    # FunSearch-style base score
    score = (b - b.max()) ** 2 / it
    score += b ** 2 / it ** 2
    score += b ** 2 / it ** 3
    mask = b > it + 1e-9
    score[mask] = -score[mask]
    score[1:] -= score[:-1].copy()
    score = np.nan_to_num(score, nan=-1e15, posinf=1e15, neginf=-1e15)
    score = np.clip(score, -1e15, 1e15)

    # Near-perfect fits take precedence (tightest first, earlier bin on ties)
    r = b - it
    thr = min(2.0, max(1.2, 0.04 * it))
    near = (r <= thr) & (b < 100.0 - 1e-9)
    if np.any(near):
        n = b.shape[0]
        out = np.full(b.shape, -1e17, dtype=float)
        idx = np.arange(n, dtype=float)
        out[near] = 1e17 - r[near] * 1e6 - idx[near] * 1e-3
        return out
    return score
