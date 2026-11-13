"""Online 1-D bin packing: instance generation, simulator, baselines, bounds.

Protocol (follows FunSearch, Romera-Paredes et al. 2024):
  * Items arrive one at a time in a fixed order; the packer cannot reorder them.
  * The bin state is an array of remaining capacities with one slot per item
    (so an empty bin is always available). Bins that cannot fit the item are
    masked out *by the simulator*; the heuristic only sees feasible bins.
  * The heuristic returns one priority score per feasible bin; the item goes
    into the arg-max bin (ties -> lowest index, i.e. oldest bin).
  * Feasibility and scoring are computed here, never by evolved code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

PriorityFn = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Instance:
    name: str
    items: np.ndarray  # integer sizes, arrival order
    capacity: int


# ----------------------------------------------------------------------------
# Instance generators (all seeded -> fully reproducible)
# ----------------------------------------------------------------------------

def weibull_instance(n: int, seed: int, capacity: int = 100, shape: float = 3.0,
                     scale: float = 45.0) -> Instance:
    """FunSearch 'Weibull' family: sizes ~ Weibull(k=3, lambda=45), rounded and
    clipped to [1, capacity]; capacity 100."""
    rng = np.random.default_rng(seed)
    items = np.clip(np.round(rng.weibull(shape, n) * scale), 1, capacity).astype(int)
    return Instance(f"weibull_n{n}_c{capacity}_s{seed}", items, capacity)


def uniform_instance(n: int, seed: int, low: int = 20, high: int = 100,
                     capacity: int = 150) -> Instance:
    """OR-Library binpack1-4 style (Falkenauer 1996): sizes ~ U{20..100}, C=150."""
    rng = np.random.default_rng(seed)
    items = rng.integers(low, high + 1, n)
    return Instance(f"uniform{low}-{high}_n{n}_c{capacity}_s{seed}", items, capacity)


def triplet_instance(n_triplets: int, seed: int, capacity: int = 1000) -> Instance:
    """Falkenauer-style 't' instances: every bin of the optimum holds exactly
    three items summing to capacity, so OPT = n_triplets exactly. Items are
    shuffled. Hard for greedy rules because waste is never free."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n_triplets):
        q = capacity // 4
        a = int(rng.integers(q + 1, capacity // 2 - 4))      # (C/4, C/2)
        b = int(rng.integers(q + 1, (capacity - a) // 2))    # (C/4, (C-a)/2)
        c = capacity - a - b                                 # then c in (C/4, C/2)
        items += [a, b, c]
    items = np.array(items, dtype=int)
    rng.shuffle(items)
    return Instance(f"triplet_n{len(items)}_c{capacity}_s{seed}", items, capacity)


# ----------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------

class InvalidHeuristic(Exception):
    pass


def pack_online(inst: Instance, priority: PriorityFn) -> int:
    """Run an online priority heuristic; return number of bins used.

    Raises InvalidHeuristic if the heuristic returns something unusable.
    """
    C = inst.capacity
    bins = np.full(len(inst.items), C, dtype=float)
    for item in inst.items:
        item = float(item)
        valid = np.nonzero(bins - item >= 0)[0]
        scores = priority(item, bins[valid])
        scores = np.asarray(scores, dtype=float)
        if scores.shape != valid.shape:
            raise InvalidHeuristic(f"priority returned shape {scores.shape}, expected {valid.shape}")
        if not np.all(np.isfinite(scores)):
            # NaN/inf scores make argmax ill-defined -> treat as failure
            raise InvalidHeuristic("priority returned non-finite values")
        bins[valid[int(np.argmax(scores))]] -= item
    used = int(np.sum(bins != C))
    # independent feasibility check (no bin overfilled)
    assert np.all(bins >= -1e-9)
    return used


# ----------------------------------------------------------------------------
# Baselines (written as priority functions so they share the simulator)
# ----------------------------------------------------------------------------

def priority_first_fit(item: float, bins: np.ndarray) -> np.ndarray:
    # bins arrive in index order, so prefer the lowest index
    return -np.arange(len(bins), dtype=float)


def priority_best_fit(item: float, bins: np.ndarray) -> np.ndarray:
    return -(bins - item)


def priority_funsearch_or(item: float, bins: np.ndarray) -> np.ndarray:
    """Heuristic reported by FunSearch for OR-Library-style instances
    (Romera-Paredes et al., Nature 2024, Fig. 3 / Supplementary). Transcribed
    from the paper; included only as a literature reference point."""
    max_bin_cap = max(bins)
    score = (bins - max_bin_cap) ** 2 / item + bins ** 2 / (item ** 2)
    score += bins ** 2 / item ** 3
    score[bins > item] = -score[bins > item]
    score[1:] -= score[:-1]
    return score


BASELINES: dict[str, PriorityFn] = {
    "first_fit": priority_first_fit,
    "best_fit": priority_best_fit,
    "funsearch_or": priority_funsearch_or,
}


def best_fit_decreasing(inst: Instance) -> int:
    """Offline reference (sees all items in advance): sort decreasing, then
    best fit. Not a fair competitor for online rules; an upper reference."""
    sorted_inst = Instance(inst.name, np.sort(inst.items)[::-1], inst.capacity)
    return pack_online(sorted_inst, priority_best_fit)


# ----------------------------------------------------------------------------
# Bounds and metrics
# ----------------------------------------------------------------------------

def l1_bound(inst: Instance) -> int:
    """Trivial lower bound ceil(sum / C)."""
    return int(math.ceil(inst.items.sum() / inst.capacity))


def l2_bound(inst: Instance) -> int:
    """Martello & Toth (1990) L2 lower bound, >= L1. O(n log n) version."""
    C = inst.capacity
    w = np.sort(inst.items)
    best = l1_bound(inst)
    for K in np.unique(np.concatenate([[0], w[w <= C // 2]])):
        J1 = w[w > C - K]
        J2 = w[(w <= C - K) & (w > C / 2)]
        J3 = w[(w <= C / 2) & (w >= K)]
        free = len(J2) * C - J2.sum()
        lb = len(J1) + len(J2) + max(0, math.ceil((J3.sum() - free) / C))
        best = max(best, lb)
    return int(best)


def excess_pct(bins_used: int, lb: int) -> float:
    """% bins above the lower bound (FunSearch's reported metric uses L1)."""
    return 100.0 * (bins_used - lb) / lb


def evaluate(priority: PriorityFn, instances: Sequence[Instance], bound: str = "l1") -> dict:
    bound_fn = l1_bound if bound == "l1" else l2_bound
    used = [pack_online(i, priority) for i in instances]
    lbs = [bound_fn(i) for i in instances]
    exc = [excess_pct(u, b) for u, b in zip(used, lbs)]
    return {"bins": used, "lb": lbs, "excess_pct": exc,
            "mean_excess_pct": float(np.mean(exc)),
            "total_bins": int(sum(used)), "total_lb": int(sum(lbs))}
