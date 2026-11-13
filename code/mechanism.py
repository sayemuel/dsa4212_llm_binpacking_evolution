"""Mechanism experiments (CPU only, no LLM calls).

  python mechanism.py scale     # excess vs instance size n for key heuristics
  python mechanism.py openbins  # how many partially-filled bins each rule keeps open
  python mechanism.py distill   # grid-search a 3-parameter 'gap-band' rule on TRAIN

Outputs JSON to results/mechanism_*.json and figures to report/figures/.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import datasets
from binpack import (BASELINES, Instance, evaluate, l1_bound, weibull_instance,
                     uniform_instance, excess_pct)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)


def load_program(path):
    ns = {"np": np}
    exec(open(path).read(), ns)
    return ns["priority"]


def key_heuristics():
    h = {"best_fit": BASELINES["best_fit"], "funsearch_or": BASELINES["funsearch_or"]}
    best = os.path.join(ROOT, "results", "runs", "evolve_claude-haiku-5-5_b300_s2", "best_program.py")
    if os.path.exists(best):
        h["haiku_evolve_s2"] = load_program(best)
    return h


def gap_band(a: float, b: float, c: float, bonus: float = 0.0):
    """Best fit minus a penalty c*C when the leftover lies in [a*C, b*C],
    plus an optional exact-fit bonus. C inferred as max(bins) (empty bin)."""
    def priority(item, bins):
        C = bins.max()
        r = (bins - item) / C
        return -r - c * ((r >= a) & (r <= b)) + bonus * (r < 1e-9)
    return priority


def run_scale():
    ns = [100, 200, 500, 1000, 2000, 5000, 10000]
    reps = {100: 20, 200: 20, 500: 10, 1000: 5, 2000: 5, 5000: 3, 10000: 2}
    H = key_heuristics()
    H["gap_band(distilled)"] = gap_band(*json.load(open(os.path.join(ROOT, "results", "mechanism_distill.json")))["best_params"]) \
        if os.path.exists(os.path.join(ROOT, "results", "mechanism_distill.json")) else None
    H = {k: v for k, v in H.items() if v is not None}
    out = {}
    for name, fn in H.items():
        out[name] = {}
        for n in ns:
            insts = [weibull_instance(n, 7000 + 100 * i + n) for i in range(reps[n])]
            r = evaluate(fn, insts)
            out[name][n] = (float(np.mean(r["excess_pct"])), float(np.std(r["excess_pct"]) / np.sqrt(len(insts))))
            print(name, n, out[name][n], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "mechanism_scale.json"), "w"), indent=1)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for name, d in out.items():
        x = list(map(int, d)); m = [d[k][0] for k in d]; s = [d[k][1] for k in d]
        ax.errorbar(x, m, yerr=s, marker="o", ms=3, capsize=2, label=name)
    ax.set_xscale("log"); ax.set_xlabel("items per instance n (Weibull, C=100)")
    ax.set_ylabel("excess over L1 (%)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "scale.pdf")); fig.savefig(os.path.join(FIG, "scale.png"), dpi=110)


def trace_open_bins(inst: Instance, priority):
    """Number of opened-but-not-full bins after each item, and final bins."""
    C = inst.capacity
    bins = np.full(len(inst.items), C, dtype=float)
    open_counts = []
    for item in inst.items:
        valid = np.nonzero(bins - item >= 0)[0]
        s = np.asarray(priority(float(item), bins[valid]), dtype=float)
        bins[valid[int(np.argmax(s))]] -= item
        used = bins != C
        open_counts.append(int(np.sum(used & (bins > 0))))
    return np.array(open_counts), int(np.sum(bins != C))


def run_openbins():
    H = key_heuristics()
    inst = weibull_instance(5000, 1000)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    out = {}
    for name, fn in H.items():
        oc, used = trace_open_bins(inst, fn)
        out[name] = {"final_bins": used, "mean_open": float(oc.mean()), "max_open": int(oc.max())}
        ax.plot(oc, label=f"{name} ({used} bins)", lw=1)
        print(name, out[name])
    ax.set_xlabel("items packed"); ax.set_ylabel("non-full open bins")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "openbins.pdf")); fig.savefig(os.path.join(FIG, "openbins.png"), dpi=110)
    json.dump(out, open(os.path.join(ROOT, "results", "mechanism_openbins.json"), "w"), indent=1)


def run_distill():
    """If the evolved programs' gain comes from penalising 'awkward' mid-sized
    leftovers, a 3-parameter rule tuned on TRAIN should recover most of it."""
    grid_a = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08]
    grid_b = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    grid_c = [0.05, 0.2, 0.5, 1.0, 2.0]  # c>=1 dominates any best-fit difference (hard band)
    rows = []
    for a, b, c in itertools.product(grid_a, grid_b, grid_c):
        r = evaluate(gap_band(a, b, c), datasets.TRAIN)
        rows.append({"a": a, "b": b, "c": c, "train_excess": r["mean_excess_pct"]})
    rows.sort(key=lambda d: d["train_excess"])
    best = rows[0]
    v = evaluate(gap_band(best["a"], best["b"], best["c"]), datasets.VALID)
    tests = {k: evaluate(gap_band(best["a"], best["b"], best["c"]), insts)["mean_excess_pct"]
             for k, insts in datasets.TEST.items()}
    out = {"best_params": [best["a"], best["b"], best["c"]], "best_train": best["train_excess"],
           "valid": v["mean_excess_pct"], "test": tests, "top10": rows[:10], "n_configs": len(rows),
           "best_fit_train": evaluate(BASELINES["best_fit"], datasets.TRAIN)["mean_excess_pct"]}
    print(json.dumps({k: out[k] for k in ["best_params", "best_train", "valid", "test", "best_fit_train"]}, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", "mechanism_distill.json"), "w"), indent=1)


def run_extra():
    """Absolute extra bins over L1 vs n: constant overhead vs proportional waste."""
    from binpack import pack_online
    reps = {100: 20, 200: 20, 500: 10, 1000: 5, 2000: 5, 5000: 3, 10000: 2}
    out = {}
    for n, k in reps.items():
        insts = [weibull_instance(n, 7000 + 100 * i + n) for i in range(k)]
        out[n] = {}
        for name in ["best_fit", "funsearch_or"]:
            extra = [pack_online(I, BASELINES[name]) - l1_bound(I) for I in insts]
            out[n][name] = (float(np.mean(extra)), float(np.std(extra) / np.sqrt(k)))
        print(n, out[n], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "mechanism_extra_bins.json"), "w"), indent=1)


if __name__ == "__main__":
    {"scale": run_scale, "openbins": run_openbins, "distill": run_distill,
     "extra": run_extra}[sys.argv[1]]()
