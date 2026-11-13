"""Behavioural distance of every proposal from Best Fit (CPU only).

For a fixed state sequence (the states Best Fit visits while packing the first
TRAIN instance), compute the fraction of states where a candidate's arg-max bin
differs from Best Fit's. This 'off-policy disagreement rate' measures how much
a proposal actually changes decisions, independently of whether the change helps.

  python disagreement.py      -> results/disagreement.json, report/figures/disagreement.pdf
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def bf_states():
    from datasets import TRAIN
    inst = TRAIN[0]
    C = inst.capacity
    bins = np.full(len(inst.items), C, dtype=float)
    states = []
    for item in inst.items:
        valid = np.nonzero(bins - item >= 0)[0]
        states.append((float(item), bins[valid].copy()))
        bins[valid[int(np.argmax(-(bins[valid] - item)))]] -= item
    return states


def worker():
    """stdin: JSON list of code strings; stdout: JSON list of rates (or null)."""
    codes = json.loads(sys.stdin.read())
    states = bf_states()
    bf_choice = [int(np.argmax(-(b - it))) for it, b in states]
    np.seterr(all="ignore")
    out = []
    for code in codes:
        try:
            ns = {"np": np}
            exec(code, ns)
            f = ns["priority"]
            d = sum(int(np.argmax(np.asarray(f(it, b), dtype=float))) != c
                    for (it, b), c in zip(states, bf_choice))
            out.append(d / len(states))
        except Exception:
            out.append(None)
    print(json.dumps(out))


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*"))):
        run = os.path.basename(d)
        if run.startswith("dryrun") or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        recs = [json.loads(l) for l in open(os.path.join(d, "proposals.jsonl"))]
        recs = [r for r in recs if r["valid"]]
        codes = [r["code"] for r in recs]
        rates = []
        for i in range(0, len(codes), 50):  # batches in a child process, with timeout
            try:
                p = subprocess.run([sys.executable, "-I", __file__, "--worker"],
                                   input=json.dumps(codes[i:i + 50]), capture_output=True,
                                   text=True, timeout=600)
                rates += json.loads(p.stdout.strip().splitlines()[-1])
            except Exception:
                rates += [None] * len(codes[i:i + 50])
        for r, q in zip(recs, rates):
            rows.append({"run": run, "idx": r["idx"], "train_excess": r["train_excess"],
                         "disagree": q})
        print(run, len(recs), flush=True)
    json.dump(rows, open(os.path.join(ROOT, "results", "disagreement.json"), "w"))


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    if "--worker" in sys.argv:
        worker()
    else:
        main()
