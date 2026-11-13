"""Score baselines and selected evolved programs on all held-out TEST sets.

  python evaluate.py                      # baselines + every results/runs/*/best_program.py
Writes results/test_eval.json (one row per method x test set).
"""
from __future__ import annotations

import glob
import inspect
import json
import os
import time

import numpy as np

import datasets
from binpack import BASELINES, best_fit_decreasing, l1_bound, excess_pct
from sandbox import evaluate_code

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "test_eval.json")


def baseline_code(fn) -> str:
    return "import numpy as np\n\n" + inspect.getsource(fn).replace(fn.__name__, "priority", 1)


def main():
    methods = {name: baseline_code(fn) for name, fn in BASELINES.items()}
    distill = os.path.join(HERE, "..", "results", "mechanism_distill.json")
    if os.path.exists(distill):  # 3-parameter rule from mechanism.py distill
        a, b, c = json.load(open(distill))["best_params"]
        methods["gap_band(distilled)"] = (
            "import numpy as np\n\ndef priority(item, bins):\n    C = bins.max()\n"
            "    r = (bins - item) / C\n"
            f"    return -r - {c} * ((r >= {a}) & (r <= {b}))\n")
    for path in sorted(glob.glob(os.path.join(HERE, "..", "results", "runs", "*", "best_program.py"))):
        run = os.path.basename(os.path.dirname(path))
        if run.startswith("dryrun"):
            continue
        methods[run] = open(path).read()

    rows = []
    for split, insts in datasets.TEST.items():
        # offline reference (not online-comparable)
        bfd = [best_fit_decreasing(i) for i in insts]
        lbs = [l1_bound(i) for i in insts]
        rows.append({"method": "best_fit_decreasing(offline)", "split": split,
                     "mean_excess_pct": float(np.mean([excess_pct(u, b) for u, b in zip(bfd, lbs)])),
                     "bins": bfd, "lb": lbs})
        for name, code in methods.items():
            t0 = time.time()
            r = evaluate_code(code, f"test:{split}", timeout=600)
            rows.append({"method": name, "split": split, "valid": r["valid"],
                         "mean_excess_pct": r.get("mean_excess_pct"), "bins": r.get("bins"),
                         "lb": r.get("lb"), "error": r.get("error"),
                         "seconds": time.time() - t0})
            print(f"{split:14s} {name:45s} {r.get('mean_excess_pct')}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
