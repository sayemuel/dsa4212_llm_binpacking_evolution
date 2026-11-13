"""Child process: read candidate code on stdin, evaluate on a split, print JSON.
Run as: python -I _eval_worker.py <split>   (split = train | valid | test:<name>)"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import math  # noqa: E402

from binpack import evaluate  # noqa: E402
import datasets  # noqa: E402


def main():
    split = sys.argv[1]
    code = sys.stdin.read()
    ns = {"np": np, "numpy": np, "math": math}
    exec(code, ns)  # isolated process; code already passed static_check
    fn = ns["priority"]
    if split == "train":
        insts = datasets.TRAIN
    elif split == "valid":
        insts = datasets.VALID
    elif split.startswith("test:"):
        insts = datasets.TEST[split[5:]]
    else:
        raise ValueError(split)
    np.seterr(all="ignore")
    res = evaluate(fn, insts)
    res["valid"] = True
    print(json.dumps(res))


if __name__ == "__main__":
    main()
