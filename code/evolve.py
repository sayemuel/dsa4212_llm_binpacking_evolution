"""LLM-guided program search for online bin-packing priority functions.

Modes (same proposal budget, same model, same seed program, same evaluator):
  evolve  FunSearch-style: island population; each prompt shows k previously
          evaluated programs (sorted worse -> better, with their scores) and asks
          for an improved version. Feedback flows from evaluation to prompts.
  sample  Best-of-N control: every prompt shows only the seed program, so
          proposals are i.i.d. and nothing learned is fed back. Isolates the
          contribution of the evolutionary loop.

Usage:
  python evolve.py --mode evolve --model claude-haiku-5-5 --budget 200 --seed 0
  python evolve.py --mode evolve --backend mock --budget 30 --seed 0   # dry run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from llm import AnthropicBackend, CodexBackend, MockBackend, extract_code
from sandbox import evaluate_code

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "runs")

SEED_PROGRAM = '''import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Best fit: prefer the bin that would be left with the least space."""
    return -(bins - item)
'''

SYSTEM_PROMPT = """You are an expert in combinatorial optimisation and Python/numpy programming. You design heuristics for the ONLINE one-dimensional bin packing problem.

Problem: items arrive one at a time and must be placed immediately and irrevocably. All bins have the same capacity. The goal is to minimise the number of bins used.

You write a function `priority(item: float, bins: np.ndarray) -> np.ndarray`.
- `item` is the size of the arriving item.
- `bins` holds the remaining capacities of every bin the item fits in (all >= item), in order of bin creation. It always includes at least one completely empty bin (remaining capacity = bin capacity), so choosing it means opening a new bin.
- Return one finite score per entry of `bins`, same shape. The item is placed in the bin with the highest score (ties -> earliest bin).
- Only `numpy` (as np) and `math` may be imported. No I/O, no randomness, no global state. The function must be fast: it is called once per item on instances with thousands of items.

Reply with a short explanation (at most 3 sentences) of your idea, then exactly one ```python code block containing the complete function named `priority` (plus any helper functions it needs)."""


def program_hash(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()[:10]


@dataclass
class Program:
    pid: int
    code: str
    score: float           # higher is better = -mean excess % over L1 on TRAIN
    signature: tuple       # bins used per train instance (behavioural cluster key)
    island: int
    parents: list


class Database:
    """Islands of programs; within an island, programs with identical
    behaviour (signature) are deduplicated, keeping the shortest code."""

    def __init__(self, n_islands: int, max_per_island: int, rng: random.Random):
        self.islands: list[list[Program]] = [[] for _ in range(n_islands)]
        self.max_per_island = max_per_island
        self.rng = rng

    def add(self, p: Program):
        isl = self.islands[p.island]
        for i, q in enumerate(isl):
            if q.signature == p.signature:
                if len(p.code) < len(q.code):
                    isl[i] = p
                return
        isl.append(p)
        isl.sort(key=lambda x: -x.score)
        del isl[self.max_per_island:]

    def sample(self, k: int, temperature: float) -> tuple[int, list[Program]]:
        """Pick an island, then k programs with rank-based softmax weights."""
        island = self.rng.randrange(len(self.islands))
        isl = self.islands[island]
        k = min(k, len(isl))
        w = [math.exp(-r / temperature) for r in range(len(isl))]
        chosen: list[Program] = []
        pool = list(range(len(isl)))
        for _ in range(k):
            j = self.rng.choices(pool, weights=[w[i] for i in pool])[0]
            pool.remove(j)
            chosen.append(isl[j])
        chosen.sort(key=lambda x: x.score)   # worse first, best last
        return island, chosen

    def reset_worst(self):
        """FunSearch-style: re-seed the worse half of islands from the best
        program of a randomly chosen surviving island."""
        order = sorted(range(len(self.islands)),
                       key=lambda i: self.islands[i][0].score if self.islands[i] else -1e9)
        half = len(order) // 2
        losers, winners = order[:half], order[half:]
        for i in losers:
            donor = self.islands[self.rng.choice(winners)][0]
            self.islands[i] = [Program(**{**asdict(donor), "island": i})]

    def best(self) -> Program:
        return max((p for isl in self.islands for p in isl), key=lambda p: p.score)

    def top(self, n: int) -> list[Program]:
        allp = {p.signature: p for isl in self.islands for p in isl}
        return sorted(allp.values(), key=lambda p: -p.score)[:n]


def build_prompt(mode: str, shown: list[Program]) -> str:
    if mode == "sample":
        return (f"Here is a baseline heuristic:\n\n```python\n{SEED_PROGRAM}```\n\n"
                "Write a new priority function that packs items into fewer bins than this baseline.")
    parts = ["Below are previous versions of the priority function, ordered from worse to better. "
             "Score = average percentage of bins used above the lower bound ceil(sum(items)/capacity) "
             "on the evaluation instances (lower is better).\n"]
    for i, p in enumerate(shown):
        parts.append(f"Version {i} (excess = {-p.score:.3f}%):\n```python\n{p.code}```\n")
    parts.append(f"Write version {len(shown)}: an improved priority function with a lower excess. "
                 "You may combine ideas from the versions above or try something new.")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["evolve", "sample"], default="evolve")
    ap.add_argument("--backend", choices=["codex", "anthropic", "mock"], default="codex")
    ap.add_argument("--model", default=None,
                    help="anthropic: e.g. claude-haiku-5-5; codex: omit for account default")
    ap.add_argument("--max-consecutive-fail", type=int, default=6,
                    help="stop the run after this many LLM failures in a row (e.g. quota)")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--budget", type=int, default=200, help="number of LLM proposals")
    ap.add_argument("--workers", type=int, default=8, help="concurrent proposals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--islands", type=int, default=4)
    ap.add_argument("--k", type=int, default=2, help="programs shown per prompt")
    ap.add_argument("--temperature", type=float, default=2.0, help="rank softmax temperature")
    ap.add_argument("--resets", type=int, default=3, help="island resets during the run")
    ap.add_argument("--max-usd", type=float, default=None, help="hard spend cap for this run")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.backend == "anthropic" and not args.model:
        args.model = "claude-opus-5-5"
    model_name = {"mock": "mock", "codex": args.model or "codex-default"}.get(args.backend, args.model)
    run_id = args.tag or f"{args.mode}_{model_name}_b{args.budget}_s{args.seed}"
    out_dir = os.path.join(RESULTS, run_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "system_prompt": SYSTEM_PROMPT, "seed_program": SEED_PROGRAM}, f, indent=2)

    if args.backend == "mock":
        backend = MockBackend(args.seed)
    elif args.backend == "codex":
        backend = CodexBackend(args.model, effort=args.effort)
    else:
        backend = AnthropicBackend(args.model, effort=args.effort)

    seed_eval = evaluate_code(SEED_PROGRAM, "train")
    assert seed_eval["valid"], seed_eval
    db = Database(args.islands, max_per_island=20, rng=rng)
    for i in range(args.islands):
        db.add(Program(0, SEED_PROGRAM, -seed_eval["mean_excess_pct"], tuple(seed_eval["bins"]), i, []))

    lock = threading.Lock()
    log_f = open(os.path.join(out_dir, "proposals.jsonl"), "w")
    counter = {"done": 0, "usd": 0.0, "in_tok": 0, "out_tok": 0, "consec_fail": 0, "stopped": None}
    reset_every = args.budget // (args.resets + 1) if args.resets else None
    t_start = time.time()

    def one(idx: int):
        with lock:
            if counter["stopped"]:
                return
            island, shown = db.sample(args.k, args.temperature)
            prompt = build_prompt(args.mode, shown)
        res = backend.complete(SYSTEM_PROMPT, prompt)
        code = extract_code(res.text) if res.ok else None
        ev = evaluate_code(code, "train") if code else {"valid": False, "error": res.error or "no code block"}
        rec = {"idx": idx, "t": time.time() - t_start, "island": island,
               "parents": [p.pid for p in shown], "parent_scores": [p.score for p in shown],
               "llm_ok": res.ok, "llm_error": res.error, "stop_reason": res.stop_reason,
               "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
               "usd": res.usd, "latency_s": res.latency_s, "request_id": res.request_id,
               "response": res.text, "code": code, "code_hash": program_hash(code) if code else None,
               "valid": ev.get("valid", False), "eval_error": ev.get("error"),
               "eval_seconds": ev.get("eval_seconds"),
               "train_bins": ev.get("bins"), "train_excess": ev.get("mean_excess_pct")}
        with lock:
            counter["done"] += 1
            counter["usd"] += res.usd
            counter["in_tok"] += res.input_tokens
            counter["out_tok"] += res.output_tokens
            counter["consec_fail"] = 0 if res.ok else counter["consec_fail"] + 1
            if args.max_usd is not None and counter["usd"] >= args.max_usd and not counter["stopped"]:
                counter["stopped"] = f"spend cap ${args.max_usd} reached"
                print("!!", counter["stopped"], flush=True)
            if counter["consec_fail"] >= args.max_consecutive_fail and not counter["stopped"]:
                counter["stopped"] = f"stopped after {counter['consec_fail']} consecutive LLM failures; last: {res.error}"
                print("!!", counter["stopped"], flush=True)
            if ev.get("valid"):
                db.add(Program(idx, code, -ev["mean_excess_pct"], tuple(ev["bins"]), island,
                               rec["parents"]))
            rec["best_so_far"] = -db.best().score
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if reset_every and args.mode == "evolve" and counter["done"] % reset_every == 0 \
                    and counter["done"] < args.budget:
                db.reset_worst()
            print(f"[{counter['done']:4d}/{args.budget}] valid={ev.get('valid')} "
                  f"exc={ev.get('mean_excess_pct')} best={rec['best_so_far']:.3f}% "
                  f"tok={counter['in_tok']}/{counter['out_tok']} ${counter['usd']:.3f} {res.latency_s:.1f}s "
                  f"{'' if res.ok else res.error[:120]}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, range(1, args.budget + 1)))
    log_f.close()

    # Final selection: re-score the top-5 train programs on VALID, keep the best.
    finalists = []
    for p in db.top(5):
        v = evaluate_code(p.code, "valid")
        finalists.append({"pid": p.pid, "train_excess": -p.score,
                          "valid_excess": v.get("mean_excess_pct"), "code": p.code})
    finalists.sort(key=lambda d: d["valid_excess"] if d["valid_excess"] is not None else 1e9)
    summary = {"run_id": run_id, "wall_seconds": time.time() - t_start,
               "total_usd": counter["usd"], "total_input_tokens": counter["in_tok"],
               "total_output_tokens": counter["out_tok"], "proposals_completed": counter["done"],
               "stopped_early": counter["stopped"], "seed_train_excess": seed_eval["mean_excess_pct"],
               "best_train_excess": -db.best().score, "selected": finalists[0], "finalists": finalists}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "best_program.py"), "w") as f:
        f.write(finalists[0]["code"])
    print(json.dumps({k: v for k, v in summary.items() if k != "finalists"}, indent=2)[:1500])


if __name__ == "__main__":
    main()
