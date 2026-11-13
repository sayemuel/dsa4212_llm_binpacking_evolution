# LLM-guided evolution of online bin-packing heuristics

DSA4212 final assignment by Samuel Goh Jun Yang (A0259280Y). This is an individual submission, as I chose to work alone. The report source is in [`report/`](report/).

## What this project does

I built a small version of FunSearch. A Claude model writes Python priority functions for online bin packing, a sandboxed simulator scores them, and the better programs are shown back to the model in later prompts. Using this setup I compare three things:

1. The evolutionary loop against a simple control, where the model is sampled independently with the same number of proposals and no feedback.
2. Two models (Claude Haiku 5.5 and Claude Opus 5.5) at low and high reasoning effort.
3. What the programs that work actually do, and whether they still work on different instance sizes and item distributions.

The whole study cost US$9.10 in API credits. Everything after the search runs on a normal laptop CPU.

## Folder layout

```
code/
  binpack.py        packing simulator, baselines (First Fit, Best Fit, FunSearch's heuristic), L1 and L2 bounds
  datasets.py       TRAIN, VALID and TEST instance sets, all from fixed seeds that do not overlap
  sandbox.py        safety checks on generated code, then runs it in a separate process with a time limit
  _eval_worker.py   the child process that runs one candidate program
  llm.py            Anthropic API wrapper that records tokens and cost, plus a Codex and a mock backend
  evolve.py         the search loop, with --mode evolve (islands and feedback) or --mode sample (the control)
  evaluate.py       scores the baselines and each run's chosen program on every TEST set
  mechanism.py      analyses with no API calls: scaling with instance size, open bin traces, the 3 parameter rule
  disagreement.py   how often each proposal picks a different bin from Best Fit
  explore.py        later exploratory analyses: best of N model, final leftovers, parameter landscape, code ideas
  analyze.py        collects all runs into the tables, statistics and figures used in the report
  run_phase1.sh     Haiku, 300 proposals, evolve and sample, seeds 0 to 2
  run_phase2.sh     Haiku, 800 proposals, seeds 10 to 12, and Opus, 120 proposals
  run_phase3.sh     high effort runs (Opus, stopped by a $1 cap, and Haiku, 150 proposals)
results/
  runs/<run_id>/    config.json, proposals.jsonl (every prompt outcome, code, tokens and cost),
                    summary.json, best_program.py
  logs/             console output of each run
  *.json            outputs of the evaluation and analysis scripts
report/
  report.tex, references.bib, figures/, tables/
```

## Setup

You need Python 3.10 or newer on macOS or Linux. No GPU is needed.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # only needed if you want to rerun the LLM search
```

There is nothing to download. All instances are generated from fixed seeds in `code/datasets.py`. The main data uses Weibull item sizes (shape 3, scale 45) with bin capacity 100, as in FunSearch. The test sets also include OR-Library style uniform sizes from 20 to 100 with capacity 150, and Falkenauer style triplet instances whose optimum is known.

## Reproducing the results

The LLM's outputs are random, so rerunning the search will give different programs. Everything after the search is deterministic, given the saved files in `results/runs/`. Steps 2 to 5 take about 10 minutes on an M1 Pro laptop.

```bash
cd code
# 1. Search. This costs API credits. My actual spend was $1.17, $6.68 and $1.25 for the three phases.
./run_phase1.sh
./run_phase2.sh
./run_phase3.sh
# 2. Fit the 3 parameter gap band rule by grid search (evaluate.py and mechanism.py scale read its output)
../.venv/bin/python mechanism.py distill
# 3. Score the baselines, the gap band rule and every run's chosen program on the TEST sets
../.venv/bin/python evaluate.py
# 4. Analyses that explain the results (no API calls)
../.venv/bin/python mechanism.py openbins
../.venv/bin/python mechanism.py scale
../.venv/bin/python mechanism.py extra
../.venv/bin/python disagreement.py
../.venv/bin/python explore.py bestofn      # does sampling behave like independent draws?
../.venv/bin/python explore.py leftovers    # leftover space in each bin, and item size shapes
../.venv/bin/python explore.py landscape    # 11 x 19 grid of the gap band rule's parameters
../.venv/bin/python explore.py ideas        # which coding ideas show up and how often they work
# 5. Tables, statistics and the remaining figures
../.venv/bin/python analyze.py
```

To check that the pipeline works without spending anything, run `python evolve.py --backend mock --budget 20 --tag dryrun_mock`.

`analyze.py` also draws `report/figures/convergence.pdf`. An earlier draft of the report used it, but the final report shows the same runs in Figure 1 (`bestofn.pdf`) instead.

## Notes

* The simulator, not the generated code, decides which bins are allowed. A heuristic only ever sees bins the item fits in, and the simulator checks that no bin is overfilled.
* Costs are calculated from the token counts the API reports and the list prices in `code/llm.py`.
* I used AI assistance (Claude Code) for coding and drafting, which the course guidelines allow. All results come from the code in this repository.
