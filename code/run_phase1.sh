#!/usr/bin/env bash
# Phase 1: Haiku 5.5, evolve vs best-of-N sampling, 3 seeds, 300 proposals each.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
mkdir -p ../results/logs
for mode in evolve sample; do
  for seed in 0 1 2; do
    $PY evolve.py --backend anthropic --model claude-haiku-5-5 --mode $mode \
      --budget 300 --workers 2 --seed $seed \
      > ../results/logs/${mode}_claude-haiku-5-5_b300_s${seed}.log 2>&1 &
  done
done
wait
echo "phase1 done"
