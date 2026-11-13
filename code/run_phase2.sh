#!/usr/bin/env bash
# Phase 2: (a) longer Haiku budget (800) to test whether feedback pays off later;
#          (b) Opus 5.5 (stronger, ~15x pricier per proposal) at 120 proposals.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
mkdir -p ../results/logs
for mode in evolve sample; do
  for seed in 10 11 12; do
    $PY evolve.py --backend anthropic --model claude-haiku-5-5 --mode $mode \
      --budget 800 --workers 2 --seed $seed --max-usd 0.6 \
      > ../results/logs/${mode}_claude-haiku-5-5_b800_s${seed}.log 2>&1 &
  done
done
for seed in 0 1; do
  $PY evolve.py --backend anthropic --model claude-opus-5-5 --mode evolve \
    --budget 120 --workers 2 --seed $seed --max-usd 1.7 \
    > ../results/logs/evolve_claude-opus-5-5_b120_s${seed}.log 2>&1 &
done
$PY evolve.py --backend anthropic --model claude-opus-5-5 --mode sample \
  --budget 120 --workers 2 --seed 0 --max-usd 1.7 \
  > ../results/logs/sample_claude-opus-5-5_b120_s0.log 2>&1 &
wait
echo "phase2 done"
