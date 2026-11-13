#!/usr/bin/env bash
# Phase 3 (follow-ups): does reasoning effort explain timid proposals?
# Completes a model x effort 2x2 (low-effort cells are in phases 1-2).
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
mkdir -p ../results/logs
$PY evolve.py --backend anthropic --model claude-opus-5-5 --effort high \
  --mode evolve --budget 60 --workers 2 --seed 0 --max-usd 1.0 \
  --tag evolve_claude-opus-5-5-high_b60_s0 \
  > ../results/logs/evolve_claude-opus-5-5-high_b60_s0.log 2>&1
$PY evolve.py --backend anthropic --model claude-haiku-5-5 --effort high \
  --mode evolve --budget 150 --workers 2 --seed 0 --max-usd 0.4 \
  --tag evolve_claude-haiku-5-5-high_b150_s0 \
  > ../results/logs/evolve_claude-haiku-5-5-high_b150_s0.log 2>&1
echo "phase3 done"
