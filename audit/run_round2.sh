#!/usr/bin/env bash
# Round 2 -- are span lifetimes runtime-revealed? Control (pd_entropy) vs treatment
# (confidence threshold) at steps_ratio < 1. At steps_ratio=1 the threshold is a
# no-op (see PREREGISTRATION_R2.md), so do not run it there.
set -euo pipefail
cd "$(dirname "$0")/.."
R=audit/results; L=$R/logs; mkdir -p "$L"
PY=${PY:-python}
N=${N:-40}; STRIDE=${STRIDE:-20}; THR=${THR:-0.9}
RATIOS=${RATIOS:-"0.5 0.25"}

for sr in $RATIOS; do
  printf '\n\033[1m===== control  pd_entropy  steps_ratio=%s  (%s prompts) =====\033[0m\n' "$sr" "$N"
  $PY audit/pd_audit.py --mode full --num_samples "$N" --stride "$STRIDE" --steps_ratio "$sr" \
    --out "$R/r2_control_sr$sr.jsonl" 2>&1 | tee "$L/r2_control_sr$sr.log" | tail -n 6
  printf '\n\033[1m===== treatment  confidence_threshold=%s  steps_ratio=%s =====\033[0m\n' "$THR" "$sr"
  $PY audit/pd_audit.py --mode full --num_samples "$N" --stride "$STRIDE" --steps_ratio "$sr" \
    --confidence_threshold "$THR" \
    --out "$R/r2_ct${THR}_sr$sr.jsonl" 2>&1 | tee "$L/r2_ct${THR}_sr$sr.log" | tail -n 6
  printf '\n\033[1m===== analysis  steps_ratio=%s =====\033[0m\n' "$sr"
  $PY audit/ct_analyze.py --treatment "$R/r2_ct${THR}_sr$sr.jsonl" --control "$R/r2_control_sr$sr.jsonl" \
    2>&1 | tee "$R/analysis_round2_sr$sr.txt"
done
echo; echo "Primary verdict is steps_ratio=0.5: $R/analysis_round2_sr0.5.txt"
