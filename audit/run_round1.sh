#!/usr/bin/env bash
# Round 1 -- compaction headroom kill test. Needs audit/results/plan_default.jsonl
# (Round 0) and a GPU. See audit/PREREGISTRATION_R1.md; rule fixed in advance.
set -euo pipefail
cd "$(dirname "$0")/.."
R=audit/results; L=$R/logs; mkdir -p "$L"
PY=${PY:-python}

printf '\n\033[1m===== 1/2  fwd(L) microbenchmark (real model, real PD masks) =====\033[0m\n'
$PY audit/compaction_bench.py --out "$R/fwd_curve.json" ${GRID:+--grid "$GRID"} 2>&1 | tee "$L/bench.log"

printf '\n\033[1m===== 2/2  replay 804 Round-0 plans through the curve =====\033[0m\n'
$PY audit/compaction_replay.py --curve "$R/fwd_curve.json" 2>&1 | tee "$R/analysis_round1.txt"

echo; echo "Outputs: $R/fwd_curve.json  $R/analysis_round1.txt  $L/bench.log"
