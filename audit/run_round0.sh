#!/usr/bin/env bash
# Round 0 gating run. Everything here must pass before a second workload is worth
# starting. See audit/PREREGISTRATION.md -- do not edit the decision rule after
# seeing any of this output.
set -euo pipefail
cd "$(dirname "$0")/.."

R=audit/results
L=$R/logs
mkdir -p "$R" "$L"
PY=${PY:-python}
N_PRIMARY=${N_PRIMARY:-}          # empty = all 805
N_FALSIFY=${N_FALSIFY:-30}
# AlpacaEval is stored ordered by subset, so a contiguous partial run samples only
# helpful_base. Spread any partial run across all five subsets instead.
if [ -n "$N_PRIMARY" ]; then STRIDE=$((805 / N_PRIMARY)); [ "$STRIDE" -lt 1 ] && STRIDE=1; else STRIDE=1; fi
FALSIFY_STRIDE=$((805 / N_FALSIFY)); [ "$FALSIFY_STRIDE" -lt 1 ] && FALSIFY_STRIDE=1

banner() { printf '\n\033[1m===== %s =====\033[0m\n' "$*"; }

banner "0/4  smoke test (3 prompts) -- first run also downloads ~15 GB of weights"
$PY audit/pd_audit.py --mode plan --num_samples 3 --out "$R/smoke.jsonl" 2>&1 | tee "$L/smoke.log"
if ! grep -q " ok " "$L/smoke.log"; then
  echo "SMOKE FAILED -- read $L/smoke.log before going further" >&2; exit 1
fi
echo
echo "Per-prompt wall time is printed above. Multiply it by 805 to predict stage 1."

banner "1/4  PRIMARY -- AlpacaEval plans, model defaults, no length_scale"
# shellcheck disable=SC2086
$PY audit/pd_audit.py --mode plan --stride "$STRIDE" ${N_PRIMARY:+--num_samples $N_PRIMARY} \
  --out "$R/plan_default.jsonl" 2>&1 | tee "$L/plan_default.log" | tail -n 20

banner "2/4  VERDICT"
$PY audit/analyze.py "$R/plan_default.jsonl" --label "primary (default plan)" \
  2>&1 | tee "$L/verdict.log"

banner "3/4  FALSIFICATION -- is the join really lockstep? (alg=pd_entropy)"
$PY audit/pd_audit.py --mode full --num_samples "$N_FALSIFY" --stride "$FALSIFY_STRIDE" \
  --out "$R/full_${N_FALSIFY}.jsonl" 2>&1 | tee "$L/full.log" | tail -n 10
$PY audit/analyze.py "$R/full_${N_FALSIFY}.jsonl" --label "full generation" \
  2>&1 | tee "$L/falsification.log"

banner "4/4  SUMMARY"
grep -E "^VERDICT|^F2 =" "$L/verdict.log" || true
grep -E "retired early" "$L/falsification.log" || true
cat <<'MSG'

Read it like this:
  * "rounds where a span retired early" must be 0/N. If it is not, the plan-derived
    join_waste is wrong, the preregistration is void, and the VERDICT above means
    nothing -- report that and stop.
  * VERDICT STOP  -> planned diffusion offers no scheduling opportunity. Do not
    start a second workload without writing this negative result down first.
  * VERDICT GRAY  -> not sufficient alone; a second workload is required.
  * VERDICT GO    -> proceed. Optional next: audit/RUN_GPU.md steps 5b, 6, 7.

All logs are in audit/results/logs/.
MSG
