# Round 0 raw results

Produced by `audit/run_round0.sh` on the GPU box; synced back unmodified.

| file | what | n |
|---|---|---|
| `plan_default.jsonl` | AR plan only, model defaults, `length_scale=None`, `use_cache` off | 805 (804 ok, 1 `ar_hit_max_length`) |
| `full_30.jsonl` | full `planned_diffusion_generate`, `alg=pd_entropy`, per-step per-block mask trajectory | 30 |
| `smoke.jsonl` | 3-prompt smoke test | 3 |
| `analysis_*.txt` | `analyze.py` output at the analysis commit (see git log) | |
| `logs/` | stdout of every stage | |

## Provenance

- code: branch `audit/round-0`; rows carry no `subset` field, so the run predates
  `74c3389` — the plan/full code paths are identical from `da0fcd3` through `25d5c1e`.
- base repo commit: `5c52637`
- model: `dmisrael/planned-diffusion-dream7b-sft-16ep`, bf16
- prompts: `tatsu-lab/alpaca_eval` `alpaca_eval.json`, dataset order, all 805
- sampling: `temperature=0.2`, `top_p=0.95`, `max_length=1024`, `steps_ratio=1.0`,
  seed 42 re-set per prompt
- device: `cuda` (GPU model not recorded by the script — fill in below)
- GPU: __________  driver/CUDA: __________  torch: __________

## Headline (preregistered rule, unchanged)

`N=804  F2=0.658  W=0.167  ->  GRAY`. `W_agg=0.399` descriptive only.
Falsification: `finish_step == l_k` for 220/220 spans — see AMENDMENT 1 in
`../PREREGISTRATION.md`.

## Additional observations from `full_30.jsonl` (raw, pre-Round-1)

Classification of the 30 fully generated requests by joining to the plan-mode
row of the same `request_id` (plans identical in 30/30 — same seed, same plan):

| class | n | note |
|---|---|---|
| clean single round (`eos`) | 11 | used for Round 1 calibration |
| multi-round (`sync` → plan → …) | 16 | 2–12 rounds; mean 2.23 over all 30 |
| **runaway AR after `sync`** | **3** | requests 0, 16, 19 |

**Runaway AR.** In 3 of the 19 `sync`-terminated plans the second AR phase never
emitted a plan or a terminator and ran to `max_length` (975–998 tokens). Because
the default path has no KV cache, that AR phase is O(n²): 58 s and 57 s of
planning for 1.6 s and 0.9 s of diffusion (req 0, 19). These three requests
account for **70% of all planning time in the sample** (119 s of 169 s). This is
a runtime robustness failure of the planner/loop, not a workload property; it
is excluded from Round 1 calibration and noted here because any latency
comparison built on the default path will be dominated by it unless guarded.

**Consequence for `total_tokens`.** It counts the runaway AR tokens, so it cannot
be used to recover the prefix length; Round 1 takes `P` from the plan-mode row.
