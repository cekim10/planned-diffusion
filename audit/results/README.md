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
- GPU: NVIDIA L40S (48 GB GDDR6; from `fwd_curve.json`)  torch: see `fwd_curve.json`  python 3.12 venv

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

## Round 1 — compaction headroom (preregistered rule in `../PREREGISTRATION_R1.md`)

Files: `fwd_curve.json` (forward latency at L=72..4028, real PD masks, L40S),
`analysis_round1.txt` (replay of the 802 primary plans), `logs/bench.log`.

**Calibration:** predicted vs measured diffusion latency on the 11 clean
single-round requests: median **1.09** (p10 1.05, p90 1.19). The cost model is
good to ~10%; the replay can be trusted.

| prefix cached, k≤50 (primary) | total | rel |
|---|---|---|
| `T_current` (repo, 1 request) | 1309 s | 100% |
| `T_compact` (retire spans) | 1251 s | 95.6% |
| `T_packedCB` (perfect request-level packing) | 601 s | 45.9% |
| `T_ideal` (perfect span-level packing) | 495 s | 37.8% |

`S_compact = 0.044` (per-request median 0.000, p90 0.070) — **latency claim dead**:
at L median 147 the L40S is memory-bound (floor 27.8 ms, saturation ~L 600), so
removing finished spans barely changes per-step time and the step count is unchanged.

`H = 0.177` — span-level packing beats perfect request-level packing by 17.7%.
Below the 0.20 GO line. Conditional on `k≥2`: 0.195; `k≥3`: 0.207; `k≥5`: 0.211.

`split = 0.07` — of the current→ideal gap, retirement captures 7%; request-level
batching captures 93%. **The dominant inefficiency in serving this model today is
the absence of request-level batching, not span scheduling.** A single PD request
runs at 42% of peak throughput.

**VERDICT: GRAY** (both axes below GO).

### Correction to Round 0's descriptive `W_agg = 0.399`

That number was labeled descriptive-only and the Round 0 rule used the median, so
no verdict changes — but it was wrong as a statement about GPU time, twice over:

| | slot-step unit (Round 0) | compute unit (Round 1) |
|---|---|---|
| all 804 | **0.399** | 0.300 |
| k ≤ 50 (802) | 0.260 | **0.177** |

1. The two repetition-loop plans (k=101, 112; 900-token blocks) inflate the
   aggregate by ~14 points.
2. The slot-step unit counts one unmask slot per span per step. GPU cost is
   resident tokens per step: a span of length `l` is resident for `l` steps at
   `l` tokens each (`l²`). Long spans, which are live longest, also cost most per
   step, so short-span retirement frees a smaller share of FLOPs than of slots.
   Example `[30,120,70]`: slot 0.389, compute 0.235.

The structural compute headroom of span-level over request-level packing on
this workload is **~18% (all requests) to ~21% (forked requests)**, not 40%.

## Round 2 — runtime-revealed lifetimes? (`../PREREGISTRATION_R2.md`)

Files: `r2_control_sr{0.5,0.25}.jsonl` (pd_entropy), `r2_ct0.9_sr{0.5,0.25}.jsonl`
(confidence threshold 0.9), 40 strided prompts each, per-step per-block mask
trajectories. `analysis_round2_sr*.txt` were regenerated locally after fixing a
tie-handling bug in the rank correlation; the GPU-produced originals are kept as
`*_GPU_ORIGINAL_tiebug.txt`. Both computed from the same jsonl.

| | sr 0.5 (primary) | sr 0.25 |
|---|---|---|
| control finish rule | `min(l_k, og_steps)` 239/239 | 254/254 |
| σ_sib treatment / control | 0.019 / 0.000 | 0.087 / 0.000 |
| ρ(l_k, finish) treatment / control | 0.822 / 0.906 | 0.538 / 0.812 |
| CT shift in steps: 0 / 1 / 2 / ≥3 | 68 / 28 / 3 / 0 % (max 3) | 56 / 27 / 15 / 3 % (max 5) |
| online ρ, flat across the round | 0.816 | ~0.35 |
| **verdict (rule as written)** | **GRAY** | DYNAMIC |

The GPU output printed DYNAMIC at sr 0.5 on ρ = 0.700 exactly; that value was
an artifact of position-broken ties. Corrected ρ is 0.822 → GRAY.

Confidence gating moves span finish by 1–2 denoising steps (5–8% of a round),
only for spans with `l_k ≥ og_steps`, and in a way the online estimator cannot
anticipate. Assessment: close PD as the lead workload for runtime-revealed DAG
scheduling; keep it as the supporting workload for the static ~18% result.

## Round 3 — does knowing the chain help a strong scheduler? (`../PREREGISTRATION_R3.md`)

No GPU. Iteration-level batching simulator (`../chain_sim.py`) on the 106 chained
requests, iteration time from `fwd_curve.json`. Files: `analysis_round3.txt`,
`chain_sim.json`. Capacity ≈ 1.2 req/s; operating window = `fcfs` P99 within
1.5–4× its unloaded value (11.8 s at sr 0.5 → 18–47 s).

Primary cell (sr 0.5, B 1024), best oracle P99 gain over Sangam-like FCFS in the
window: **9.6%**. Chain-blind size-aware ordering beats the clairvoyant chain
oracle at every load (chain value −2 to −18 points). Critical-path-first is
catastrophic (−7% to −144%). Same pattern in every `(sr, B)`.

**VERDICT: KILL.** Planned Diffusion scheduling is closed after four preregistered
tests (compaction, static span scheduling, CT dynamic scheduling, chain-aware
scheduling).

## What the PD track established (keep)

- Spans have exact natural retirement points: `finish_k = min(l_k, og_steps)`,
  493/493 spans (R0 + R2 controls).
- 60% of requests are multi-round; the execution graph is series-parallel,
  materializing one round at a time; 53% of diffusion work is invisible at admission.
- At the paper's speed settings, 35–54% of request latency is the AR planner
  running at ~32% of GPU peak (uncached path).
- Static span-level packing is worth ~18% over perfect request-level batching;
  compaction alone ~4% latency; 93% of the gap to ideal is request-level batching.
- Confidence-threshold decoding moves span finish by 1–2 steps, only for spans
  with `l_k ≥ og_steps`, not learnable mid-round.
- Runaway AR after `<sync>` in 3/19 cases (70% of sample planning time) — a
  runtime robustness hole.
- Chain knowledge has no marginal scheduling value over region-size awareness.
