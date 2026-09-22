# Round 1 — Compaction headroom kill test

**Status:** preregistered before any Round 1 measurement. Written after Round 0
(GRAY, `W=0.167`, `W_agg=0.399`, `finish_step == l_k` 220/220).

## Question

Round 0 showed that in a planned-diffusion round each span retires at step
`l_k` while the region runs `max_k l_k` steps, so finished spans are recomputed
for `max_k l_k - l_k` steps. Across the 804 AlpacaEval plans that is 39.9% of
span-token-steps. **Is that 39.9% real GPU time?**

The danger is specific. Round-1 sequences are short — total `L` median 147,
p90 413 tokens (prefix ~60 + blocks ~76). For a 7B model at batch 1 that is the
memory-bound regime where a forward pass mostly streams weights and its latency
is nearly flat in `L`. If so, removing finished spans barely changes per-step
time, the step count is unchanged (`max_k l_k`), and **span retirement alone
saves nothing for a single request.** The 39.9% is then only realizable by
filling freed slots with other requests' spans — a throughput claim under load,
whose honest baseline is not "the repo today" but **request-level continuous
batching** that already packs whole regions across requests.

So there are two distinct claims, and the test must separate them:

- **Latency claim:** retiring spans makes a single request faster.
- **Throughput claim:** span-level packing beats request-level packing.

## Method

### 1. `fwd(L)` microbenchmark — real model, real masks

`audit/compaction_bench.py` builds a planned-diffusion region with the repo's own
`create_pd_inputs` (prefix + `k` async blocks, block-sparse mask, inverted bf16
additive mask exactly as `_diff_sample` passes it) and times `model(...)` at a grid
of total lengths `L` from ~64 to 4096. Output: median ms and tokens/s per `L`.

From the curve: `thr_sat` = max tokens/s; `L_sat` = smallest `L` with tokens/s
≥ 0.9·`thr_sat`; linearity of ms vs `L` above `L_sat` (R² and intercept share).
If per-token cost is not roughly constant above `L_sat`, "capacity" is not
fungible and the throughput claim is discounted accordingly.

### 2. Replay of the measured plan distribution

`audit/compaction_replay.py` takes the 804 first-round plans from Round 0
(`prompt_tokens`, `plan_token_count`, `span_lengths`) and, using the measured
curve, computes for each request with `S = max_k l_k` steps:

| quantity | per step `s = 1..S` | meaning |
|---|---|---|
| `T_current`  | `fwd(P + Σ_k (l_k+2))` | repo today: one request, no retirement |
| `T_compact`  | `fwd(P + Σ_{k: l_k ≥ s} (l_k+2))` | one request, finished spans removed |
| `T_packedCB` | `(P + Σ_k (l_k+2)) / thr_sat` | perfect request-level packing, no retirement |
| `T_ideal`    | `(P + Σ_{k: l_k ≥ s} (l_k+2)) / thr_sat` | perfect span-level packing |

Span `k` is live at step `s` iff `s ≤ l_k` (it finishes at step `l_k`, Round 0).
`P` is the prefix (prompt + plan). Two variants: **prefix cached** (`P → 0`,
the paper's `--use_cache` setting and any serving system; primary) and prefix
uncached (`P` recomputed every step, the repo's default path; reported).

### 3. Calibration

The default-path run in Round 0 (`full_30.jsonl`, `use_cache` off) recorded
`diffusion_latency` and the step count per request. For its single-round
requests, compare `S · fwd(P + Σ(l_k+2))` (uncached) against the measured
diffusion latency and report the ratio. It should be ≥ 1 (measured includes
sampling/`block_unmask`/hook overhead per step); if it is far from 1 the cost
model is wrong and the replay is not to be trusted.

## Which requests enter the aggregate — decided before measurement

Round 0 contains two degenerate plans (`request_id` 103 and 445: `k = 101` and
`k = 112`, repetition loops emitting "step 1, step 2, …", the latter declaring
900-token blocks and a 60k-token region). A dry run of the replay on a
*synthetic* curve, done before any GPU measurement, showed these two requests
dominating every aggregate `ΣT` — the aggregate `S_compact` was 0.29 while the
per-request median was 0.00. They are runtime robustness failures of the
planner, not the workload this test is about.

**Primary aggregate: the 802 plans with `k ≤ 50`.** The all-804 aggregate is
reported alongside as secondary. Both per-request distributions are reported.
This exclusion is fixed here, before Round 1 data exists, and is the only
change to the plan distribution relative to Round 0.

## Metrics (aggregate over the replayed plans, prefix-cached)

- `S_compact = 1 − ΣT_compact / ΣT_current` — single-request latency gain from
  span retirement alone.
- `H = 1 − ΣT_ideal / ΣT_packedCB` — span-level headroom over perfect
  request-level batching. (This is near-structural by construction; the GPU
  measurement's job here is to validate that per-token cost is constant at
  saturation, i.e. that `H` is realizable.)
- `split = (ΣT_current − ΣT_compact) / (ΣT_current − ΣT_ideal)` — share of the
  total gap that static retirement alone captures. Reported, not a rule input.

Also reported per request: distribution of `1 − T_compact/T_current`, and
everything again for the uncached variant.

## Decision rule — fixed in advance

| condition | verdict |
|---|---|
| `S_compact < 0.05` **and** `H < 0.10` | **STOP** — neither a latency nor a throughput story |
| `S_compact ≥ 0.15` | **GO-latency** — span retirement is worth having on its own |
| `H ≥ 0.20` **and** curve linear above `L_sat` (R² ≥ 0.98, intercept ≤ 15% of `ms(L_sat)`) | **GO-throughput** — span-level packing beats request-level CB by a margin a systems paper can carry |
| `H ≥ 0.20` but curve **not** linear | **GRAY** — headroom exists structurally but per-token cost is not constant; needs a packed-kernel measurement before it counts |
| otherwise | **GRAY** |

GO-latency and GO-throughput are independent; either alone is a GO for the
corresponding claim only. Expectation stated in advance: `S_compact` is likely
small (memory-bound regime), and the case, if any, will rest on `H`.

## Evidence hygiene

- The plan distribution is the Round 0 primary run: model defaults,
  `length_scale=None`, `k = 1` included (they
  contribute `T_compact = T_current` and `T_ideal = T_packedCB`, diluting both
  savings, as they should).
- The benchmark uses the repo's mask construction and forward call; it does not
  use a packed/ragged kernel. `T_packedCB` and `T_ideal` therefore assume a
  packer can reach `thr_sat` with negligible masking overhead. This is stated as
  an assumption, and the GRAY row above exists for when the curve says it is
  not safe.
- Multi-round requests (60% of plans end in `<sync>`) are **not** replayed here
  — only round 1 is available for all 804. This understates every `T`; it does
  not bias the ratios in an obvious direction and is left for Round 2.
