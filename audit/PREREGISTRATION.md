# Round 0 — Workload Existence Audit (Planned Diffusion)

**Status:** preregistered before any data was collected.
**Repo:** official Planned Diffusion (Israel et al., 2025), commit recorded in `audit/RUN_MANIFEST.md`.

## Primary question

> Do *naturally generated* Planned-Diffusion plans produce substantial
> intra-request execution heterogeneity?

If NO, there is no scheduling opportunity and the broader "intra-request DAG"
line of work stops here regardless of how attractive a wavefront runtime sounds.

## What the execution model actually is (read from source, not assumed)

Planned diffusion is a **fork–join**, not an arbitrary DAG:

```
AR plan  ->  {span_1 ... span_k}  (parallel diffusion)  ->  concat  -> [optional next plan round]
```

Two facts fixed by the implementation:

1. **Span lengths are quantized.** `dream/pd_utils.py` parses a 1–2 digit integer
   `N` out of each `<promise>...N...</topic>` and sets `block_size = N * 10`
   (or `N * length_scale`). So `l_k in {10, 20, ..., 990}`.

2. **The join is already lockstep, and its cost is set by the longest span.**
   `dream/generation_utils.py:484`:
   `generation_config.steps = int(max_block_size * steps_ratio)` where
   `max_block_size = max_k l_k`. `block_unmask` then applies one *global*
   unmask ratio to every block, so every span runs for the same number of steps
   and **no span ever retires early**.

Consequence: per-span finish times carry no information in this implementation,
and the quantity of interest is derivable from the plan alone. We therefore make
the plan the unit of measurement, and use full generation only to *falsify* fact 2.

## Metrics (per request, over the first planning round)

Let `l_1..l_k` be the span lengths emitted by the plan.

- `k` = num_spans
- `CV_len   = std(l) / mean(l)`   (population std)
- `imbalance = max(l) / mean(l)`  in `[1, k]`
- `join_waste = 1 - mean(l)/max(l) = 1 - 1/imbalance`  in `[0, 1)`

`join_waste` is the fraction of span-step slots a lockstep-to-max schedule burns
that an idealized per-span-retiring (wavefront) schedule would not, under the
model's own assumption that span work scales with span length — which is exactly
the assumption the repo encodes at line 484.

Requests with `k = 1` have `join_waste = 0` by definition and contain no fork–join
structure at all; they are reported separately and included in the primary
statistic (excluding them would inflate the result).

## Decision rule — fixed in advance

Primary statistic: **median `join_waste` over all evaluated requests**, call it `W`.
Guard statistic: `F2` = fraction of requests with `k >= 2`.

| Condition | Verdict |
|---|---|
| `F2 < 0.50` | **STOP** — most requests are not even fork–join |
| `W < 0.10`  | **STOP** — DAG exists, scheduler has nothing to eat |
| `0.10 <= W < 0.20` | **GRAY** — not sufficient on its own; needs a second workload before proceeding |
| `W >= 0.20` | **GO** — proceed to simulator (packed CB + critical path vs. dynamic wavefront) |

Secondary descriptors reported but **not** part of the rule: median `k`,
median `CV_len`, distribution of `k`, fraction of multi-round (`<sync>`) requests.

## Evidence hygiene

- **Primary workload is the default, model-generated plan.** `length_scale` is
  left at its default (`None` -> `N * 10`) for the primary run.
- `length_scale in {0.5x, 1x, 2x}` is reported **only** as a sensitivity/stress
  test, never as primary evidence. Note it is a pure affine rescale of every
  `l_k`, so `CV_len`, `imbalance` and `join_waste` are mathematically invariant
  to it up to integer truncation. It therefore *cannot* manufacture the result,
  and we state this explicitly rather than relying on it.
- Sampling params are the repo's AlpacaEval defaults (`temperature=0.2`,
  `top_p=0.95`, fixed seed 42 re-set per prompt, as in `eval/alpaca_eval_diffusion.py`).
- Prompts are AlpacaEval (`tatsu-lab/alpaca_eval`) in dataset order, no cherry-picking.
  Target N = 805 (full set); any short run reports its own N.

## Falsification check on fact 2

On a subset (>= 20 prompts) we run *full* generation and log, at every diffusion
step, the number of remaining `[MASK]` tokens per block. Fact 2 predicts every
block reaches zero at the same step. If any span retires early, the plan-derived
`join_waste` is wrong and this preregistration is void.
