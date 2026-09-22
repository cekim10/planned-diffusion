# Round 3 — Last kill test: does knowing the chain help a strong scheduler?

**Status:** preregistered before the simulator was run. Half-day scope. No GPU.
**If this fails, Planned Diffusion scheduling is closed** — no threshold changes,
no scheduler variants, no workload re-selection to rescue it.

## The two questions, and why only the first is asked

Q1. If a scheduler knew each request's entire remaining chain (all future
    planning and diffusion rounds), could it beat a strong chain-blind scheduler?
Q2. How would an online scheduler learn that chain?

Q2 is worthless if Q1 is NO. Only Q1 is tested here, with a clairvoyant oracle.
No predictor, no critical-path estimator, no online mechanism is built.

## What is known from the data (Rounds 0–2)

- 60% of requests are multi-round (`sync`); given `sync`, 1–11 more rounds, CV 0.74.
- Next-round work is uncorrelated with current-round work (corr 0.06 / −0.15).
- Median 53% of a request's diffusion steps lie in rounds ≥ 2 — invisible at
  admission, revealed exactly when each planning phase completes.
- A request alternates memory-bound AR planning (1 token/iteration) and
  compute-bound diffusion regions (`Σ(l_k+2)` tokens/iteration for
  `max_l·steps_ratio` iterations), ~2.2 rounds on average.

## Simulator (`audit/chain_sim.py`)

Iteration-level batching, one GPU. Iteration time = measured `fwd(Σ tokens)`
(Round 1 curve, L40S, calibrated 1.09×). Assumes packed sequences cost the same
per token as one sequence of that length (same assumption as Round 1; stated).

Request templates: the 106 fully generated requests with round structure
(`full_30`, `r2_control_sr0.5`, `r2_control_sr0.25`; runaway-AR excluded),
resampled with replacement. Each template = rounds of `(AR tokens, span lengths)`.
Region live tokens per step follow the verified rule `finish_k = min(l_k, og)`.
Prefix is KV-cached (AR step = 1 token). Poisson arrivals.

Per iteration: every request in an AR phase is admitted (decode-first, as
Sangam). Diffusion regions are admitted from a ready queue while
`Σ tokens ≤ B` (token budget); a region is indivisible per step and, once
admitted, runs to completion (no preemption). A region becomes *ready* the
instant its planning phase completes — that is when the chain materializes.

Policies differ **only** in the order of the ready queue:

| policy | knows | order |
|---|---|---|
| `fcfs` (Sangam-like) | current region size (needed for admission) | materialization time |
| `sjf_region` (chain-blind, size-aware) | current region size | smallest region work first |
| `srpt_chain` (**oracle**) | entire remaining chain | shortest remaining iterations first |
| `lrpt_chain` (oracle, critical-path) | entire remaining chain | longest remaining first |

`sjf_region` exists so that a gap can be attributed: `srpt_chain − sjf_region`
is the value of chain knowledge; `sjf_region − fcfs` is ordinary size-awareness
any scheduler could have.

Sweeps: `steps_ratio ∈ {0.5, 1.0}`, `B ∈ {512, 1024, 2048}`, arrival rate over a
range that brackets saturation, 2 seeds, 2000 requests each (first 10% discarded).

## Metrics

- **Contention**: fraction of iterations in which the ready queue was non-empty
  after admission (a scheduling choice existed). Also the fraction of regions
  that waited ≥ 1 iteration.
- **P99 and mean E2E latency** per policy, and the relative improvement of each
  oracle over `fcfs`.

## Decision rule — fixed in advance

Evaluated at the loads where `fcfs` P99 is between 1.5× and 4× its unloaded
value (the regime a serving system is actually operated in), `B = 1024` primary,
others reported.

| condition | verdict |
|---|---|
| contention < 10% of iterations at every such load | **KILL** — no scheduling choice exists |
| best oracle P99 improvement over `fcfs` < 15% | **KILL** |
| 15–25% | weak — not a top-tier main idea; record and close |
| ≥ 25% at ≥ 2 loads, **and** `srpt_chain − sjf_region ≥ 10 points` | proceed to Q2 (online mechanism for a progressively materializing graph) |
| ≥ 25% but `sjf_region` captures most of it | the gain is size-awareness, not chain knowledge — **KILL** for the chain thesis |

Both `srpt_chain` and `lrpt_chain` count as "an oracle"; the better one is used.

---

## RESULT — 2026-09-22, rule applied as written

Simulator run locally, no GPU. 106 templates, 1500 requests × 2 seeds per cell,
`steps_ratio ∈ {0.5, 1.0}`, `B ∈ {512, 1024, 2048}`, arrival rate 0.3–1.0 req/s
(capacity ≈ 1.2 req/s). Full sweep: `results/analysis_round3.txt`, `results/chain_sim.json`.

**Window reference.** The rule says "1.5×–4× its unloaded value" for `fcfs` P99.
Unloaded `fcfs` P99 (rate 0.02) is **11.8 s** at sr 0.5 (P50 2.8 s; the tail is the
12-round template) and 20.6 s at sr 1.0, so the window is **18–47 s** (sr 0.5) and
31–82 s (sr 1.0). The simulator's `"x solo"` column compared against the solo
*mean* (3.1 s) and is display only; it has been relabeled.

### Primary cell: sr = 0.5, B = 1024, loads inside the window

| rate | fcfs P99 | contention | `sjf_region` (chain-blind) | `srpt_chain` (oracle) | `lrpt_chain` (oracle) | chain value `srpt − sjf` |
|---|---|---|---|---|---|---|
| 0.5 | 19.6 s | 3.0% | +11.9% | +9.6% | −7.1% | −2.3 pts |
| 0.6 | 23.8 s | 5.7% | +8.2% | +4.3% | −55.6% | −3.9 pts |
| 0.7 | 29.9 s | 10.0% | +19.6% | +1.9% | −100.7% | −17.7 pts |
| 0.8 | 38.4 s | 19.1% | +16.5% | +1.5% | −143.6% | −15.0 pts |

Rule rows:

- Row 1 (contention < 10% at every load): does not fire alone — 10.0% and 19.1% at the two highest loads.
- **Row 2 (best oracle P99 gain < 15%): fires.** Best oracle gain in the window is **9.6%** (`srpt_chain`, rate 0.5); `lrpt_chain` is negative everywhere.
- **Row 5 (`sjf_region` captures most of it): fires.** The chain-blind size-aware policy beats the clairvoyant chain oracle at every load; chain knowledge has **negative** marginal value on P99 (−2 to −18 points).

### **VERDICT: KILL.** Two independent rows fire.

### Does it hold elsewhere?

Every `(sr, B)` shows the same pattern inside its window: `sjf_region ≥ srpt_chain`
on P99, `lrpt_chain` catastrophic. No cell anywhere has `srpt_chain ≥ 25%` or
`srpt − sjf ≥ +10` points. (The single +36 pt cell at B=512, rate 0.8 is outside
the window at 71.6 s and reflects `sjf_region` collapsing there, not `srpt` winning:
−16.8% vs −53.2%.)

Mean latency — SRPT's home objective — does not rescue it: `srpt_chain` loses to
`sjf_region` on mean at every load in every cell (e.g. sr 0.5 B 1024 rate 0.8:
+13.4% vs +19.4%).

### Why, mechanically

P99 is dominated by long chains. Knowing the chain tells the scheduler which
requests are long; the tail-optimal use of that information is to *not* act on it —
SRPT deprioritizes exactly the P99 requests, and critical-path-first (`lrpt`)
starves everything else. Ordering by the immediate region's size, which a
Sangam-like scheduler already knows, is as good or better. The "progressively
materializing graph" is real as a description of the workload, but at the loads a
serving system runs at, contention is 3–19% of iterations and the choice it
offers is not worth more than what request-level batching plus region-size
awareness already gets.

### Not tested, stated

Two seeds; 106 templates resampled; no preemption of admitted regions; packed
per-token cost equal to single-sequence cost (Round 1 assumption); prefix KV
cached, prompt prefill folded into the first AR step; the KV footprint is not
modeled. None of these plausibly turns a −2 to −18 point chain value into +10.

**Planned Diffusion scheduling is closed.** Compaction (R1), static span
scheduling (R1), CT-driven dynamic scheduling (R2), and chain-aware scheduling
(R3) have each been tested against a preregistered rule and failed it.
