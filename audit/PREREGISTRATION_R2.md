# Round 2 — Are span lifetimes revealed at runtime, or fixed by the plan?

**Status:** preregistered before any Round 2 measurement. Written after Round 1
(GRAY: `S_compact=0.044`, `H=0.177`, `split=0.07`).

## Why this round exists

Rounds 0–1 established that spans have exact natural retirement points
(`finish_k == l_k`, 220/220) but that exploiting them statically is worth ~18%
over request-level batching. The remaining question is whether the *lifetime*
of a span is known at plan time or only revealed during denoising. If the
former, a DAG-aware scheduler solves an offline packing problem with known node
costs and the technical problem is thin. If the latter, the workload is a
**runtime-revealed DAG** and the scheduling problem changes character.

## What the code does — verified without a model

`block_unmask_confidence_threshold` (`dream/pd_utils.py`) gives each block a base
budget of `remaining // (og_steps − i)` tokens per step (floored at 1), always
transfers the top-1, and gates the *additional* candidates on
`confidence ≥ threshold`, deferring failures and extending `steps` at the end.

Consequence, confirmed by driving the function with synthetic confidences:

| steps_ratio | base budget | threshold ever consulted? | finish for `[80,80,80,40]` |
|---|---|---|---|
| 1.0 | 1 token/step for every block | **never** (only the top-1 exists) | `[80,80,80,40]` — identical to `pd_entropy` |
| 0.5 | 2/step for blocks with `l_k > max_l/2`, 1/step otherwise | yes | `[45,47,48,40]`, `[42,45,48,40]`, `[45,47,43,40]` across trials; `steps` 40→48–49 |
| 0.25 | up to 4/step | yes | `[31,32,30,25]`, `[29,30,32,24]`, `[33,29,33,25]`; `steps` 20→33–34 |

**At `steps_ratio=1.0` the confidence threshold is a no-op.** A Round 2 run at
the default `steps_ratio` would have returned the static signature by
construction. The experiment therefore runs at `steps_ratio < 1`, which is also
the regime the paper's speed–quality frontier actually uses.

Control at the same `steps_ratio` is `pd_entropy`, whose per-block schedule is
`max(1, int(remaining · unmask_ratio))`. Its finish rule is
**`finish_k = min(l_k, og_steps)`**, `og_steps = max_l · steps_ratio`: spans
shorter than `og_steps` are held at the one-token-per-step floor and retire at
step `l_k`; the rest finish together on the last step. Deterministic given the
plan, so siblings with the same `l_k` must have identical finish and
`σ_sib = 0`, but **not** lockstep and **not** a constant lifetime ratio — the
ratio `finish/(l_k · r)` ranges from 1 (longest span) to `1/r` (spans at the
floor). Metric 1 therefore does not distinguish control from treatment; metric
2 does.

*Correction record.* The version of this section written before the run said
the control would be lockstep (every span at `og_steps`). That was read off a
synthetic case `[80,80,80,40]` at `r = 0.5`, where the short span sits exactly
at `og_steps = 40` and the floor is invisible. The Round 2 control data
(`finish = min(l_k, og_steps)` on 493/493 spans) corrected it. The rule below
does not use metric 1 and is unaffected.

**Structural consequence for the treatment.** The confidence gate only acts on
candidates beyond the top-1, i.e. on blocks whose base budget
`remaining // (og_steps − i)` is ≥ 2 — the spans with `l_k ≥ og_steps`. Spans at
the floor are untouched by CT by construction, so any dynamic effect is
confined to the long spans of each round.

## Design

Full generation, 40 AlpacaEval prompts strided across all five subsets
(`--stride 20`), `max_length=1024`, defaults otherwise, per-step per-block mask
trajectory recorded (as in Round 0). Four runs:

| run | alg | steps_ratio | threshold |
|---|---|---|---|
| control-0.5 | `pd_entropy` | 0.5 | — |
| **treatment-0.5** (primary) | `pd_confidence_threshold` | 0.5 | 0.9 |
| control-0.25 | `pd_entropy` | 0.25 | — |
| treatment-0.25 | `pd_confidence_threshold` | 0.25 | 0.9 |

`threshold=0.9` is the repo README's example value. The primary comparison is
treatment-0.5 vs control-0.5.

## The four measurements (`audit/ct_analyze.py`)

Over every fork–join round with `k ≥ 2` (multi-round requests contribute one
round each):

1. **Lifetime ratio** `finish_k / (l_k · steps_ratio)`. Static ⇒ one constant.
   Report median, p10/p90, CV, fraction of spans off the static value.
2. **Sibling variance.** Within a round, spans with identical `l_k`: relative
   spread `(max−min)/mean` of their finish steps. This is the direct test —
   same plan-time information, different lifetimes. Static ⇒ 0.
3. **Stagger.** Within a round, `(max−min)/max` of finish steps and the fraction
   of distinct finish steps. (Present in the static case too; reported for the
   retirement-timeline picture, not used in the rule.)
4. **Predictability.** (a) From the plan alone: pooled `R²(finish | l_k)` and
   within-round Spearman `ρ(l_k, finish)`. (b) Online: at fraction `s` of the
   round, estimate each span's finish as `s + remaining(s)/rate_so_far` and
   rank-correlate with the truth, for `s ∈ {0, .1, .25, .5, .75}`. The shape of
   this curve is what "revealed during denoising" means quantitatively.

## Decision rule — fixed in advance (treatment-0.5)

Let `σ_sib` = median sibling relative spread (2), `ρ` = median within-round
rank correlation of `l_k` with finish (4a; a round where all spans finish
together counts as ρ = 1, perfectly predictable).

| condition | verdict |
|---|---|
| no sibling groups | INCONCLUSIVE — rerun with more prompts |
| `σ_sib ≤ 0.02` **and** `ρ ≥ 0.90` | **STATIC** — finish is a function of the plan. Close the PD lead: "confidence threshold" is a static schedule with a different constant. |
| `σ_sib ≥ 0.10` **or** `ρ ≤ 0.70` | **DYNAMIC** — lifetimes are runtime-revealed. Proceed to Round 3 (oracle-vs-online gap), not to a scheduler. |
| otherwise | GRAY |

Pooled `R²` is reported but deliberately not in the rule: with `l_k` spanning
10–200 the between-length variance keeps `R²` high even when siblings diverge.

The control must show every span finishing at `og_steps` (`σ_sib = 0`, CV = 0);
if it does not, the recorder or the reading of `block_unmask` is wrong and the
treatment numbers are not interpretable.

## What Round 2 does not measure

The oracle-vs-online scheduling gap needs contention — multiple requests'
spans competing for packed capacity — and therefore a simulator. Round 2 only
establishes the precondition: that an online scheduler *cannot* know node
lifetimes from the plan. The synthetic-confidence trials above are not evidence
about the model; real confidences may be far more or far less variable than
uniform noise.

## Framing decision carried forward

"40% wasted compute" is retired. The defensible Round 0–1 statement is:
*natural span retirement is real; the incremental capacity opportunity of
static span-level packing over strong request-level batching is ~18%.*

---

## RESULT — 2026-09-21, rule applied as written, with two disclosures

Runs: 40 strided prompts × {control, treatment} × {0.5, 0.25}; 82–90 rounds
each, 46–51 forked. Files `results/r2_*.jsonl`, `results/analysis_round2_sr*.txt`.

### Disclosure 1 — a metric bug changed the primary verdict

The Spearman used for ρ broke ties by array position (`argsort().argsort()`).
In the control, all spans with `l_k ≥ og_steps` finish on the same step, so
their finish ranks were arbitrary; when CT perturbs that block by 1–2 steps the
arbitrary order becomes a different arbitrary order and ρ falls for no reason.
Symptoms visible in the GPU output: a deterministic control with ρ < 1 and an
"online" curve that *declined* over time for a deterministic process. Fixed to
average-rank ties (`rankdata`), re-analysed locally on the same jsonl; the GPU
originals are kept as `analysis_round2_sr*_GPU_ORIGINAL_tiebug.txt`.

| | GPU output (tie bug) | corrected |
|---|---|---|
| sr=0.5 ρ treatment / control | 0.700 / 1.000 | **0.822 / 0.906** |
| sr=0.5 verdict | DYNAMIC (ρ ≤ 0.70, on the boundary) | **GRAY** |
| sr=0.25 ρ treatment / control | 0.819 / 0.991 | **0.538 / 0.812** |
| sr=0.25 verdict | GRAY | **DYNAMIC** |

The rule was not changed; its input was computed correctly.

### Disclosure 2 — the ρ arm of the rule is miscalibrated

The thresholds (`ρ ≥ 0.90` static, `ρ ≤ 0.70` dynamic) were set assuming a
static process gives ρ = 1. It does not at `steps_ratio < 1`: the control's
`finish = min(l_k, og_steps)` has a tied top, so a *perfectly deterministic*
control scores ρ = 0.906 (sr 0.5) and 0.812 (sr 0.25). At sr 0.25 the control
itself would fail the STATIC row. The meaningful comparison is treatment vs
control, reported below; the absolute ρ verdicts should be read with this in
mind. The σ_sib arm is correctly calibrated (control = 0.000 exactly, both ratios).

### Primary, sr = 0.5 — GRAY

| metric | treatment | control |
|---|---|---|
| σ_sib (median relative spread, identical-`l_k` siblings) | **0.019** | 0.000 |
| sibling groups with identical finish | 44% (12/27) | 100% |
| within-round ρ(`l_k`, finish) median / p25 | 0.822 / 0.343 | 0.906 / 0.841 |
| online ρ at s = 0 / .25 / .5 / .75 | 0.816 / 0.816 / 0.816 / 0.816 | 0.906 / 0.906 / 0.906 / 0.906 |

`σ_sib = 0.019` is on the STATIC side of its threshold (0.02); ρ = 0.822 is in
the gray band. Verdict GRAY.

### Secondary, sr = 0.25 — DYNAMIC by the rule

σ_sib = 0.087 (gray band), ρ = 0.538 vs control 0.812 → DYNAMIC via the ρ arm,
with Disclosure 2 applying.

### Magnitude — the number that matters for scheduling

`finish_CT − min(l_k, og_steps)`, in denoising steps:

| | 0 | 1 | 2 | ≥3 | max | `og_steps` median | rounds with any span moved |
|---|---|---|---|---|---|---|---|
| sr 0.5 | 68% | 28% | 3% | 0% | 3 | 20 | 29/48 |
| sr 0.25 | 56% | 27% | 15% | 3% | 5 | 12 | 38/48 |

Within a round, the range of finish among long spans: median 0 (p90 2) at
sr 0.5; median 1 (p90 2, max 5) at sr 0.25. Region extension `steps_run/og`:
median 1.03 (p90 1.06, max 1.20) at sr 0.5. Spans at the floor (`l_k < og`)
are untouched by CT in 100% of cases, as the code predicts.

The hoped-for picture — siblings planned at 80 retiring at 31 / 74 / 45 — does
not occur. What occurs is siblings planned at 40 retiring at 20 / 20 / 21 / 22.

### Timing — is anything revealed *during* denoising?

Shortfall against the static schedule, by decile of a long span's life:

| decile | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 |
|---|---|---|---|---|---|---|---|---|---|---|
| sr 0.5 | 20% | 17% | 15% | 11% | 8% | 5% | 3% | 1% | 2% | 16% |
| sr 0.25 | 12% | 9% | 8% | 5% | 5% | 4% | 3% | 6% | 22% | 26% |

Deferrals happen throughout, but deferred tokens are carried over and mostly
transferred on the next step, so mid-life shortfall does not persist into the
finish; only the tail does. Consequently the online estimator learns nothing:
its ρ is flat across the round (0.816 at every `s`) and no better than the plan
alone (0.822). Lifetime uncertainty here is a **1–2-step tail resolved at the
end**, not a progressively revealed quantity.

### Assessment against the pre-stated criterion

The criterion written before the run: if `finish_k ≈ f(l_k)` and the ranking is
knowable from the plan, close the PD lead. Measured: `finish_k = min(l_k, og) +
{0, 1, 2}` (rarely more), ranking among distinct `l_k` preserved, jitter only
reorders the tied long-span block, nothing learnable before the end. The
confidence threshold makes lifetimes *technically* runtime-dependent, but by an
amount (5–8% of a round) and in a form (unlearnable tail) that does not create
the scheduling problem Round 3 was meant to quantify.

**Recommendation: close Planned Diffusion as the lead workload for
runtime-revealed DAG scheduling.** It remains a clean supporting workload for
the narrower static statement from Rounds 0–1.

### Not tested

`threshold = 0.9` only, one seed, 40 prompts. A more aggressive threshold would
defer more and lengthen the tail; that is a sensitivity axis, and — like
`length_scale` — evidence obtained by turning the knob up is evidence about the
knob, not about the workload. It is not proposed as a path to reopen the lead.
