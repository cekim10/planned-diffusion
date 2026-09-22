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
`max(1, int(remaining · unmask_ratio))`. Driving `block_unmask` the same way:

| steps_ratio | `pd_entropy` finish for `[80,80,80,40]` |
|---|---|
| 1.0 | `[80,80,80,40]` — staggered, `finish_k = l_k` (Round 0) |
| 0.5 | `[40,40,40,40]` — **all spans finish together at `og_steps`** |
| 0.25 | `[20,20,20,20]` — same |

Analytically: with `steps = max_l · r`, block `k` transfers `l_k / steps` tokens
per step, so every block empties on the last step. **The staggered natural
retirement found in Round 0 is a `steps_ratio = 1` phenomenon** — the
`max(1, …)` floor forces one token per step and short spans run out early. At
the speed settings the paper's frontier actually uses (`steps_ratio < 1`), the
default algorithm has no span retirement at all, and the only source of
lifetime heterogeneity is the confidence threshold. Round 2 is therefore not a
"second axis" on top of Round 0–1; at speed it is the only axis.

Control expectation, stated in advance: every span finishes at `og_steps`
(`σ_sib = 0`, lifetime-ratio CV = 0, ρ undefined → treated as 1).

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
