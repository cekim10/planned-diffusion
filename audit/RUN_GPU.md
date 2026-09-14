# Round 0 audit — GPU runbook

Everything below is one process per GPU, `bf16`, no `torchrun` needed.
Read `audit/PREREGISTRATION.md` first: the decision rule is fixed and must not be
edited after seeing results.

## 0. Get the code onto the GPU box

```bash
git clone <your-fork-url> planned-diffusion && cd planned-diffusion
```

(The `audit/` directory must be committed and pushed from the laptop first.)

## 1. Environment

```bash
conda create -y -n pd-env python=3.10 && conda activate pd-env
pip install -U pip && pip install -r requirements.txt
```

`audit/pd_audit.py` does **not** import `datasets` — it reads
`tatsu-lab/alpaca_eval`'s `alpaca_eval.json` directly, so the script-based-dataset
breakage on `datasets>=4` cannot bite you.

## 2. Smoke test (3 prompts, ~1 min)

```bash
python audit/pd_audit.py --mode plan --num_samples 3 --out audit/results/smoke.jsonl
```

Expect lines like `[0] ok  k=4  l=[30, 60, 40, 50]  2.1s (plan 2.1s)`.
If `k=` is empty or status is `error`, stop and read the error before going further.

## 3. PRIMARY RUN — all 805 AlpacaEval prompts, default plan (~30-90 min on one A100)

```bash
python audit/pd_audit.py \
  --mode plan \
  --out audit/results/plan_default.jsonl
```

Defaults are the repo's AlpacaEval settings: `temperature=0.2`, `top_p=0.95`,
`max_length=1024`, `steps_ratio=1.0`, seed 42 re-set per prompt,
`length_scale=None` (i.e. the model's own `N * 10`). **Do not pass `--length_scale`
here** — this run is the primary evidence.

Resumable: re-running the same command skips `request_id`s already in the file.

### Multi-GPU sharding (optional)

```bash
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python audit/pd_audit.py --mode plan \
    --start $((g*202)) --num_samples 202 \
    --out audit/results/plan_default.part$g.jsonl &
done; wait
```

`analyze.py` takes multiple files, so pass all four shards to it.

## 4. Apply the preregistered decision rule

```bash
python audit/analyze.py audit/results/plan_default*.jsonl --label "primary (default plan)"
```

This prints median/p25/p75/p90 for `k`, `CV_len`, `imbalance`, `join_waste`,
the `k` histogram, `F2`, aggregate waste, and the **VERDICT**.

## 5. Falsification check — is the join really lockstep? (30 prompts, slower)

```bash
python audit/pd_audit.py \
  --mode full --num_samples 30 \
  --out audit/results/full_30.jsonl
```

```bash
python audit/analyze.py audit/results/full_30.jsonl --label "full generation"
```

Look for the line `lockstep falsification: N rounds ... M had a span retire early`.
**M must be 0.** If M > 0, the plan-derived `join_waste` is wrong and the
preregistration is void — report that and stop.

## 6. Sensitivity — `length_scale` (report only, never primary)

```bash
for ls in 0.5 1.0 2.0; do
  python audit/pd_audit.py --mode plan --num_samples 200 --length_scale $ls \
    --out audit/results/plan_ls$ls.jsonl
  python audit/analyze.py audit/results/plan_ls$ls.jsonl --label "length_scale=$ls"
done
```

Expected: `CV_len`, `imbalance` and `join_waste` are *identical* across
`1.0 / 2.0` and shift only at `0.5` from integer truncation, because
`length_scale` is an exact affine rescale of every `l_k`. Verified on this repo's
own `create_pd_inputs`:

| length_scale | lengths | CV | imbalance | join_waste |
|---|---|---|---|---|
| None (=x10) | [30, 120, 70] | 0.5021 | 1.6364 | 0.3889 |
| 1.0 | [3, 12, 7] | 0.5021 | 1.6364 | 0.3889 |
| 2.0 | [6, 24, 14] | 0.5021 | 1.6364 | 0.3889 |
| 0.5 | [1, 6, 3] | 0.6164 | 1.8000 | 0.4444 |

This is the point to make to a reviewer: the knob **cannot** manufacture the
imbalance, so the primary result does not depend on avoiding it.

## 7. Robustness — does the KV-cache path change the plan? (optional, 50 prompts)

```bash
python audit/pd_audit.py --mode plan --num_samples 50 --use_cache \
  --out audit/results/plan_cache50.jsonl
python audit/analyze.py audit/results/plan_cache50.jsonl --label "use_cache"
```

Span lengths should match the first 50 rows of the primary run. `use_cache`
routes to a *different mixin class* (`DreamGenerationMixinWithCache`) with a
different `_ar_sample` signature, so this is worth confirming rather than assuming.

## What to send back

- `audit/results/*.jsonl`
- the stdout of every `analyze.py` invocation
- the `VERDICT` line from step 4

## Implementation notes (why the instrumentation looks the way it does)

- `DreamModel.__getattribute__` (`dream/modeling_dream.py:870`) intercepts
  `planned_diffusion_generate`, `diffusion_generate`, `_ar_sample`, `_diff_sample`
  and returns the mixin class's method, **ignoring the instance dict**. Patching
  `model._ar_sample` therefore does nothing and fails silently. `instrument()`
  patches the mixin class instead, choosing the right mixin from `use_cache`.
- All latencies are taken after `torch.cuda.synchronize()`; without it the
  reported planning/diffusion split is meaningless on GPU.
- `top_k` is passed as `None`, not `0` — `top_k_logits()` would index an empty
  `torch.topk` result and crash.
