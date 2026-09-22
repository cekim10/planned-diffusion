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

The README uses conda; plain `venv` works too and is what the commands below use.

```bash
python3 --version                 # repo targets 3.10; torch 2.7.1 covers 3.9-3.13
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

If `python3 -m venv` fails with an `ensurepip` error (Debian/Ubuntu ships venv
separately): `sudo apt install -y python3-venv`. Without sudo, `pip install --user
virtualenv && python3 -m virtualenv .venv` does the same job.

Check the GPU is visible before running anything:

```bash
nvidia-smi
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`requirements.txt` pins the repo's exact environment, which is what a
preregistered audit wants. If its CUDA wheel pins fight your driver, the audit
itself only needs: `torch transformers==4.49.0 huggingface_hub numpy safetensors`.

`audit/pd_audit.py` does **not** import `datasets` — it reads
`tatsu-lab/alpaca_eval`'s `alpaca_eval.json` directly, so the script-based-dataset
breakage on `datasets>=4` cannot bite you.

## 2. One-shot gating run (recommended)

Stages 2-5 in order, with logs, stopping if the smoke test fails:

```bash
source .venv/bin/activate
bash audit/run_round0.sh
```

(or without activating: `PY=.venv/bin/python bash audit/run_round0.sh`)

Everything it runs must pass **before** a second workload (e.g. SGLang `fork`) is
worth starting. Override with `N_PRIMARY=50` for a quick pass or `N_FALSIFY=10`; the script then
strides across all five AlpacaEval subsets rather than sampling only the first.
The individual stages are documented below.

## 2b. Smoke test (3 prompts, ~1 min)

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

Shard by **stride**, not by contiguous ranges:

```bash
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python audit/pd_audit.py --mode plan \
    --start $g --stride 4 \
    --out audit/results/plan_default.part$g.jsonl &
done; wait
```

`alpaca_eval.json` is stored **ordered by subset** — helpful_base 0-128,
koala 129-284, oasst 285-472, selfinstruct 473-724, vicuna 725-804 — so
contiguous shards each see one or two subsets. That is harmless once all shards
are merged, but it means a shard that dies, or a run you stop early, leaves a
badly skewed sample. Stride sharding keeps every shard proportional, so partial
results stay interpretable. Same reason `--stride` exists for single-GPU partial
runs; `run_round0.sh` sets it automatically when `N_PRIMARY` is given.

`analyze.py` takes multiple files, so pass all four shards to it. It also prints
a per-subset table, which doubles as a coverage check.

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

Look for `plan-derived join_waste is VALID/INVALID`. It is VALID iff every span's
`finish_step == l_k` (AMENDMENT 1 in PREREGISTRATION.md: with `steps_ratio=1`
each block unmasks one token per step and finishes at its declared length; the
region runs to `max_k l_k` and finished spans stay resident). If INVALID, the
length-derived metric does not describe the run — report that and stop.

## 5b. SECOND AXIS — is span lifetime runtime-revealed? (Round 2)

**Do not run `--confidence_threshold` at the default `steps_ratio=1.0`: it is a
no-op there** (base budget is 1 token/step and the top-1 always transfers, so the
threshold is never consulted — verified by driving `block_unmask_confidence_threshold`
directly; see `PREREGISTRATION_R2.md`). The experiment runs at `steps_ratio < 1`
with a `pd_entropy` control at the same ratio.

```bash
bash audit/run_round2.sh
```

Four full-generation runs of 40 strided prompts (control/treatment at
`steps_ratio` 0.5 and 0.25), then `ct_analyze.py` on each pair. Primary verdict is
the 0.5 pair. `N=20` for a quick pass; the sibling-variance test needs forked
rounds with repeated `l_k`, so fewer than ~30 prompts is likely INCONCLUSIVE.

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

## Expected runtime

No GPU numbers are measured yet — these are derived from the execution structure,
so treat them as an order of magnitude and let the smoke test replace them.

At batch 1 and these sequence lengths (~100-300 tokens) every forward pass is
**memory-bound**, not compute-bound: it streams all ~14 GB of bf16 weights.
On an A100-80GB (~1.9 TB/s, ~80% achieved) that is a ~9 ms floor per forward;
with kernel-launch and Python overhead, call it **10-20 ms per forward**.
H100 is roughly 1.7x faster. Sequence length barely matters here.

| step | forwards per prompt | per prompt | total |
|---|---|---|---|
| 3. plan, 805 prompts | `T` = plan tokens, ~30-80 | 0.3-1.6 s | **5-20 min** |
| 5. full, 30 prompts | `T` + `steps` (= `max_k l_k * steps_ratio`, ~30-100) | 0.6-3.6 s | **~2 min** |
| 5b. full + CT, 20 prompts | same, but sync-bound (see warning above) | unknown, likely 10x | **measure first** |
| 6. sensitivity, 3 x 200 | as step 3 | | **5-15 min** |
| 7. use_cache, 50 | as step 3 | | **~1 min** |

Plus a one-off ~15 GB model download and ~1-2 min load per process.

Two things dominate the uncertainty, and both are visible from the smoke test,
which prints per-prompt wall time (`2.1s (plan 2.1s)`):
- **`T`, the plan length.** Unknown until measured. Multiply the smoke test's
  `plan` time by 805 for step 3.
- **`max_k l_k`**, which sets `steps` and therefore all of step 5.

Worst case per prompt: if a plan never emits `<sync>`/`<|im_end|>`, `_ar_sample`
runs to `max_length=1024`, and because it is not KV-cached its cost is quadratic
in `T` — roughly 10-20 s for that one prompt. Bounded, but watch for
`status: ar_hit_max_length` rows in the output.

## Round 1 — compaction headroom kill test (~10-20 min)

Runs only after Round 0 results are in `audit/results/`. Rule is fixed in
`audit/PREREGISTRATION_R1.md`.

```bash
bash audit/run_round1.sh
```

Stage 1 times a real forward pass at `L` = 64..4096 with the repo's block-sparse
PD masks (~5-10 min incl. model load). Stage 2 replays the 804 measured plans
through that curve and prints `T_current / T_compact / T_packedCB / T_ideal`,
`S_compact`, `H`, a calibration against Round 0's measured diffusion latencies,
and the verdict. Shrink the grid with `GRID=64,128,256,512,1024,2048` if short on
time; the top end matters (it sets `thr_sat`).

## What to send back

- `audit/results/*.jsonl`
- the stdout of every `analyze.py` invocation
- the `VERDICT` line from step 4
- the `VALID/INVALID` line from step 5, and the measured spread from 5b
- Round 1: `audit/results/fwd_curve.json` and `analysis_round1.txt`
- Round 2: `audit/results/r2_*.jsonl` and `analysis_round2_sr*.txt`

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
