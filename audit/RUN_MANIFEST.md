# Run manifest

## Authoring / dry validation (no model weights executed)

- repo commit: 5c52637473c1a1f1fe78d87bcc0e407e73881fbc
- repo source untouched: `audit/` is additive only, no file under `dream/`, `eval/`, `train/` modified
- host: Apple M5, 24 GB unified, torch 2.14.0 / MPS, transformers 4.49.0
- date: 2026-09-14T07:50:23Z
- validated without weights:
  - metric definitions + decision rule on 5 synthetic cases (STOP / GRAY / GO / F2-guard / early-retire)
  - span parsing and `length_scale` invariance via the repo's own `create_pd_inputs`
  - MPS/CPU compatibility of the attention-mask update and bf16 SDPA
  - that instance-level patching of `_ar_sample` is silently ignored by
    `DreamModel.__getattribute__`, and that class-level patching works

**audit/PREREGISTRATION.md was written before any model was run.**

## GPU run (fill in on the GPU box)

- host / GPU:
- torch / transformers / CUDA:
- model revision (`huggingface-cli scan-cache` or the snapshot hash):
- date:
- commands run: see audit/RUN_GPU.md
