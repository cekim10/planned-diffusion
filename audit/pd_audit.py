"""
Round 0 -- Workload Existence Audit for Planned Diffusion.

Measures the intra-request execution heterogeneity that the model's *own*
AR planner produces on AlpacaEval. See audit/PREREGISTRATION.md.

Two modes:
  --mode plan      (default) run only the AR planning stage, parse the emitted
                   spans with the repo's own create_pd_inputs(). Cheap: this is
                   sufficient for every preregistered metric, because
                   generation_utils.py:484 ties the whole fork-join region's
                   step count to max_k l_k.
  --mode full      run the complete planned_diffusion_generate and additionally
                   log, per diffusion step, the remaining [MASK] count per block.
                   Used to falsify the lockstep-join assumption above.

Device-agnostic (cuda / mps / cpu). Resumable: appends JSONL, skips done ids.
"""
import argparse, json, os, sys, time, contextlib
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dream.generation_utils as GU
import dream.generation_utils_cached_pd as GUC
from dream.modeling_dream import DreamModel
from dream.control_tags import MASK_TOKEN_ID, SYNC_TOKEN_ID, EOS_TOKEN_ID
from transformers import AutoTokenizer


def pick_device(explicit=None):
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device, explicit=None):
    if explicit:
        return getattr(torch, explicit)
    return torch.float32 if device.type == "cpu" else torch.bfloat16


def sync(device):
    """CUDA/MPS kernels are async; without this every latency we report is a lie."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def gen_module(use_cache):
    """DreamModel.__getattribute__ (modeling_dream.py:870) routes the four generation
    methods to a mixin CLASS chosen by config.use_cache, bypassing the instance dict.
    So instrumentation must patch the class, and must patch the right one."""
    return (GUC, GUC.DreamGenerationMixinWithCache) if use_cache else (GU, GU.DreamGenerationMixin)


class Recorder:
    """Captures plan structure and (in full mode) the per-step unmask trajectory."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.rounds = []          # one entry per planning round
        self.block_info = None
        self.traj = []            # [[remaining_masks_per_block], ...] for current round
        self.ar_time = 0.0
        self.diff_time = 0.0

    def on_plan(self, num_promises, block_info):
        self.block_info = block_info
        self.traj = []
        self.rounds.append({
            "num_spans": num_promises,
            "span_lengths": [bs for _, _, bs in block_info],
        })

    def on_diff_step(self, x):
        if self.block_info is None:
            return
        # one device->host sync per step, not one per block
        counts = torch.stack([(x[0, s:e] == MASK_TOKEN_ID).sum()
                              for s, e, _ in self.block_info])
        self.traj.append([int(v) for v in counts.tolist()])

    def close_round(self):
        if self.rounds and self.traj:
            self.rounds[-1]["mask_trajectory"] = self.traj
        self.traj = []
        self.block_info = None   # AR steps of the next round must not be recorded


@contextlib.contextmanager
def instrument(model, rec, use_cache, device):
    """Patch create_pd_inputs / _ar_sample / _diff_sample on the mixin class.

    NOTE: patching the *instance* (model._ar_sample = ...) is silently ignored,
    because DreamModel.__getattribute__ intercepts these four names and always
    returns the mixin's method. That would leave mask_trajectory empty and every
    latency at 0 without raising anything.
    """
    mod, cls = gen_module(use_cache)
    orig_create = mod.create_pd_inputs
    orig_ar, orig_diff = cls._ar_sample, cls._diff_sample

    def traced_create(*a, **kw):
        out = orig_create(*a, **kw)
        rec.on_plan(out[2], out[3])
        return out

    def traced_ar(self, *a, **kw):
        t0 = time.perf_counter()
        out = orig_ar(self, *a, **kw)
        sync(device)
        rec.ar_time += time.perf_counter() - t0
        return out

    def traced_diff(self, *a, **kw):
        t0 = time.perf_counter()
        out = orig_diff(self, *a, **kw)
        sync(device)
        rec.diff_time += time.perf_counter() - t0
        rec.close_round()
        return out

    mod.create_pd_inputs = traced_create
    cls._ar_sample = traced_ar
    cls._diff_sample = traced_diff
    try:
        yield
    finally:
        mod.create_pd_inputs = orig_create
        cls._ar_sample = orig_ar
        cls._diff_sample = orig_diff


def run_plan_only(model, inputs, args, rec, device, tok=None):
    """Replicates the AR-plan half of planned_diffusion_generate, then stops.

    The cached mixin's _ar_sample takes an extra past_key_values arg and returns a
    3-tuple, so the call has to branch on use_cache.
    """
    ident = lambda step, x, logits: x
    ident_l = lambda step, x, logits: logits

    t0 = time.perf_counter()
    if args.use_cache:
        seq, mask, _ = model._ar_sample(
            inputs, None, None, args.max_length,
            args.temperature, args.top_p, None,
            generation_tokens_hook_func=ident, generation_logits_hook_func=ident_l)
    else:
        seq, mask = model._ar_sample(
            inputs, None, args.max_length,
            args.temperature, args.top_p, None,
            generation_tokens_hook_func=ident, generation_logits_hook_func=ident_l)
    sync(device)
    rec.ar_time = time.perf_counter() - t0

    if mask is None:
        return {"status": "ar_hit_max_length", "plan_token_count": int(seq.shape[1] - inputs.shape[1])}

    final_id = int(seq[0, -1].item())
    body, mask = seq[:, :-1], mask[:, :-1, :-1]

    _, _, num_promises, block_info = GU.create_pd_inputs(
        input_ids=body, prev_attention_mask=mask, device=body.device,
        length_scale=args.length_scale,
        disable_block_sparsity=args.disable_block_sparsity,
    )
    rec.on_plan(num_promises, block_info)
    out = {
        "status": "ok",
        "plan_token_count": int(seq.shape[1] - inputs.shape[1]),
        "plan_terminator": "sync" if final_id == SYNC_TOKEN_ID else ("eos" if final_id == EOS_TOKEN_ID else str(final_id)),
    }
    if tok is not None:
        out["plan_text"] = tok.decode(seq[0, inputs.shape[1]:].tolist(), skip_special_tokens=False)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="dmisrael/planned-diffusion-dream7b-sft-16ep")
    p.add_argument("--out", default="audit/results/plan_default.jsonl")
    p.add_argument("--mode", choices=["plan", "full"], default="plan")
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--steps_ratio", type=float, default=1.0)
    p.add_argument("--confidence_threshold", type=float, default=None,
                   help="Switch the diffusion alg to pd_confidence_threshold. Unlike "
                        "pd_entropy this makes per-span work data-dependent, so spans "
                        "can retire at different steps (full mode only).")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--length_scale", type=float, default=None)
    p.add_argument("--disable_block_sparsity", action="store_true")
    p.add_argument("--use_cache", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    print(f"[audit] device={device} dtype={dtype} mode={args.mode} length_scale={args.length_scale}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True,
        use_cache=args.use_cache, disable_block_sparsity=args.disable_block_sparsity,
    ).to(device).eval()

    # Load AlpacaEval straight from its data file. The `datasets` loader for this
    # repo is a script-based dataset (needs trust_remote_code and is broken on
    # datasets>=4); alpaca_eval.py only reads this exact json, so the data is
    # identical and no third-party code has to be executed.
    from huggingface_hub import hf_hub_download
    eval_set = json.load(open(hf_hub_download(
        "tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["request_id"])
                except Exception:
                    pass
    print(f"[audit] {len(done)} already done in {args.out}", flush=True)

    lo = args.start
    hi = len(eval_set) if args.num_samples is None else min(len(eval_set), lo + args.num_samples)

    for i in range(lo, hi):
        if i in done:
            continue
        ex = eval_set[i]
        torch.manual_seed(args.random_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.random_seed)

        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": ex["instruction"]}]
        inputs = tok.apply_chat_template(messages, return_tensors="pt",
                                         add_generation_prompt=True, tokenize=True).to(device)

        rec = Recorder()
        t0 = time.perf_counter()
        try:
            if args.mode == "plan":
                meta = run_plan_only(model, inputs, args, rec, device, tok)
            else:
                with instrument(model, rec, args.use_cache, device):
                    hook = lambda step, x, logits: (rec.on_diff_step(x), x)[1]
                    out = model.planned_diffusion_generate(
                        inputs, max_length=args.max_length, steps_ratio=args.steps_ratio,
                        return_dict_in_generate=True,
                        alg=("pd_confidence_threshold" if args.confidence_threshold is not None
                             else "pd_entropy"),
                        threshold=args.confidence_threshold,
                        temperature=args.temperature, top_p=args.top_p, alg_temp=0.,
                        length_scale=args.length_scale,
                        generation_tokens_hook_func=hook,
                    )
                sync(device)
                meta = {"status": "ok",
                        "total_tokens": int(out.sequences.shape[1] - inputs.shape[1])}
        except Exception as e:
            meta = {"status": "error", "error": f"{type(e).__name__}: {e}"}

        wall = time.perf_counter() - t0
        row = {
            "request_id": i,
            "instruction": ex["instruction"],
            "prompt_tokens": int(inputs.shape[1]),
            "total_request_latency": wall,
            "planning_latency": rec.ar_time,
            "diffusion_latency": rec.diff_time,
            "rounds": rec.rounds,
            "length_scale": args.length_scale,
            "steps_ratio": args.steps_ratio,
            "confidence_threshold": args.confidence_threshold,
            "mode": args.mode,
            **meta,
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

        spans = rec.rounds[0]["span_lengths"] if rec.rounds else []
        print(f"[{i}] {meta['status']:<18} k={len(spans):<3} l={spans} "
              f"{wall:.1f}s (plan {rec.ar_time:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
