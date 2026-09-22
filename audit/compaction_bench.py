"""
Round 1 -- fwd(L) microbenchmark with the repo's own planned-diffusion masks.

Builds a PD region (prefix + k async blocks) through create_pd_inputs, inverts the
mask exactly as _diff_sample does, and times model(input_ids, attention_mask) at
a grid of total lengths. This is the per-step cost the diffusion loop pays.
See audit/PREREGISTRATION_R1.md.
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dream.modeling_dream import DreamModel
from dream.pd_utils import create_pd_inputs, invert_and_expand_attention_mask
from dream.control_tags import PROMISE_START, PROMISE_END

# Dream tokenizer ids for the digits 0-9 (same table create_pd_inputs parses).
DIGIT = {0: 15, 1: 16, 2: 17, 3: 18, 4: 19, 5: 20, 6: 21, 7: 22, 8: 23, 9: 24}
FILLER = 1000          # any ordinary vocab id; content does not affect timing


def promise_tokens(n_tens):
    """<promise>-<topic> ... N </topic> declaring a block of n_tens*10 tokens."""
    assert 1 <= n_tens <= 99
    return [PROMISE_START, FILLER] + [DIGIT[int(c)] for c in str(n_tens)] + [FILLER, PROMISE_END]


def build_region(prefix_len, block_tens, device):
    """Return (input_ids, inverted_mask) for a PD region with the given blocks."""
    plan = [t for n in block_tens for t in promise_tokens(n)]
    pad = max(0, prefix_len - len(plan))
    prefix = torch.full((1, pad), FILLER, dtype=torch.long)
    ids = torch.cat([prefix, torch.tensor([plan], dtype=torch.long)], dim=1).to(device)
    prev_mask = torch.tril(torch.ones(1, ids.shape[1], ids.shape[1], dtype=torch.bool, device=device))
    full_ids, full_mask, k, block_info = create_pd_inputs(ids, prev_mask, device)
    return full_ids, full_mask, block_info


@torch.no_grad()
def time_forward(model, ids, mask, warmup, reps):
    attn = invert_and_expand_attention_mask(mask, model.dtype)
    for _ in range(warmup):
        model(input_ids=ids, attention_mask=attn).logits
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(input_ids=ids, attention_mask=attn).logits
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {"ms_median": ts[len(ts) // 2], "ms_p10": ts[len(ts) // 10], "ms_p90": ts[(9 * len(ts)) // 10],
            "ms_min": ts[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="dmisrael/planned-diffusion-dream7b-sft-16ep")
    p.add_argument("--out", default="audit/results/fwd_curve.json")
    p.add_argument("--prefix", type=int, default=60, help="prefix tokens (Round 0 median prompt+plan = 60)")
    p.add_argument("--spans", type=int, default=4, help="blocks per region (Round 0 median k>=2 is 3)")
    p.add_argument("--grid", default="64,96,128,160,192,256,320,384,512,640,768,1024,1536,2048,3072,4096",
                   help="target total lengths L")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    assert torch.cuda.is_available(), "Round 1 is a GPU measurement"
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    model = DreamModel.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    name = torch.cuda.get_device_name(0)
    print(f"[bench] {name}  torch {torch.__version__}  dtype {dtype}  prefix={args.prefix} spans={args.spans}", flush=True)

    rows = []
    for target in [int(x) for x in args.grid.split(",")]:
        # fewer spans at the low end so the grid can reach L ~ prefix + 12 (memory-bound floor)
        spans = max(1, min(args.spans, (target - args.prefix) // 12))
        block_total = target - args.prefix - 2 * spans
        per = max(1, round(block_total / spans / 10))               # tens of tokens per block
        per = min(per, 99)
        tens = [per] * spans
        ids, mask, block_info = build_region(args.prefix, tens, device)
        L = ids.shape[1]
        try:
            t = time_forward(model, ids, mask, args.warmup, args.reps)
        except torch.cuda.OutOfMemoryError:
            print(f"  L={L}: OOM, stopping grid", flush=True)
            torch.cuda.empty_cache()
            break
        row = {"L_target": target, "L": L, "spans": spans, "prefix": ids.shape[1] - sum(b[1] - b[0] for b in block_info),
               "block_sizes": [b[2] for b in block_info], **t,
               "tok_per_s": L / (t["ms_median"] / 1000.0)}
        rows.append(row)
        print(f"  L={L:>5}  blocks={row['block_sizes']}  {t['ms_median']:7.2f} ms  "
              f"(p10 {t['ms_p10']:.2f} / p90 {t['ms_p90']:.2f})  {row['tok_per_s']:9.0f} tok/s", flush=True)

    out = {"device": name, "torch": torch.__version__, "dtype": str(dtype), "model": args.model_path,
           "prefix": args.prefix, "spans": args.spans, "warmup": args.warmup, "reps": args.reps,
           "mask": "pd_block_sparse (create_pd_inputs) inverted to additive bf16, as _diff_sample",
           "curve": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
