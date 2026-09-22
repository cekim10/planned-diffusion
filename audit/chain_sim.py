"""
Round 3 -- iteration-level batching simulator for planned-diffusion request chains.
See audit/PREREGISTRATION_R3.md. No GPU: iteration time comes from the measured
fwd(L) curve.
"""
import argparse, functools, heapq, json, random
import numpy as np
print = functools.partial(print, flush=True)

# ---------------------------------------------------------------- data ----------
def load_templates(plan_path, full_paths):
    plan = {json.loads(l)["request_id"]: json.loads(l) for l in open(plan_path)}
    T = []
    for fp in full_paths:
        for line in open(fp):
            r = json.loads(line)
            if r["status"] != "ok" or not r["rounds"]:
                continue
            p = plan.get(r["request_id"])
            if p is None:
                continue
            R = len(r["rounds"])
            if p["plan_terminator"] == "sync" and R == 1 and r.get("total_tokens", 0) >= 1024 - r["prompt_tokens"] - 2:
                continue                                       # runaway AR
            blocks = sum(sum(l + 2 for l in rd["span_lengths"]) for rd in r["rounds"])
            plan1 = p["plan_token_count"]
            rest = max(0, r["total_tokens"] - blocks - (R - 1) - plan1)
            ar = [plan1] + [max(3, round(rest / (R - 1)))] * (R - 1) if R > 1 else [plan1]
            T.append({"prompt": r["prompt_tokens"],
                      "rounds": [{"ar": ar[i], "l": list(rd["span_lengths"])} for i, rd in enumerate(r["rounds"])]})
    return T


def region_profile(l, sr):
    """Live tokens per step for a region, using the verified rule finish_k = min(l_k, og)."""
    l = np.asarray(l); og = max(1, int(l.max() * sr))
    fin = np.minimum(l, og)
    return [int(((l + 2)[fin >= s]).sum()) for s in range(1, og + 1)]


def instantiate(t, sr):
    """Expand a template into a phase list for a given steps_ratio."""
    phases = []
    for i, rd in enumerate(t["rounds"]):
        ar_tok = t["prompt"] if i == 0 else 1        # first AR step also prefills the prompt
        phases.append({"kind": "ar", "steps": rd["ar"], "first_tok": ar_tok})
        phases.append({"kind": "diff", "prof": region_profile(rd["l"], sr)})
    return phases


def phase_iters(ph):
    return ph["steps"] if ph["kind"] == "ar" else len(ph["prof"])


# ------------------------------------------------------------ fwd(L) ------------
def make_fwd(curve_path):
    d = json.load(open(curve_path)); c = sorted(d["curve"], key=lambda r: r["L"])
    L = np.array([r["L"] for r in c], float); ms = np.array([r["ms_median"] for r in c], float)
    slope = (ms[-1] - ms[-2]) / (L[-1] - L[-2])
    def fwd(x):
        if x <= L[0]: return ms[0] / 1000.0
        if x >= L[-1]: return (ms[-1] + slope * (x - L[-1])) / 1000.0
        return float(np.interp(x, L, ms)) / 1000.0
    return fwd


# ------------------------------------------------------------ simulation --------
class Req:
    __slots__ = ("id", "arr", "phases", "pi", "step", "ready_t", "done", "remaining")
    def __init__(self, id, arr, phases):
        self.id, self.arr, self.phases, self.pi, self.step = id, arr, phases, 0, 0
        self.ready_t = arr; self.done = None
        self.remaining = sum(phase_iters(p) for p in phases)

    def cur(self): return self.phases[self.pi]
    def tokens_now(self):
        p = self.cur()
        if p["kind"] == "ar":
            return p["first_tok"] if self.step == 0 else 1
        return p["prof"][self.step]


def simulate(templates, fwd, sr, rate, B, policy, n_req, seed, warm=0.1, overload_q=400, max_iters=2_000_000):
    rng = random.Random(seed)
    # arrivals
    t = 0.0; arrivals = []
    for i in range(n_req):
        t += rng.expovariate(rate)
        arrivals.append(Req(i, t, instantiate(rng.choice(templates), sr)))
    ai = 0
    now = 0.0
    running_ar = []          # requests in AR phase (always admitted)
    running_diff = []        # admitted regions
    ready = []               # regions waiting for admission (heap by priority)
    seq = 0
    contended_iters = 0; total_iters = 0; waited = 0; regions = 0

    def prio(r):
        if policy == "fcfs":        return (r.ready_t, r.id)
        if policy == "sjf_region":  return (sum(r.cur()["prof"]), r.ready_t, r.id)
        if policy == "srpt_chain":  return (r.remaining, r.ready_t, r.id)
        if policy == "lrpt_chain":  return (-r.remaining, r.ready_t, r.id)
        raise ValueError(policy)

    def push_ready(r):
        nonlocal seq; seq += 1
        heapq.heappush(ready, (prio(r), seq, r))

    finished = []
    while ai < n_req or running_ar or running_diff or ready:
        # admit arrivals up to now (they start in AR)
        while ai < n_req and arrivals[ai].arr <= now:
            running_ar.append(arrivals[ai]); ai += 1
        if not (running_ar or running_diff or ready):
            now = arrivals[ai].arr; continue

        # admission: AR always; regions while budget allows
        tokens = sum(r.tokens_now() for r in running_ar) + sum(r.tokens_now() for r in running_diff)
        # Regions are indivisible per step. One that exceeds B on its own can never
        # satisfy tokens + region <= B; Sangam's deficit scheduler admits such a
        # prefill once nothing else competes. Same here: admit an oversized head
        # region when no other region is running, otherwise it would block forever.
        while ready and (tokens + ready[0][2].tokens_now() <= B or not running_diff):
            _, _, r = heapq.heappop(ready)
            if r.ready_t < now: waited += 1
            running_diff.append(r); tokens += r.tokens_now()
        total_iters += 1
        if ready: contended_iters += 1
        if total_iters > max_iters:
            raise RuntimeError(f"iteration cap {max_iters} hit: rate={rate} B={B} policy={policy} (ready={len(ready)})")
        if len(ready) > overload_q:                       # unbounded queue: past capacity
            lat = np.array([r.done - r.arr for r in finished]) if finished else np.array([np.nan])
            return {"p50": float(np.nanpercentile(lat, 50)), "p99": float('inf'), "mean": float('inf'),
                    "contention": contended_iters / total_iters, "regions_waited": waited / max(regions, 1),
                    "iters": total_iters, "makespan": now, "overloaded": True}

        dt = fwd(tokens) if tokens > 0 else fwd(1)
        now += dt

        # advance one step
        nxt_ar, nxt_diff = [], []
        for r in running_ar + running_diff:
            r.step += 1; r.remaining -= 1
            if r.step >= phase_iters(r.cur()):
                r.pi += 1; r.step = 0
                if r.pi >= len(r.phases):
                    r.done = now; finished.append(r); continue
                r.ready_t = now
                if r.cur()["kind"] == "ar": nxt_ar.append(r)
                else: regions += 1; push_ready(r)          # region materializes -> ready queue
            else:
                (nxt_ar if r.cur()["kind"] == "ar" else nxt_diff).append(r)
        running_ar, running_diff = nxt_ar, nxt_diff

    fin = sorted(finished, key=lambda r: r.id)[int(warm * n_req):]
    lat = np.array([r.done - r.arr for r in fin])
    return {"p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99)), "mean": float(lat.mean()),
            "contention": contended_iters / max(total_iters, 1), "regions_waited": waited / max(regions, 1),
            "iters": total_iters, "makespan": now, "overloaded": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="audit/results/fwd_curve.json")
    ap.add_argument("--plans", default="audit/results/plan_default.jsonl")
    ap.add_argument("--full", nargs="+", default=["audit/results/full_30.jsonl", "audit/results/r2_control_sr0.5.jsonl",
                                                   "audit/results/r2_control_sr0.25.jsonl"])
    ap.add_argument("--sr", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--B", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--rates", nargs="+", type=float, default=None, help="arrivals/sec")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="audit/results/chain_sim.json")
    args = ap.parse_args()

    T = load_templates(args.plans, args.full); fwd = make_fwd(args.curve)
    print(f"templates: {len(T)}  rounds/template mean {np.mean([len(t['rounds']) for t in T]):.2f}")
    policies = ["fcfs", "sjf_region", "srpt_chain", "lrpt_chain"]
    out = []
    for sr in args.sr:
        # unloaded reference for the preregistered window: fcfs P99 at negligible load
        solo = np.mean([simulate(T, fwd, sr, 0.02, args.B[0], "fcfs", 1500, s)["p99"] for s in range(args.seeds)])
        rates = args.rates or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        print(f"\n=== steps_ratio={sr}   unloaded fcfs P99 {solo:.1f}s  -> window 1.5-4x = {1.5*solo:.0f}-{4*solo:.0f}s ===")
        for B in args.B:
            print(f"\n  B={B}")
            print(f"  {'rate':>5} {'contend':>8} {'waited':>7} | " + " ".join(f"{p:>11}" for p in policies) + " |  gain vs fcfs (p99): " + " ".join(f"{p:>10}" for p in policies[1:]))
            for rate in rates:
                res = {p: {k: np.mean([simulate(T, fwd, sr, rate, B, p, args.n, s)[k] for s in range(args.seeds)])
                           for k in ("p50", "p99", "mean", "contention", "regions_waited")} for p in policies}
                f = res["fcfs"]
                gains = {p: 1 - res[p]["p99"] / f["p99"] for p in policies[1:]}
                row = {"sr": sr, "B": B, "rate": rate, "solo": solo, **{f"{p}_{k}": res[p][k] for p in policies for k in res[p]}, **{f"gain_{p}": gains[p] for p in gains}}
                out.append(row)
                fmt = lambda v: f"{v:>8.2f}s  " if np.isfinite(v) else f"{'OVERLOAD':>10} "
                print(f"  {rate:>5.2f} {f['contention']:>8.1%} {f['regions_waited']:>7.1%} | " +
                      " ".join(fmt(res[p]['p99']) for p in policies) + " | " +
                      " ".join((f"{gains[p]:>+9.1%} " if np.isfinite(gains[p]) else f"{'n/a':>10} ") for p in policies[1:]) +
                      (("   <- fcfs p99 is %.1fx unloaded" % (f["p99"] / solo)) if np.isfinite(f["p99"]) else ""))
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
