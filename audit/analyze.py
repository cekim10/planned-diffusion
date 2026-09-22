"""Applies the preregistered Round 0 decision rule to pd_audit.py output."""
import argparse, json, sys
import numpy as np


# tatsu-lab/alpaca_eval's alpaca_eval.json is stored ORDERED BY SUBSET, in these
# contiguous blocks. Recorded here so that runs predating the "subset" field (and
# any contiguous partial run) can still be broken down and checked for bias.
SUBSET_RANGES = [("helpful_base", 0, 128), ("koala", 129, 284), ("oasst", 285, 472),
                 ("selfinstruct", 473, 724), ("vicuna", 725, 804)]


def subset_of(request_id, explicit=None):
    if explicit:
        return explicit
    for name, a, b in SUBSET_RANGES:
        if a <= request_id <= b:
            return name
    return "?"


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def per_request(rows):
    out, skipped = [], []
    for r in rows:
        if r.get("status") != "ok" or not r.get("rounds"):
            skipped.append(r)
            continue
        l = np.array(r["rounds"][0]["span_lengths"], dtype=float)
        if l.size == 0:
            skipped.append(r)
            continue
        mean, mx = l.mean(), l.max()
        out.append({
            "request_id": r["request_id"],
            "subset": subset_of(r["request_id"], r.get("subset")),
            "k": int(l.size),
            "lengths": l.tolist(),
            "cv": float(l.std() / mean) if mean > 0 else 0.0,
            "imbalance": float(mx / mean) if mean > 0 else 1.0,
            "join_waste": float(1.0 - mean / mx) if mx > 0 else 0.0,
            "total_len": float(l.sum()),
            "n_rounds": len(r["rounds"]),
            "terminator": r.get("plan_terminator"),
            "planning_latency": r.get("planning_latency"),
            "diffusion_latency": r.get("diffusion_latency"),
        })
    return out, skipped


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def falsification_check(rows):
    """Per-block finish steps from the logged mask trajectories.

    Under alg=pd_entropy the join is lockstep by construction, so every block must
    reach zero masks on the same step; any early retirement voids the preregistration.
    Under alg=pd_confidence_threshold per-span work is data-dependent, so early
    retirement is EXPECTED and its spread is the measured (not derived) join waste.
    """
    checked = early = 0
    details, measured = [], []
    for r in rows:
        for rd in r.get("rounds", []):
            traj = rd.get("mask_trajectory")
            if not traj or len(traj[0]) < 2:
                continue
            T = np.array(traj)                      # [steps, blocks]
            finish = []
            for b in range(T.shape[1]):
                z = np.flatnonzero(T[:, b] == 0)
                finish.append(int(z[0]) if z.size else T.shape[0] - 1)
            checked += 1
            f = np.array(finish, dtype=float) + 1.0     # steps are 0-indexed
            if f.max() > 0:
                measured.append(1.0 - f.mean() / f.max())
            if len(set(finish)) > 1:
                early += 1
                details.append((r["request_id"], rd["span_lengths"], finish))
    return checked, early, details[:5], np.array(measured)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = load(args.files)
    recs, skipped = per_request(rows)
    if not recs:
        print("no usable records"); sys.exit(1)

    ks = np.array([r["k"] for r in recs])
    cv = np.array([r["cv"] for r in recs])
    imb = np.array([r["imbalance"] for r in recs])
    jw = np.array([r["join_waste"] for r in recs])

    N = len(recs)
    F2 = float((ks >= 2).mean())
    W = float(np.median(jw))

    tag = f" [{args.label}]" if args.label else ""
    print(f"\n=== Round 0 Workload Existence Audit{tag} ===")
    print(f"N evaluated          : {N}   (skipped/errored: {len(skipped)})")
    print(f"mode / length_scale  : {rows[0].get('mode')} / {rows[0].get('length_scale')}")
    print()
    print(f"{'metric':<22}{'median':>10}{'p25':>10}{'p75':>10}{'p90':>10}{'mean':>10}")
    for name, a in (("num_spans k", ks), ("CV_len", cv), ("imbalance max/mean", imb), ("join_waste", jw)):
        print(f"{name:<22}{np.median(a):>10.3f}{q(a,25):>10.3f}{q(a,75):>10.3f}{q(a,90):>10.3f}{a.mean():>10.3f}")

    print(f"\nk distribution       : ", end="")
    vals, cnts = np.unique(ks, return_counts=True)
    print("  ".join(f"k={v}:{c}({c/N:.0%})" for v, c in zip(vals, cnts)))
    print(f"fraction k>=2  (F2)  : {F2:.3f}")
    terms = [r["terminator"] for r in recs if r["terminator"]]
    if terms:
        print(f"plan terminator      : " + "  ".join(
            f"{t}:{terms.count(t)}({terms.count(t)/len(terms):.0%})" for t in sorted(set(terms))))
    print(f"span len quantization: unique lengths seen = {sorted(set(int(x) for r in recs for x in r['lengths']))[:20]}")

    # per-subset breakdown: AlpacaEval is subset-ordered, so this shows both how
    # complete the run is and whether the headline is driven by one subset.
    print(f"\nby subset (coverage and per-subset rule inputs):")
    print(f"  {'subset':<14}{'n':>6}{'/total':>8}{'F2':>8}{'median W':>11}{'median k':>10}")
    for name, a, b in SUBSET_RANGES:
        sel = [r for r in recs if r["subset"] == name]
        if not sel:
            print(f"  {name:<14}{0:>6}{b-a+1:>8}{'-':>8}{'-':>11}{'-':>10}")
            continue
        sk = np.array([r["k"] for r in sel]); sw = np.array([r["join_waste"] for r in sel])
        print(f"  {name:<14}{len(sel):>6}{b-a+1:>8}{(sk>=2).mean():>8.2f}"
              f"{np.median(sw):>11.3f}{np.median(sk):>10.1f}")

    # --- descriptive only, NOT part of the preregistered rule ---
    tot_useful = sum(r["total_len"] for r in recs)
    tot_sched  = sum(r["k"] * max(r["lengths"]) for r in recs)
    W_agg = 1.0 - tot_useful / tot_sched if tot_sched else 0.0
    print(f"\naggregate capacity waste (descriptive, not the rule):")
    print(f"  sum l_k                  = {tot_useful:,.0f} span-tokens")
    print(f"  sum k*max(l_k) (lockstep)= {tot_sched:,.0f} span-token-slots")
    print(f"  W_agg                    = {W_agg:.3f}")

    # join_waste as a function of k
    print(f"\njoin_waste by k:")
    for kv in sorted(set(ks)):
        sub = jw[ks == kv]
        print(f"  k={kv:<3} n={len(sub):<5} median W={np.median(sub):.3f}  mean W={sub.mean():.3f}")

    # histogram of join_waste
    print(f"\njoin_waste histogram:")
    edges = np.arange(0, 1.01, 0.1)
    h, _ = np.histogram(jw, bins=edges)
    for i in range(len(h)):
        bar = "#" * int(60 * h[i] / max(h.max(), 1))
        print(f"  [{edges[i]:.1f},{edges[i+1]:.1f})  {h[i]:>5}  {bar}")

    ck, early, det, meas = falsification_check(rows)
    if ck:
        ct = rows[0].get("confidence_threshold")
        alg = "pd_confidence_threshold" if ct is not None else "pd_entropy"
        print(f"\nper-span finish steps ({ck} fork-join rounds with >=2 spans, alg={alg}):")
        print(f"  rounds where a span retired early: {early}/{ck}")
        for d in det:
            print(f"   req {d[0]} lengths={d[1]} finish_steps={d[2]}")
        if ct is None:
            print("  interpretation: alg=pd_entropy is lockstep by construction, so this")
            print("                  MUST be 0/N. Anything else voids the preregistration.")
        else:
            print(f"  measured join_waste (1 - mean(finish)/max(finish)):")
            print(f"     median={np.median(meas):.3f}  p25={q(meas,25):.3f}  "
                  f"p75={q(meas,75):.3f}  mean={meas.mean():.3f}")
            print("  interpretation: under pd_confidence_threshold per-span work is")
            print("                  data-dependent, so early retirement is expected and")
            print("                  this spread is MEASURED, not derived from lengths.")

    # --- preregistered decision rule ---
    print("\n--- preregistered decision rule ---")
    print(f"F2 = {F2:.3f}   W (median join_waste) = {W:.3f}")
    if F2 < 0.50:
        v = "STOP  -- most requests are not even fork-join (F2 < 0.50)"
    elif W < 0.10:
        v = "STOP  -- DAG exists but scheduler has nothing to eat (W < 0.10)"
    elif W < 0.20:
        v = "GRAY  -- insufficient alone; needs a second workload (0.10 <= W < 0.20)"
    else:
        v = "GO    -- proceed to simulator (W >= 0.20)"
    print(f"VERDICT: {v}\n")


if __name__ == "__main__":
    main()
