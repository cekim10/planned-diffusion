"""Applies the preregistered Round 0 decision rule to pd_audit.py output."""
import argparse, json, sys
import numpy as np


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
    """Fact 2: under lockstep, every block hits zero masks at the same step."""
    checked = early = 0
    details = []
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
            if len(set(finish)) > 1:
                early += 1
                details.append((r["request_id"], rd["span_lengths"], finish))
    return checked, early, details[:5]


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

    ck, early, det = falsification_check(rows)
    if ck:
        print(f"\nlockstep falsification: {ck} fork-join rounds with >=2 spans checked, "
              f"{early} had a span retire early")
        for d in det:
            print(f"   req {d[0]} lengths={d[1]} finish_steps={d[2]}")

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
