"""
Round 1 -- replay the Round 0 plan distribution through the measured fwd(L) curve.

T_current / T_compact / T_packedCB / T_ideal per request, aggregated, plus the
preregistered rule (audit/PREREGISTRATION_R1.md) and a calibration against the
diffusion latencies actually measured in full_30.jsonl.
"""
import argparse, json, sys
import numpy as np


def load_curve(path):
    d = json.load(open(path))
    c = sorted(d["curve"], key=lambda r: r["L"])
    L = np.array([r["L"] for r in c], float)
    ms = np.array([r["ms_median"] for r in c], float)
    return d, L, ms


def make_fwd(L, ms):
    """ms per forward as a function of total length. Flat below the grid (memory-bound
    floor), linear interpolation inside, linear extrapolation on the last segment above."""
    slope_hi = (ms[-1] - ms[-2]) / (L[-1] - L[-2])

    def fwd(x):
        x = np.asarray(x, float)
        y = np.interp(x, L, ms)
        hi = x > L[-1]
        y = np.where(hi, ms[-1] + slope_hi * (x - L[-1]), y)
        return y
    return fwd


def curve_stats(L, ms):
    tps = L / (ms / 1000.0)
    thr_sat = tps.max()
    sat_idx = int(np.argmax(tps >= 0.9 * thr_sat))
    L_sat = L[sat_idx]
    # linearity above L_sat: ms = a + b*L
    m = L >= L_sat
    if m.sum() >= 3:
        b, a = np.polyfit(L[m], ms[m], 1)
        pred = a + b * L[m]
        r2 = 1 - ((ms[m] - pred) ** 2).sum() / ((ms[m] - ms[m].mean()) ** 2).sum()
        intercept_share = a / ms[m][0]
    else:
        b = a = r2 = intercept_share = float("nan")
    return dict(thr_sat=thr_sat, L_sat=L_sat, r2=r2, intercept_share=intercept_share,
                ms_floor=ms[0], slope_ms_per_tok=b, n_above=int(m.sum()))


def replay_one(P, l, fwd, thr_sat):
    l = np.asarray(l, int)
    S = int(l.max())
    steps = np.arange(1, S + 1)
    live_tok = np.array([(l[l >= s] + 2).sum() for s in steps])      # span k live iff s <= l_k
    full_tok = (l + 2).sum()
    T_cur = S * fwd(P + full_tok)
    T_cmp = fwd(P + live_tok).sum()
    T_cb = S * (P + full_tok) / thr_sat * 1000.0
    T_id = (P + live_tok).sum() / thr_sat * 1000.0
    return T_cur, T_cmp, T_cb, T_id


def run_variant(plans, fwd, thr_sat, cached):
    agg = np.zeros(4); per = []
    for P, l in plans:
        Pe = 0 if cached else P
        t = replay_one(Pe, l, fwd, thr_sat)
        agg += t
        per.append(1 - t[1] / t[0])
    Tc, Tm, Tb, Ti = agg
    return dict(T_current=Tc, T_compact=Tm, T_packedCB=Tb, T_ideal=Ti,
                S_compact=1 - Tm / Tc, H=1 - Ti / Tb,
                split=(Tc - Tm) / (Tc - Ti) if Tc > Ti else float("nan"),
                per_request_S_compact=np.array(per))


def calibrate(full_path, plans_path, fwd):
    """Clean single-round requests from the uncached full run: predicted vs measured
    diffusion time. The prefix P is taken from the plan-mode row for the same
    request_id (same seed => same plan; checked), because full-mode total_tokens
    also counts any runaway AR after a <sync> and cannot be used for P."""
    full = [json.loads(x) for x in open(full_path) if x.strip()]
    plan = {r["request_id"]: r for r in (json.loads(x) for x in open(plans_path) if x.strip())}
    out = []
    for r in full:
        p = plan.get(r["request_id"])
        if r.get("status") != "ok" or len(r.get("rounds", [])) != 1 or p is None:
            continue
        if p.get("plan_terminator") != "eos":          # sync => a 2nd AR phase ran; not a clean round
            continue
        rd = r["rounds"][0]; l = np.array(rd["span_lengths"]); traj = rd.get("mask_trajectory")
        if traj is None or not r.get("diffusion_latency") or l.tolist() != p["rounds"][0]["span_lengths"]:
            continue
        S = len(traj) - 1                                # hook fires once before the loop
        blocks = int((l + 2).sum())
        P = p["prompt_tokens"] + p["plan_token_count"]
        pred = S * float(fwd(P + blocks)) / 1000.0
        out.append((r["request_id"], r["diffusion_latency"], pred, r["diffusion_latency"] / pred, S, P + blocks))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="audit/results/fwd_curve.json")
    ap.add_argument("--plans", default="audit/results/plan_default.jsonl")
    ap.add_argument("--full", default="audit/results/full_30.jsonl")
    ap.add_argument("--k_max", type=int, default=50,
                    help="primary aggregate excludes degenerate plans with k > k_max (PREREGISTRATION_R1.md)")
    args = ap.parse_args()

    meta, L, ms = load_curve(args.curve)
    fwd = make_fwd(L, ms)
    cs = curve_stats(L, ms)

    print(f"\n=== Round 1 compaction headroom  [{meta.get('device')}] ===")
    print(f"fwd(L) curve: {len(L)} points, L={int(L[0])}..{int(L[-1])}")
    print(f"  floor ms(L={int(L[0])})      : {cs['ms_floor']:.2f} ms")
    print(f"  thr_sat                     : {cs['thr_sat']:,.0f} tok/s   L_sat (>=0.9 thr_sat): {int(cs['L_sat'])}")
    print(f"  linear above L_sat (n={cs['n_above']}): R2={cs['r2']:.4f}  intercept/ms(L_sat)={cs['intercept_share']:.2f}  "
          f"slope={cs['slope_ms_per_tok']*1000:.1f} us/tok")
    linear_ok = cs["r2"] >= 0.98 and cs["intercept_share"] <= 0.15
    print(f"  curve linear per rule       : {'YES' if linear_ok else 'NO'}")
    print(f"  {'L':>6} {'ms':>9} {'tok/s':>10}")
    for a, b in zip(L, ms):
        print(f"  {int(a):>6} {b:>9.2f} {a/(b/1000):>10.0f}")

    rows = [json.loads(x) for x in open(args.plans) if x.strip()]
    all_plans = [(r["prompt_tokens"] + r["plan_token_count"], r["rounds"][0]["span_lengths"])
                 for r in rows if r.get("status") == "ok" and r.get("rounds")]
    primary = [(P, l) for P, l in all_plans if len(l) <= args.k_max]
    print(f"\nreplaying round-1 plans: primary n={len(primary)} (k<={args.k_max}), all n={len(all_plans)}")

    def show(v, tag):
        Tc = v["T_current"]
        print(f"\n--- {tag} ---")
        print(f"  {'':<14}{'total':>12}{'rel':>8}")
        for k in ("T_current", "T_compact", "T_packedCB", "T_ideal"):
            print(f"  {k:<14}{v[k]/1000:>10.1f} s{v[k]/Tc:>8.1%}")
        pr = v["per_request_S_compact"]
        print(f"  S_compact = 1 - T_compact/T_current : {v['S_compact']:.3f}   "
              f"(per-request median {np.median(pr):.3f}, p75 {np.percentile(pr,75):.3f}, p90 {np.percentile(pr,90):.3f})")
        print(f"  H         = 1 - T_ideal/T_packedCB  : {v['H']:.3f}")
        print(f"  split: share of (current -> ideal) gap captured by retirement alone: {v['split']:.2f}")

    res = {}
    for cached in (True, False):
        v = run_variant(primary, fwd, cs["thr_sat"], cached)
        res[cached] = v
        show(v, ("prefix CACHED, k<=%d  (PRIMARY)" if cached else "prefix UNCACHED, k<=%d  (repo default path)") % args.k_max)
    v_all = run_variant(all_plans, fwd, cs["thr_sat"], True)
    show(v_all, "prefix CACHED, ALL plans incl. degenerate  (secondary)")

    cal = calibrate(args.full, args.plans, fwd)
    if cal:
        rat = np.array([c[3] for c in cal])
        print(f"\ncalibration vs measured diffusion latency (uncached, clean single-round eos requests, n={len(cal)}):")
        print(f"  measured/predicted: median={np.median(rat):.2f}  p10={np.percentile(rat,10):.2f}  p90={np.percentile(rat,90):.2f}")
        for rid, meas, pred, ratio, S, Ltot in cal[:4]:
            print(f"    req {rid:<4} S={S:<4} L={Ltot:<5} measured={meas:6.2f}s predicted={pred:6.2f}s  x{ratio:.2f}")
        print("  (>1 expected: measured includes per-step sampling/block_unmask/hook overhead)")

    v = res[True]
    print(f"\n--- preregistered decision rule (prefix cached, k<={args.k_max}) ---")
    print(f"S_compact = {v['S_compact']:.3f}   H = {v['H']:.3f}   curve_linear = {linear_ok}")
    verdicts = []
    if v["S_compact"] < 0.05 and v["H"] < 0.10:
        verdicts.append("STOP  -- neither a latency nor a throughput story")
    if v["S_compact"] >= 0.15:
        verdicts.append("GO-latency  -- span retirement alone is worth having")
    if v["H"] >= 0.20:
        verdicts.append("GO-throughput  -- span-level packing beats request-level CB" if linear_ok
                        else "GRAY  -- H >= 0.20 structurally but per-token cost not constant; needs packed-kernel measurement")
    if not verdicts:
        verdicts.append("GRAY")
    for x in verdicts:
        print("VERDICT:", x)
    print()


if __name__ == "__main__":
    main()
