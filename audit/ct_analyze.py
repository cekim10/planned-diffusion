"""
Round 2 -- are span lifetimes revealed at runtime, or fixed by the plan?

Reads full-mode pd_audit.py output (mask_trajectory per round) and answers four
questions per PREREGISTRATION_R2.md, comparing a treatment file (confidence
threshold) against a control file (pd_entropy) at the same steps_ratio:

 1. lifetime ratio      finish_k / l_k                (static => one constant)
 2. sibling variance    spread of finish among spans with the same l_k in a round
 3. stagger             within a round, how spread out retirements are in time
 4. predictability      how well finish rank is known from the plan alone (s=0),
                        and how it becomes known during denoising (s>0)
"""
import argparse, json
import numpy as np


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return np.nan
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    return np.corrcoef(ra, rb)[0, 1]


def load_rounds(path):
    rounds = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        for ri, rd in enumerate(r.get("rounds", [])):
            T = rd.get("mask_trajectory")
            if not T or len(T[0]) < 1:
                continue
            T = np.array(T, float)                       # [steps+1, k]; row 0 = before step 1
            l = np.array(rd["span_lengths"], float)
            fin = np.array([int(np.flatnonzero(T[:, b] == 0)[0]) if (T[:, b] == 0).any() else T.shape[0] - 1
                            for b in range(T.shape[1])], float)
            rounds.append(dict(rid=r["request_id"], ri=ri, l=l, fin=fin, T=T,
                               sr=r.get("steps_ratio", 1.0), ct=r.get("confidence_threshold")))
    return rounds


def q1_lifetime(rounds, sr):
    l = np.concatenate([x["l"] for x in rounds]); f = np.concatenate([x["fin"] for x in rounds])
    ratio = f / (l * sr)                                  # static expectation: exactly 1.0
    return dict(n=len(l), ratio_median=np.median(ratio), ratio_p10=np.percentile(ratio, 10),
                ratio_p90=np.percentile(ratio, 90), ratio_cv=ratio.std() / ratio.mean(),
                frac_off_static=float(np.mean(np.abs(ratio - 1.0) > 1e-9)))


def q2_siblings(rounds):
    """Within a round, spans with identical l_k: spread of their finish steps."""
    groups = []
    for x in rounds:
        for L in np.unique(x["l"]):
            f = x["fin"][x["l"] == L]
            if len(f) >= 2:
                groups.append(dict(l=L, n=len(f), spread=f.max() - f.min(), rel=(f.max() - f.min()) / f.mean(), std=f.std()))
    if not groups:
        return None
    rel = np.array([g["rel"] for g in groups])
    return dict(groups=len(groups), spans=sum(g["n"] for g in groups),
                rel_spread_median=np.median(rel), rel_spread_p90=np.percentile(rel, 90),
                frac_groups_identical=float(np.mean(rel == 0)))


def q3_stagger(rounds):
    out = []
    for x in rounds:
        if len(x["l"]) < 2:
            continue
        f = x["fin"]; out.append(dict(k=len(f), span=(f.max() - f.min()) / f.max(),
                                       n_distinct=len(np.unique(f)) / len(f)))
    if not out:
        return None
    return dict(rounds=len(out), rel_span_median=np.median([o["span"] for o in out]),
                rel_span_p90=np.percentile([o["span"] for o in out], 90),
                distinct_frac_median=np.median([o["n_distinct"] for o in out]))


def q4_predictability(rounds, sr, grid=(0.0, 0.1, 0.25, 0.5, 0.75)):
    """Rank correlation between true finish and (a) l_k at s=0 (plan only), and (b) an
    online estimate at fraction s of the round: s + remaining(s)/rate_so_far."""
    plan_rho, plan_r2 = [], []
    online = {g: [] for g in grid}
    L_all, F_all = [], []
    for x in rounds:
        l, f, T = x["l"], x["fin"], x["T"]
        L_all.append(l); F_all.append(f)
        if len(l) < 2:
            continue
        plan_rho.append(spearman(l, f))
        S = int(f.max())
        for g in grid:
            s = max(1, int(round(g * S)))
            s = min(s, T.shape[0] - 1)
            rem = T[s]; done = T[0] - rem
            rate = np.where(done > 0, done / s, np.nan)
            est = s + rem / rate
            est = np.where(rem == 0, np.minimum(est, s), est)     # already finished
            est = np.where(np.isnan(est), l * sr, est)             # no progress yet -> plan
            online[g].append(spearman(est, f))
    L_all = np.concatenate(L_all); F_all = np.concatenate(F_all)
    if F_all.std() == 0 or L_all.std() == 0:
        # every span finishes at the same step (true lockstep): perfectly predictable
        b, a, r2, resid_rel = 0.0, F_all.mean(), 1.0, 0.0
    else:
        b, a = np.polyfit(L_all, F_all, 1)
        r2 = 1 - ((F_all - (a + b * L_all)) ** 2).sum() / ((F_all - F_all.mean()) ** 2).sum()
        resid_rel = np.sqrt(((F_all - (a + b * L_all)) ** 2).mean()) / F_all.mean()
    return dict(plan_rho_median=np.nanmedian(plan_rho), plan_rho_p25=np.nanpercentile(plan_rho, 25),
                pooled_r2=r2, pooled_resid_rel=resid_rel, slope=b,
                online={g: np.nanmedian(v) for g, v in online.items()}, n_rounds=len(plan_rho))


def report(path, label):
    rounds = load_rounds(path)
    if not rounds:
        print(f"[{label}] no usable rounds in {path}"); return None
    sr = rounds[0]["sr"]; ct = rounds[0]["ct"]
    forked = [x for x in rounds if len(x["l"]) >= 2]
    print(f"\n=== {label}  [{path}] ===")
    print(f"alg={'pd_confidence_threshold thr=%s' % ct if ct is not None else 'pd_entropy'}  steps_ratio={sr}  "
          f"rounds={len(rounds)} (forked {len(forked)})  spans={sum(len(x['l']) for x in rounds)}")

    a = q1_lifetime(rounds, sr)
    print(f"\n1. lifetime ratio finish_k / (l_k * steps_ratio)   [static => 1.000 for all]")
    print(f"   median {a['ratio_median']:.3f}  p10 {a['ratio_p10']:.3f}  p90 {a['ratio_p90']:.3f}  CV {a['ratio_cv']:.3f}  "
          f"spans off the static value: {a['frac_off_static']:.0%}")

    b = q2_siblings(forked)
    print(f"\n2. siblings with identical l_k inside one round   [static => spread 0]")
    if b:
        print(f"   {b['groups']} groups / {b['spans']} spans: relative spread median {b['rel_spread_median']:.3f}  "
              f"p90 {b['rel_spread_p90']:.3f}   groups with identical finish: {b['frac_groups_identical']:.0%}")
    else:
        print("   no sibling groups")

    c = q3_stagger(forked)
    print(f"\n3. stagger within a round  (max-min)/max of finish steps")
    if c:
        print(f"   median {c['rel_span_median']:.3f}  p90 {c['rel_span_p90']:.3f}   "
              f"distinct finish steps / k: median {c['distinct_frac_median']:.2f}")

    d = q4_predictability(forked, sr)
    print(f"\n4. predictability of finish")
    print(f"   from the plan alone: pooled R2(finish | l_k) = {d['pooled_r2']:.3f}  residual/mean = {d['pooled_resid_rel']:.3f}  "
          f"slope {d['slope']:.3f}")
    print(f"   within-round rank corr(l_k, finish): median {d['plan_rho_median']:.3f}  p25 {d['plan_rho_p25']:.3f}   (n={d['n_rounds']})")
    print(f"   online rank corr(estimated finish, true finish) at fraction of round elapsed:")
    print("     " + "  ".join(f"s={g:.2f}: {v:.3f}" for g, v in d["online"].items()))
    return dict(q1=a, q2=b, q3=c, q4=d, sr=sr, ct=ct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", required=True, help="full-mode jsonl with --confidence_threshold")
    ap.add_argument("--control", default=None, help="full-mode jsonl, pd_entropy, same steps_ratio")
    args = ap.parse_args()
    t = report(args.treatment, "TREATMENT")
    c = report(args.control, "CONTROL") if args.control else None

    if t is None:
        return
    print("\n--- preregistered rule (PREREGISTRATION_R2.md) ---")
    sib = t["q2"]["rel_spread_median"] if t["q2"] else float("nan")
    rho = t["q4"]["plan_rho_median"]; cv = t["q1"]["ratio_cv"]
    rho_v = 1.0 if np.isnan(rho) else rho            # nan = all finish together = perfectly predictable
    print(f"sibling relative spread (median) = {sib:.3f}   within-round rho(l,finish) = {rho_v:.3f}   "
          f"lifetime-ratio CV = {cv:.3f}   R2(finish|l) = {t['q4']['pooled_r2']:.3f}")
    if c is not None:
        csib = c["q2"]["rel_spread_median"] if c["q2"] else float("nan")
        print(f"control: sibling spread {csib:.3f}  CV {c['q1']['ratio_cv']:.3f}  "
              f"(control must be ~0 / ~0 for the treatment numbers to mean anything)")
    if np.isnan(sib):
        v = "INCONCLUSIVE -- no sibling groups; need more forked rounds"
    elif sib <= 0.02 and rho_v >= 0.90:
        v = "STATIC  -- finish is a function of the plan; close the PD lead"
    elif sib >= 0.10 or rho_v <= 0.70:
        v = "DYNAMIC -- lifetimes are runtime-revealed; Round 3 = oracle-vs-online gap"
    else:
        v = "GRAY"
    print("VERDICT:", v, "\n")


if __name__ == "__main__":
    main()
