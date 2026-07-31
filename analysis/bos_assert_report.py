#!/usr/bin/env python3
"""Does BoS reproduce the reactive-opponent signature? (belief high, mapping poor)

Prints, per model, the BoS pre-CoT belief-to-action agreement measured from a
small isolated sweep, next to that model's ALREADY-COLLECTED 100-seed P1_v4
numbers for the two reference groups:

  Bayesian reference   : bandit, monty   (mapping should be intact)
  Reactive reference   : stag_hunt       (mapping should be degraded)

The assertion holds if BoS lands near stag_hunt and well below the Bayesian
reference, with the belief probe still high.

  python analysis/bos_assert_report.py --out-root BOS_ASSERT --tag BOS_ASSERT \
      --models qwen phi gemma31
"""
import argparse
import glob
import json
import os

AMBIG_EPS = 0.10
BAYES_REF = ["bandit", "monty"]
REACTIVE_REF = "stag_hunt"


def pool_by_t(rec, key):
    num = den = 0
    for _, v in rec.items():
        if v.get(key) is None or not v.get("n"):
            continue
        num += v[key] * v["n"]
        den += v["n"]
    return 100 * num / den if den else None


def balanced_agreement(paths):
    """Balanced pre-CoT agreement with the baseline-optimal action.

    decision = (delta_pre_sem > 0) -> model prefers act1; macro-averaged over
    the two baseline-optimal classes so an unbalanced label mix cannot inflate.
    """
    c1 = n1 = c0 = n0 = 0
    for f in paths:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in d.get("rounds", []):
            o = r.get("oracle_p_act1")
            g = r.get("delta_pre_sem")
            if o is None or g is None or abs(o - 0.5) <= AMBIG_EPS:
                continue
            prefers_act1 = g > 0
            if o > 0.5:
                n1 += 1
                c1 += prefers_act1
            else:
                n0 += 1
                c0 += not prefers_act1
    if not (n1 and n0):
        return None, n1, n0
    return 50 * (c1 / n1 + c0 / n0), n1, n0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="BOS_ASSERT")
    ap.add_argument("--tag", default="BOS_ASSERT")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ref-tag", default="P1_v4")
    args = ap.parse_args()

    probe_by_t = json.load(open(f"results/{args.ref_tag}/probe_by_t.json"))
    behav = json.load(open(f"results/{args.ref_tag}/behavior_by_t.json"))["behavior"]

    print("=" * 78)
    print("BoS TREND ASSERTION -- pre-CoT belief-to-action mapping")
    print("=" * 78)
    print(f"{'model':10}{'BoS act':>9}{'BoS bel':>9}   | "
          f"{'Bayes act':>10}{'Bayes bel':>10}   | {'Stag act':>9}{'Stag bel':>9}   {'verdict':>10}")

    verdicts = []
    for m in args.models:
        paths = glob.glob(os.path.join(args.out_root, "responses", f"{m}_bos_seed*.json"))
        bos_act, n1, n0 = balanced_agreement(paths)

        bos_bel = None
        pf = f"results/{args.tag}/probe_{m}_bos.json"
        if os.path.exists(pf):
            p = json.load(open(pf))
            bos_bel = 100 * (p.get("best_accuracy_bal") or p.get("best_accuracy") or 0)

        bay_act = [pool_by_t(behav[f"{m}_{e}"], "agree_bal")
                   for e in BAYES_REF if f"{m}_{e}" in behav]
        bay_bel = [pool_by_t(probe_by_t[f"{m}_{e}"]["per_t"], "acc_bal")
                   for e in BAYES_REF if f"{m}_{e}" in probe_by_t]
        bay_act = sum(bay_act) / len(bay_act) if bay_act else None
        bay_bel = sum(bay_bel) / len(bay_bel) if bay_bel else None

        sh_act = pool_by_t(behav.get(f"{m}_{REACTIVE_REF}", {}), "agree_bal")
        sh_bel = pool_by_t(probe_by_t.get(f"{m}_{REACTIVE_REF}", {}).get("per_t", {}), "acc_bal")

        def f(x):
            return f"{x:.0f}" if x is not None else "--"

        verdict = "?"
        if bos_act is not None and bay_act is not None:
            # holds if BoS mapping is clearly below the Bayesian reference
            verdict = "HOLDS" if bay_act - bos_act >= 8 else "WEAK"
            verdicts.append((m, verdict, bay_act - bos_act))

        print(f"{m:10}{f(bos_act):>9}{f(bos_bel):>9}   | {f(bay_act):>10}{f(bay_bel):>10}   | "
              f"{f(sh_act):>9}{f(sh_bel):>9}   {verdict:>10}")
        if bos_act is None:
            print(f"{'':10}  (no scorable BoS rounds: n_act1={n1}, n_act0={n0}, "
                  f"{len(paths)} session files)")

    print()
    print("Reference columns are the full 100-seed P1_v4 numbers; the BoS column is")
    print("the small isolated sweep. Verdict HOLDS = BoS mapping at least 8pp below")
    print("that model's Bayesian reference, i.e. the reactive-opponent deficit")
    print("reproduces without IPD.")
    if verdicts:
        n_hold = sum(1 for _, v, _ in verdicts if v == "HOLDS")
        mean_drop = sum(d for _, _, d in verdicts) / len(verdicts)
        print()
        print(f"SUMMARY: {n_hold}/{len(verdicts)} models hold; "
              f"mean Bayesian-minus-BoS mapping drop = {mean_drop:.1f}pp")


if __name__ == "__main__":
    main()
