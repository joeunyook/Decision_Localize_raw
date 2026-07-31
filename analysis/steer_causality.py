#!/usr/bin/env python3
"""Is the decodable belief direction CAUSALLY coupled to the action?

For each (model, env) this compares how much the pre-CoT A/B logit gap moves
when the residual is pushed along the trained probe direction, against the same
push along random directions of matched norm. A direction that carries belief
content but is not read by the answer-token circuit moves the gap no more than
a random vector does.

Reports the full dose-response (alpha = 1, 2, 4, 8), because the signature that
matters is a monotone increase with alpha in one family and a flat line in the
other -- a single alpha cannot distinguish "inert" from "under-powered".

Note alpha is scaled by sigma_proj (the std of the projection onto the probe
direction), so the ABSOLUTE perturbation differs per env; sigma is printed so a
null can be checked against perturbation size rather than assumed.

  python analysis/steer_causality.py --tag P1_v4
"""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

FAMILIES = {
    "Bayesian (pure)":      ["bandit", "monty"],
    "Bayesian (EU thresh)": ["trust_game", "machine"],
    "game-theoretic":       ["stag_hunt", "chicken", "bos"],
    "excluded (horizon)":   ["ipd"],
}
MODELS = ["gemma2b", "llama", "phi", "qwen", "dsqwen7",
          "qwen27", "gemma31", "llama70", "dsllama70"]
ALPHAS = [1, 2, 4, 8]


def effect(model, env, steer_dir):
    """mean |delta gap| for probe vs random direction, per alpha."""
    dirs = defaultdict(list)
    rands = defaultdict(list)
    sigma = None
    n_rounds = 0
    for f in glob.glob(os.path.join(steer_dir, f"{model}_{env}_seed*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        sigma = d.get("sigma_proj", sigma)
        for r in d.get("rounds", []):
            g = r.get("gaps") or {}
            base = g.get("baseline")
            if base is None:
                continue
            n_rounds += 1
            for a in ALPHAS:
                if f"rescue_a{a}" in g:
                    dirs[a].append(abs(g[f"rescue_a{a}"] - base))
                for k in g:
                    if k.startswith("rand") and k.endswith(f"_a{a}"):
                        rands[a].append(abs(g[k] - base))
    if not n_rounds:
        return None
    out = {"sigma": sigma, "n": n_rounds, "ratio": {}, "dir": {}}
    for a in ALPHAS:
        if dirs.get(a) and rands.get(a):
            dm, rm = st.mean(dirs[a]), st.mean(rands[a])
            out["dir"][a] = dm
            out["ratio"][a] = dm / rm if rm else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="P1_v4")
    ap.add_argument("--steer-dir", default=None)
    args = ap.parse_args()
    steer_dir = args.steer_dir or f"steer/{args.tag}"

    print(f"Steering effect on the pre-CoT logit gap, relative to random directions")
    print(f"(values are mean|delta gap| / mean|delta gap| under random dirs; 1.0 = inert)")
    print(f"source: {steer_dir}\n")

    fam_rows = defaultdict(list)
    for fam, envs in FAMILIES.items():
        print(f"--- {fam} ---")
        print(f"{'model':11}{'env':12}{'sigma':>8}" + "".join(f"{'a='+str(a):>8}" for a in ALPHAS))
        for m in MODELS:
            for e in envs:
                r = effect(m, e, steer_dir)
                if not r or not r["ratio"]:
                    continue
                row = f"{m:11}{e:12}{(r['sigma'] or 0):>8.2f}"
                row += "".join(f"{r['ratio'].get(a, float('nan')):>8.1f}" for a in ALPHAS)
                print(row)
                if 8 in r["ratio"]:
                    fam_rows[fam].append(r["ratio"][8])
        print()

    print("=" * 62)
    print(f"{'family':24}{'median ratio @a=8':>20}{'n combos':>12}")
    for fam in FAMILIES:
        v = fam_rows.get(fam, [])
        if v:
            print(f"{fam:24}{st.median(v):>20.2f}{len(v):>12}")
    print()
    print("A ratio near 1.0 means pushing along the belief direction changes the")
    print("action logit no more than a random vector of the same norm: the belief")
    print("is decodable but not read by the answer-token circuit.")


if __name__ == "__main__":
    main()
