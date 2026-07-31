#!/usr/bin/env python3
"""Reviewer-defense diagnostics for the causal-steering claim (paper Result 2).

Regenerates every number in DOCUMENTATION.txt "SESSION LOG -- 2026-07-29".
Reads only steer/{tag}/*.json, so it is CPU-only and needs no GPU or residuals.

It reports the same experiment under BOTH partitions the paper has used, so the
main text and the appendix can be checked against each other:

  TASK partition       4 Bayesian inference  vs  4 game theoretic
  REACTIVITY partition 5 nonreactive         vs  3 opponent reactive

and then the four checks a referee will run:

  FLOOR       is rescue just flipping hairline errors? -> |baseline gap| dist
              on wrong rounds, and rescue within matched |gap| bins
  MAGNITUDE   is rescue just a bigger push? -> mean |delta gap| probe vs random
              (NOTE: the probe direction moves the logits LESS than random)
  SPECIFICITY does the direction damage correct rounds? -> rescue/anti/random
              damage rates across the full dose-response
  POOLING     is the pooled rate one model or one scenario? -> per-model and
              per-scenario breakdowns with session-level bootstrap CIs

  python3 analysis/steer_defense.py --tag P1_v4
"""
import argparse
import glob
import json
import os
import random
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = ["gemma2b", "llama", "phi", "qwen", "dsqwen7",
          "qwen27", "gemma31", "llama70", "dsllama70"]
# task partition (main text)
BAYES = ["bandit", "monty", "trust_game", "machine"]
GAMES = ["chicken", "stag_hunt", "bos", "pennies"]
# reactivity partition (appendix): chicken's opponent is iid given its type
NONREACT = ["bandit", "monty", "trust_game", "machine", "chicken"]
REACT = ["stag_hunt", "bos", "pennies"]
DROPPED = ["ipd"]          # horizon confound, excluded from the paper
ALPHAS = [1.0, 2.0, 4.0, 8.0]
A = 4.0                    # the alpha quoted in the paper
GAP_BINS = [(0, .25), (.25, .5), (.5, 1), (1, 2), (2, 1e9)]
N_BOOT = 2000


# ------------------------------------------------------------------ loading

def load(tag):
    """env -> list of unambiguous round dicts, annotated with model and session."""
    data = defaultdict(list)
    for path in glob.glob(os.path.join(ROOT, "steer", tag, "*_seed*.json")):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if d["model_key"] not in MODELS:
            continue
        for r in d.get("rounds", []):
            if not r.get("unambiguous", True):
                continue
            if "baseline" not in (r.get("gaps") or {}):
                continue
            r["_sess"] = (d["model_key"], d["env"], d["seed"])
            r["_model"] = d["model_key"]
            data[d["env"]].append(r)
    return data


# ------------------------------------------------------------------ metrics

def ok(gap, gt_act1):
    return (gap > 0) == gt_act1


def wrong(rows):
    return [r for r in rows if not ok(r["gaps"]["baseline"], r["gt_act1"])]


def right(rows):
    return [r for r in rows if ok(r["gaps"]["baseline"], r["gt_act1"])]


def rate(rows, key):
    """% of rows whose gap under condition `key` agrees with the oracle."""
    if not rows:
        return float("nan")
    return 100 * sum(ok(r["gaps"].get(key, r["gaps"]["baseline"]), r["gt_act1"])
                     for r in rows) / len(rows)


def rand_rate(rows, a, n_random=2):
    if not rows:
        return float("nan")
    return st.mean([rate(rows, f"rand{j}_a{a:g}") for j in range(n_random)])


def diff(rows, a=A):
    return rate(rows, f"rescue_a{a:g}") - rand_rate(rows, a)


def boot_ci(rows, fn=diff, n_boot=N_BOOT, seed=0):
    """Percentile CI resampling whole SESSIONS (10 rounds share a hidden param)."""
    sess = defaultdict(list)
    for r in rows:
        sess[r["_sess"]].append(r)
    keys, rng, vals = list(sess), random.Random(seed), []
    if not keys:
        return float("nan"), float("nan")
    for _ in range(n_boot):
        rs = [r for k in (rng.choice(keys) for _ in keys) for r in sess[k]]
        if rs:
            vals.append(fn(rs))
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def gap_stats(rows):
    g = sorted(abs(r["gaps"]["baseline"]) for r in rows)
    q = lambda p: g[int(p * (len(g) - 1))]
    return q(.25), q(.5), q(.75), st.mean(g)


# ------------------------------------------------------------------ reports

def family_block(name, rows):
    w, c = wrong(rows), right(rows)
    lo, hi = boot_ci(w)
    print(f"\n{name}   n_rounds={len(rows)}  n_wrong={len(w)}  n_correct={len(c)}")
    print(f"  alpha={A:g}: rescue {rate(w, f'rescue_a{A:g}'):5.1f}%   "
          f"random {rand_rate(w, A):5.1f}%   diff {diff(w):+5.1f} pp  [{lo:+.1f},{hi:+.1f}]")
    print("  dose-response on WRONG rounds (% recovered)")
    print("    rescue    " + " ".join(f"a{a:g}={rate(w, f'rescue_a{a:g}'):5.1f}%" for a in ALPHAS))
    print("    random    " + " ".join(f"a{a:g}={rand_rate(w, a):5.1f}%" for a in ALPHAS))
    print("    anti      " + " ".join(f"a{a:g}={rate(w, f'anti_a{a:g}'):5.1f}%" for a in ALPHAS))
    print("  damage on CORRECT rounds (% flipped to wrong)")
    print("    rescue    " + " ".join(f"a{a:g}={100 - rate(c, f'rescue_a{a:g}'):5.1f}%" for a in ALPHAS))
    print("    anti      " + " ".join(f"a{a:g}={100 - rate(c, f'anti_a{a:g}'):5.1f}%" for a in ALPHAS))
    print("    random    " + " ".join(f"a{a:g}={100 - rand_rate(c, a):5.1f}%" for a in ALPHAS))
    p25, med, p75, mean = gap_stats(w)
    print(f"  |baseline gap| on wrong rounds: p25 {p25:.3f}  median {med:.3f}  "
          f"p75 {p75:.3f}  mean {mean:.3f}  (nats)")
    print("  mean |delta gap| probe vs random, all unambiguous rounds")
    for a in ALPHAS:
        md = st.mean([abs(r["gaps"][f"rescue_a{a:g}"] - r["gaps"]["baseline"]) for r in rows])
        mr = st.mean([abs(r["gaps"][f"rand{j}_a{a:g}"] - r["gaps"]["baseline"])
                      for r in rows for j in range(2)])
        print(f"    a={a:g}  probe {md:6.3f}   random {mr:6.3f}   ratio {md / mr:5.2f}")


def bin_block(groups):
    print("\nRESCUE WITHIN MATCHED |baseline gap| BINS (rescue% / random%, n)")
    print(f"  {'bin (nats)':>14} " + "".join(f"{g[:22]:>26}" for g in groups))
    for lo_b, hi_b in GAP_BINS:
        line = f"  [{lo_b:5.2f},{hi_b if hi_b < 1e8 else 99:5.2f}) "
        for rows in groups.values():
            w = [r for r in wrong(rows) if lo_b <= abs(r["gaps"]["baseline"]) < hi_b]
            line += f"{rate(w, f'rescue_a{A:g}'):9.1f}% /{rand_rate(w, A):6.1f}% n={len(w):<5d}"
        print(line)


def per_key_block(title, keyed):
    print(f"\n{title}  (alpha={A:g})")
    print(f"  {'key':14}{'rescue':>9}{'random':>9}{'diff':>8}{'95% CI':>18}{'n_wrong':>9}")
    for k, rows in keyed.items():
        w = wrong(rows)
        if not w:
            print(f"  {k:14}{'-':>9}{'-':>9}{'-':>8}{'(no wrong rounds)':>18}{0:>9}")
            continue
        lo, hi = boot_ci(w)
        print(f"  {k:14}{rate(w, f'rescue_a{A:g}'):8.1f}%{rand_rate(w, A):8.1f}%"
              f"{diff(w):+8.1f}{f'[{lo:+.1f},{hi:+.1f}]':>18}{len(w):>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="P1_v4")
    args = ap.parse_args()
    data = load(args.tag)
    pull = lambda envs: [r for e in envs for r in data[e]]

    kept = NONREACT + REACT
    print(f"tag {args.tag}   unambiguous rounds: {sum(len(data[e]) for e in kept)} "
          f"over {len(kept)} scenarios (paper set); "
          f"{sum(len(data[e]) for e in DROPPED)} more in dropped {DROPPED}")
    print("per scenario: " + "  ".join(f"{e}={len(data[e])}" for e in kept + DROPPED))

    print("\n" + "=" * 78)
    print("PARTITION A -- TASK TYPE (main text)")
    print("=" * 78)
    family_block("BAYESIAN INFERENCE (4)", pull(BAYES))
    family_block("GAME THEORETIC (4)", pull(GAMES))
    bin_block({"BAYESIAN(4)": pull(BAYES), "GAME(4)": pull(GAMES)})

    print("\n" + "=" * 78)
    print("PARTITION B -- OPPONENT REACTIVITY (appendix)")
    print("=" * 78)
    family_block("NONREACTIVE (5)", pull(NONREACT))
    family_block("OPPONENT REACTIVE (3)", pull(REACT))
    bin_block({"NONREACTIVE(5)": pull(NONREACT), "REACTIVE(3)": pull(REACT)})

    print("\n" + "=" * 78)
    print("POOLING CHECKS")
    print("=" * 78)
    per_key_block("PER SCENARIO", {e: data[e] for e in kept + DROPPED})
    per_key_block("PER MODEL -- nonreactive (5)",
                  {m: [r for e in NONREACT for r in data[e] if r["_model"] == m]
                   for m in MODELS})
    per_key_block("PER MODEL -- reactive (3)",
                  {m: [r for e in REACT for r in data[e] if r["_model"] == m]
                   for m in MODELS})
    per_key_block("PER MODEL -- Chicken alone (is it one model?)",
                  {m: [r for r in data["chicken"] if r["_model"] == m] for m in MODELS})

    print("\nLEAVE-ONE-OUT robustness")
    for label, rows in (
            ("game theoretic(4) minus Chicken", pull(["stag_hunt", "bos", "pennies"])),
            ("Chicken minus phi", [r for r in data["chicken"] if r["_model"] != "phi"]),
            ("Bayesian(4) minus phi", [r for e in BAYES for r in data[e] if r["_model"] != "phi"]),
            ("nonreactive(5) minus phi",
             [r for e in NONREACT for r in data[e] if r["_model"] != "phi"])):
        w = wrong(rows)
        lo, hi = boot_ci(w)
        print(f"  {label:34} rescue {rate(w, f'rescue_a{A:g}'):5.1f}%  "
              f"random {rand_rate(w, A):5.1f}%  diff {diff(w):+5.1f} "
              f"[{lo:+.1f},{hi:+.1f}]  n={len(w)}")

    print("\nFAMILY INTERACTION (difference of paired differences, session bootstrap)")
    for label, (ea, eb) in (("task: Bayesian(4) - game(4)", (BAYES, GAMES)),
                            ("reactivity: nonreactive(5) - reactive(3)", (NONREACT, REACT))):
        sa, sb = defaultdict(list), defaultdict(list)
        for r in wrong(pull(ea)):
            sa[r["_sess"]].append(r)
        for r in wrong(pull(eb)):
            sb[r["_sess"]].append(r)
        rng, vals, ka, kb = random.Random(0), [], list(sa), list(sb)
        for _ in range(N_BOOT):
            ra = [r for k in (rng.choice(ka) for _ in ka) for r in sa[k]]
            rb = [r for k in (rng.choice(kb) for _ in kb) for r in sb[k]]
            vals.append(diff(ra) - diff(rb))
        vals.sort()
        print(f"  {label:42} {st.mean(vals):+6.1f} pp  "
              f"[{vals[int(.025 * len(vals))]:+.1f},{vals[int(.975 * len(vals))]:+.1f}]")


if __name__ == "__main__":
    main()
