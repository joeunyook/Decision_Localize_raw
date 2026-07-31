#!/usr/bin/env python
"""Statistics for the steering intervention, from already-collected replays.

Answers the three things the rescue-rate table alone cannot:

  PAIRED TEST      rescue and the matched random directions are evaluated on
                   the SAME rounds, so the comparison is paired. McNemar's
                   exact test on the discordant pairs, plus a seed-level
                   bootstrap of the difference (a session is resampled whole,
                   since its ten rounds share a hidden parameter).
  DOSE-RESPONSE    alpha in {1,2,4,8} was collected but never reported. If the
                   rescue rate climbs with alpha under Bayes-EU and stays flat
                   in the games, that is specificity the random control alone
                   does not establish.
  COST ON CORRECT  the rescue direction applied to rounds the model already had
                   right. A direction that only flips things at random would
                   damage these at the same rate it rescues the others.

Usage:  python3 analysis/steer_stats.py [--tag P1_v4]
"""
import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = ["gemma2b", "llama", "phi", "qwen", "dsqwen7",
          "qwen27", "gemma31", "llama70", "dsllama70"]
BAYES_EU = ["trust_game", "machine"]
GAMES = ["stag_hunt", "chicken", "bos", "pennies"]
ALPHAS = [1.0, 2.0, 4.0, 8.0]
MAIN_ALPHA = 4.0


def load(tag):
    """(model, env) -> list of round dicts with the per-condition gaps."""
    out = defaultdict(list)
    for path in glob.glob(os.path.join(ROOT, "steer", tag, "*.json")):
        d = json.load(open(path))
        key = (d["model_key"], d["env"])
        if key[0] not in MODELS:
            continue
        for r in d.get("rounds", []):
            if not r.get("unambiguous", True):
                continue
            g = r.get("gaps", {})
            if "baseline" not in g:
                continue
            r["_sess"] = (d["model_key"], d["env"], d["seed"])
            out[key].append(r)
    return out


def correct(gap, gt_act1):
    return (gap > 0) == gt_act1


def mcnemar(b, c):
    """McNemar on discordant pairs, continuity-corrected normal approximation."""
    n = b + c
    if n == 0:
        return 1.0
    z = (abs(b - c) - 1) / math.sqrt(n)
    if z <= 0:
        return 1.0
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def boot_diff(sessions, f_a, f_b, n_boot=2000, seed=0):
    """Percentile CI for (rate_a - rate_b), resampling whole sessions."""
    rng = random.Random(seed)
    keys = list(sessions)
    diffs = []
    for _ in range(n_boot):
        pick = [sessions[rng.choice(keys)] for _ in keys]
        rows = [r for s in pick for r in s]
        na = sum(f_a(r) for r in rows)
        nb = sum(f_b(r) for r in rows)
        d = len(rows)
        if d:
            diffs.append(100 * (na - nb) / d)
    diffs.sort()
    if not diffs:
        return float("nan"), float("nan")
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


def analyse(data, envs, label, n_random=2):
    print(f"\n{'=' * 100}\n{label}\n{'=' * 100}")
    rows = [r for m in MODELS for e in envs for r in data.get((m, e), [])]
    wrong = [r for r in rows if not correct(r["gaps"]["baseline"], r["gt_act1"])]
    right = [r for r in rows if correct(r["gaps"]["baseline"], r["gt_act1"])]
    print(f"n = {len(rows)} unambiguous rounds; {len(wrong)} wrong pre-steer, "
          f"{len(right)} correct pre-steer")

    a = MAIN_ALPHA
    rk, rndk = f"rescue_a{a:g}", [f"rand{j}_a{a:g}" for j in range(n_random)]

    def resc(r):
        return correct(r["gaps"].get(rk, r["gaps"]["baseline"]), r["gt_act1"])

    def rand_any(r, j):
        return correct(r["gaps"].get(rndk[j], r["gaps"]["baseline"]), r["gt_act1"])

    n_r = sum(resc(r) for r in wrong)
    print(f"\n  rescue (alpha={a:g}):  {100 * n_r / len(wrong):.1f}%  ({n_r}/{len(wrong)})")
    for j in range(n_random):
        n_j = sum(rand_any(r, j) for r in wrong)
        b = sum(resc(r) and not rand_any(r, j) for r in wrong)
        c = sum(rand_any(r, j) and not resc(r) for r in wrong)
        p = mcnemar(b, c)
        sess = defaultdict(list)
        for r in wrong:
            sess[r["_sess"]].append(r)
        lo, hi = boot_diff(dict(sess), resc, lambda r, j=j: rand_any(r, j))
        ps = "<1e-4" if p < 1e-4 else f"{p:.4g}"
        print(f"  random {j}:            {100 * n_j / len(wrong):.1f}%  ({n_j}/{len(wrong)})"
              f"   paired diff {100 * (n_r - n_j) / len(wrong):+.1f} pp"
              f"  [{lo:+.1f},{hi:+.1f}]  McNemar b={b} c={c} p={ps}")

    print(f"\n  dose-response (rescue rate among the {len(wrong)} wrong rounds):")
    line = "    "
    for al in ALPHAS:
        k = f"rescue_a{al:g}"
        v = 100 * sum(correct(r["gaps"].get(k, r["gaps"]["baseline"]), r["gt_act1"])
                      for r in wrong) / len(wrong)
        line += f"alpha={al:g}: {v:5.1f}%   "
    print(line)
    line = "    random mean  "
    for al in ALPHAS:
        vs = []
        for j in range(n_random):
            k = f"rand{j}_a{al:g}"
            vs.append(100 * sum(correct(r["gaps"].get(k, r["gaps"]["baseline"]),
                                        r["gt_act1"]) for r in wrong) / len(wrong))
        line += f"{sum(vs) / len(vs):5.1f}%        "
    print(line)

    print(f"\n  cost on rounds already correct (alpha={a:g}):")
    if right:
        dmg_r = 100 * sum(not correct(r["gaps"].get(rk, r["gaps"]["baseline"]),
                                      r["gt_act1"]) for r in right) / len(right)
        anti = f"anti_a{a:g}"
        dmg_a = 100 * sum(not correct(r["gaps"].get(anti, r["gaps"]["baseline"]),
                                      r["gt_act1"]) for r in right) / len(right)
        dmg_n = []
        for j in range(n_random):
            k = f"rand{j}_a{a:g}"
            dmg_n.append(100 * sum(not correct(r["gaps"].get(k, r["gaps"]["baseline"]),
                                               r["gt_act1"]) for r in right) / len(right))
        print(f"    rescue direction damages {dmg_r:.1f}%   "
              f"anti-direction damages {dmg_a:.1f}%   "
              f"random damages {sum(dmg_n) / len(dmg_n):.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="P1_v4")
    args = ap.parse_args()
    data = load(args.tag)
    print(f"loaded {sum(len(v) for v in data.values())} unambiguous rounds "
          f"from {len(data)} combos (tag {args.tag})")
    analyse(data, BAYES_EU, "BAYES-EU  (trust_game, machine)")
    analyse(data, GAMES, "GAME-THEORETIC  (stag_hunt, chicken, bos, pennies)")


if __name__ == "__main__":
    main()
