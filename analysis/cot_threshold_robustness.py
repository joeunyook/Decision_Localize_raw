#!/usr/bin/env python
"""Is the CoT threshold effect an artifact of where the bins were cut?

Result 3 reports a contrast between combos with pre-CoT agreement (P2) below
65 and between 65 and 80. Those cut points were chosen after seeing the data,
so the binned contrast alone is post hoc. This script provides the two checks
that settle it:

  (1) CONTINUOUS. Correlate P2 against each post-CoT quantity across all
      model x scenario combos -- no binning at all. Pearson and Spearman, with
      Fisher-z confidence intervals and p-values.
  (2) CUT-POINT SWEEP. Slide the threshold from 55 to 80 in steps of 1 and
      report the contrast at every cut. If the effect only appears at 65, it
      is an artifact; if it holds across the sweep, the bin is presentational.

Usage:  python3 analysis/cot_threshold_robustness.py
"""
import glob
import json
import math
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESP = os.path.join(ROOT, "P3_postcot", "responses")
AMBIG = 0.10
MIN_WRONG = 25          # combos with too few pre-CoT errors have no salvage denominator

MODELS = ["gemma2b", "llama", "phi", "qwen", "dsqwen7",
          "qwen27", "gemma31", "llama70", "dsllama70"]
REASONING = {"dsqwen7", "dsllama70"}
SCEN = {"bandit": "bayes", "monty": "bayes", "trust_game": "bayes",
        "machine": "bayes", "chicken": "game",
        "stag_hunt": "game", "bos": "game", "pennies": "game"}
ANSWER_RE = re.compile(r"Answer\s*:\s*([AB])\b|\\boxed\{([AB])\}")


def parse_letter(t):
    m = list(ANSWER_RE.finditer(t or ""))
    return (m[-1].group(1) or m[-1].group(2)) if m else None


def split(b):
    for e in SCEN:
        if b.endswith("_" + e):
            return b[: -(len(e) + 1)], e
    return None


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):                      # average ties
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    return pearson(rank(xs), rank(ys))


def fisher(r, n):
    """(lo, hi, two-sided p) for a correlation via the Fisher z transform."""
    if n < 5 or abs(r) >= 1:
        return float("nan"), float("nan"), float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1 / math.sqrt(n - 3)
    lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z / se) / math.sqrt(2))))
    return lo, hi, p


def collect():
    rows = defaultdict(list)
    for path in glob.glob(os.path.join(RESP, "*.json")):
        hit = split(os.path.basename(path)[: -len(".json")].rsplit("_seed", 1)[0])
        if hit is None or hit[0] not in MODELS:
            continue
        for r in json.load(open(path)).get("rounds", []):
            o, pre = r.get("oracle_p_act1"), r.get("delta_pre_sem")
            if o is None or pre is None or abs(o - 0.5) <= AMBIG:
                continue
            L = parse_letter(r.get("cot_text"))
            rows[hit].append(dict(
                gt=o > 0.5, pre_ok=(pre > 0) == (o > 0.5), term=L is not None,
                post_ok=(L == "B") == (o > 0.5) if L else False))
    return rows


def combo_stats(rows, envs, include_reasoning):
    """One record per (model, scenario) combo."""
    out = []
    for (m, e), rr in rows.items():
        if e not in envs or (m in REASONING and not include_reasoning):
            continue
        good = [r for r in rr if r["pre_ok"]]
        wrong = [r for r in rr if not r["pre_ok"]]
        if len(wrong) < MIN_WRONG or not good:
            continue
        bal = {}
        for field, over_term in (("pre_ok", False), ("post_ok", False)):
            acc = {}
            for c in (True, False):
                s = [r for r in rr if r["gt"] is c]
                acc[c] = sum(r[field] for r in s) / len(s) if s else float("nan")
            bal[field] = 100 * (acc[True] + acc[False]) / 2
        out.append(dict(
            combo=f"{m} x {e}",
            p2=bal["pre_ok"],
            p3=bal["post_ok"],
            salvage=100 * sum(r["post_ok"] for r in wrong) / len(wrong),
            flip=100 * sum(r["term"] and not r["post_ok"] for r in good) / len(good),
            noans=100 * sum(not r["term"] for r in good) / len(good),
            term=100 * sum(r["term"] for r in rr) / len(rr),
        ))
    return out


def continuous(recs, label):
    print(f"\n{'=' * 92}\n(1) CONTINUOUS -- no binning.  {label}  (n={len(recs)} combos)\n{'=' * 92}")
    xs = [r["p2"] for r in recs]
    print(f"{'outcome vs P2':22s}{'Pearson':>9s}{'95% CI':>18s}{'p':>10s}{'Spearman':>10s}")
    for key, name in (("salvage", "correction"), ("flip", "corruption"),
                      ("noans", "non-termination"), ("p3", "final accuracy"),
                      ("term", "termination rate")):
        ys = [r[key] for r in recs]
        rp = pearson(xs, ys)
        lo, hi, p = fisher(rp, len(recs))
        ps = "<1e-4" if p < 1e-4 else f"{p:.4f}"
        print(f"  {name:20s}{rp:9.2f}   [{lo:5.2f},{hi:5.2f}]{ps:>10s}{spearman(xs, ys):10.2f}")


def sweep(recs, label):
    print(f"\n{'=' * 92}\n(2) CUT-POINT SWEEP.  {label}\n{'=' * 92}")
    print(f"{'cut':>5s}{'n below':>9s}{'n above':>9s}"
          f"{'corruption below':>19s}{'above':>8s}{'ratio':>8s}"
          f"{'final acc below':>18s}{'above':>8s}")
    for cut in range(55, 81):
        lo = [r for r in recs if r["p2"] < cut]
        hi = [r for r in recs if r["p2"] >= cut]
        if len(lo) < 4 or len(hi) < 4:
            continue
        fl = sum(r["flip"] for r in lo) / len(lo)
        fh = sum(r["flip"] for r in hi) / len(hi)
        al = sum(r["p3"] for r in lo) / len(lo)
        ah = sum(r["p3"] for r in hi) / len(hi)
        mark = "  <- reported" if cut == 65 else ""
        print(f"{cut:5d}{len(lo):9d}{len(hi):9d}{fl:19.1f}{fh:8.1f}"
              f"{fl / fh if fh else float('inf'):8.1f}{al:18.1f}{ah:8.1f}{mark}")


def main():
    rows = collect()
    game = [e for e in SCEN if SCEN[e] == "game"]
    allenv = list(SCEN)

    g = combo_stats(rows, game, include_reasoning=False)
    continuous(g, "game-theoretic, non-reasoning models")
    sweep(g, "game-theoretic, non-reasoning models")

    a = combo_stats(rows, allenv, include_reasoning=False)
    continuous(a, "all eight scenarios, non-reasoning models")
    sweep(a, "all eight scenarios, non-reasoning models")

    ar = combo_stats(rows, allenv, include_reasoning=True)
    continuous(ar, "all eight scenarios, ALL nine models")


if __name__ == "__main__":
    main()
