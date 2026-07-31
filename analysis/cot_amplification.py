#!/usr/bin/env python
"""Does chain of thought build a belief-to-action mapping, or only amplify one?

Computes, per model and scenario family:
  P2          balanced agreement of the pre-CoT action preference sign
  P3 term     balanced agreement of the post-CoT answer, TERMINATED responses only
  P3 eff      the same with a non-terminating response scored as WRONG
  coverage    fraction of steps that terminate with a parseable answer
  salvage     among steps the model got WRONG pre-CoT, the fraction CoT turns
              correct -- the exact analogue of the steering rescue rate, on the
              same seeds (0-39), so the two interventions are comparable
  break       among steps it got RIGHT pre-CoT, the fraction CoT turns wrong

Both salvage and break are reported twice: counting a non-terminating response
as a failure (effective, what a deployed system gets) and over terminated
responses only (which is self-selected -- the rounds a model can resolve are
the rounds it resolves).

Answers are ALWAYS re-parsed from cot_text with the current last-match /
\\boxed{} regex; the stored post_valid flag predates that fix for some models
and is never trusted (DOCUMENTATION.txt, Bug 5).

Usage:  python3 analysis/cot_amplification.py
"""
import glob
import json
import math
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESP = os.path.join(ROOT, "P3_postcot", "responses")

AMBIG = 0.10          # drop steps where the baseline is near-indifferent

MODELS = ["gemma2b", "llama", "phi", "qwen", "dsqwen7",
          "qwen27", "gemma31", "llama70", "dsllama70"]
MD = {"gemma2b": "Gemma-2 2B", "llama": "Llama-3.2 3B", "phi": "Phi-4-mini 3.8B",
      "qwen": "Qwen-3.5 4B", "dsqwen7": "R1-Qwen 7B", "qwen27": "Qwen-3.6 27B",
      "gemma31": "Gemma-4 31B", "llama70": "Llama-3.3 70B",
      "dsllama70": "R1-Llama 70B"}
REASONING = {"dsqwen7", "dsllama70"}

SCEN = {"bandit": "bayes", "monty": "bayes", "trust_game": "bayes",
        "machine": "bayes", "chicken": "game",
        "stag_hunt": "game", "bos": "game", "pennies": "game"}
# the pair the steering rescue rate is defined on (bandit/monty have no
# denominator: the large models make almost no errors there)
BAYES_EU = {"trust_game", "machine"}

ANSWER_RE = re.compile(r"Answer\s*:\s*([AB])\b|\\boxed\{([AB])\}")


def parse_letter(text):
    m = list(ANSWER_RE.finditer(text or ""))
    if not m:
        return None
    g = m[-1]
    return g.group(1) or g.group(2)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def split(base):
    for env in SCEN:
        if base.endswith("_" + env):
            return base[: -(len(env) + 1)], env
    return None


def collect():
    """(model, env) -> list of per-step dicts."""
    out = defaultdict(list)
    for path in glob.glob(os.path.join(RESP, "*.json")):
        stem = os.path.basename(path)[: -len(".json")]
        stem = stem.rsplit("_seed", 1)[0]
        hit = split(stem)
        if hit is None or hit[0] not in MODELS:
            continue
        d = json.load(open(path))
        for r in d.get("rounds", []):
            p = r.get("oracle_p_act1")
            if p is None or abs(p - 0.5) <= AMBIG:
                continue
            pre = r.get("delta_pre_sem")
            if pre is None:
                continue
            letter = parse_letter(r.get("cot_text"))
            out[hit].append({
                "gt": p > 0.5,                       # baseline-optimal action
                "pre_ok": (pre > 0) == (p > 0.5),    # pre-CoT sign correct
                "term": letter is not None,
                # canonical mapping only in the post-CoT run: A = act0, B = act1
                "post_ok": (letter == "B") == (p > 0.5) if letter else False,
            })
    return out


def balanced(rows, field, over_terminated=False):
    """Macro-average over the two baseline-optimal classes."""
    acc = {}
    for cls in (True, False):
        sub = [r for r in rows if r["gt"] is cls]
        if over_terminated:
            sub = [r for r in sub if r["term"]]
        if not sub:
            return float("nan")
        acc[cls] = sum(r[field] for r in sub) / len(sub)
    return 100 * (acc[True] + acc[False]) / 2


def report(title, rows_by_model, envs):
    print(f"\n{'=' * 108}\n{title}\n{'=' * 108}")
    print(f"{'model':17s}{'P2':>7s}{'P3 term':>9s}{'P3 eff':>8s}{'cov%':>7s}"
          f"{'delta':>8s}   {'salvage (eff)':>20s}{'break':>8s}   n_wrong")
    agg = []
    for m in MODELS:
        rows = [r for e in envs for r in rows_by_model.get((m, e), [])]
        if not rows:
            continue
        p2 = balanced(rows, "pre_ok")
        p3t = balanced(rows, "post_ok", over_terminated=True)
        p3e = balanced(rows, "post_ok")
        cov = 100 * sum(r["term"] for r in rows) / len(rows)
        wrong = [r for r in rows if not r["pre_ok"]]
        right = [r for r in rows if r["pre_ok"]]
        sal = 100 * sum(r["post_ok"] for r in wrong) / len(wrong) if wrong else float("nan")
        lo, hi = wilson(sum(r["post_ok"] for r in wrong), len(wrong))
        brk = 100 * sum(not r["post_ok"] for r in right) / len(right) if right else float("nan")
        agg.append((m, p2, p3t, p3e, cov, sal, brk, len(wrong)))
        star = "*" if m in REASONING else " "
        print(f"{star}{MD[m]:16s}{p2:7.1f}{p3t:9.1f}{p3e:8.1f}{cov:7.1f}"
              f"{p3e - p2:+8.1f}   {sal:6.1f} [{lo:4.1f},{hi:5.1f}]{brk:8.1f}   {len(wrong):5d}")
    print("  * reasoning-distilled; their P3 term is over a self-selected subset")
    return agg


def main():
    data = collect()
    print(f"loaded {sum(len(v) for v in data.values())} scored steps "
          f"across {len(data)} model x scenario combos")

    bayes = [e for e in SCEN if SCEN[e] == "bayes"]
    game = [e for e in SCEN if SCEN[e] == "game"]

    a_b = report("BAYESIAN INFERENCE (bandit, monty, trust_game, machine)", data, bayes)
    a_g = report("GAME-THEORETIC (stag_hunt, chicken, bos, pennies)", data, game)
    report("BAYES-EU ONLY (trust_game, machine) -- the steering comparison set",
           data, sorted(BAYES_EU))

    # Does the CoT gain track the pre-CoT mapping quality?
    print(f"\n{'=' * 108}\nDOES THE GAIN TRACK P2?  (game-theoretic, "
          f"non-reasoning models)\n{'=' * 108}")
    pts = [(p2, p3e - p2, m) for m, p2, p3t, p3e, cov, s, b, n in a_g
           if m not in REASONING]
    for p2, d, m in sorted(pts):
        print(f"  {MD[m]:17s} P2 {p2:5.1f}   effective gain {d:+6.1f}")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sx = math.sqrt(sum((p[0] - mx) ** 2 for p in pts))
    sy = math.sqrt(sum((p[1] - my) ** 2 for p in pts))
    r = sum((p[0] - mx) * (p[1] - my) for p in pts) / (sx * sy)
    print(f"  Pearson r(P2, effective gain) = {r:.2f}  (n={n})")

    # Termination, the reasoning models' failure mode
    print(f"\n{'=' * 108}\nTERMINATION RATE BY SCENARIO (reasoning-distilled)"
          f"\n{'=' * 108}")
    for m in ("dsqwen7", "dsllama70"):
        cells = []
        for e in SCEN:
            rows = data.get((m, e), [])
            if rows:
                cells.append((100 * sum(r["term"] for r in rows) / len(rows), e))
        cells.sort(reverse=True)
        print(f"  {MD[m]:15s} " +
              "  ".join(f"{e} {v:.0f}%" for v, e in cells))


if __name__ == "__main__":
    main()
