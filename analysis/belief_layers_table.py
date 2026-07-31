#!/usr/bin/env python
"""Numbers behind the belief-formation figure. Tables only, no plotting.

Balanced probe accuracy against NORMALIZED depth (layer / L), because models
span 27-81 layers. Balanced, so layer 0 is pinned at 50 (chance) rather than at
each scenario's majority base rate.

Usage:  python3 analysis/belief_layers_table.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = ["gemma2b", "llama", "phi", "qwen", "dsqwen7",
          "qwen27", "gemma31", "llama70", "dsllama70"]
MDISP = {"gemma2b": "Gemma-2 2B", "llama": "Llama-3.2 3B", "phi": "Phi-4-mini 3.8B",
         "qwen": "Qwen-3.5 4B", "dsqwen7": "R1-Qwen 7B", "qwen27": "Qwen-3.6 27B",
         "gemma31": "Gemma-4 31B", "llama70": "Llama-3.3 70B",
         "dsllama70": "R1-Llama 70B"}

SCENARIOS = {
    "bandit":     ("bayes", "Bandit"),
    "monty":      ("bayes", "Monty Hall"),
    "trust_game": ("bayes", "Trust Game"),
    "machine":    ("bayes", "Machine"),
    "chicken":    ("nonreact", "Chicken"),
    "stag_hunt":  ("react", "Stag Hunt"),
    "bos":        ("react", "BoS"),
    "pennies":    ("react", "Pennies"),
}
FAMILIES = [("bayes", "Bayesian inference"),
            ("nonreact", "Game, non-reactive (Chicken)"),
            ("react", "Game, reactive")]

GRID = np.arange(0.0, 100.0 + 1e-9, 1.0)
COLS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def load():
    raw = {}
    for tag in ("P1_v4_layerbal", "P1_v4"):        # canonical P1_v4 wins
        for path in glob.glob(os.path.join(ROOT, "results", tag, "probe_*.json")):
            base = os.path.basename(path)[len("probe_"):-len(".json")]
            hit = None
            for env in SCENARIOS:
                if base.endswith("_" + env):
                    hit = (base[: -(len(env) + 1)], env)
                    break
            if hit is None or hit[0] not in MODELS:
                continue
            d = json.load(open(path))
            lab = d.get("layer_accuracy_bal")
            if not lab:
                continue
            layers = sorted(int(k) for k in lab)
            acc = np.array([lab[str(k)] for k in layers]) * 100.0
            depth = np.array(layers, dtype=float) / layers[-1] * 100.0
            raw[hit] = (np.interp(GRID, depth, acc), layers[-1])
    return raw


def row(vals):
    return " ".join(f"{np.interp(c, GRID, vals):5.1f}" for c in COLS)


def d90(curve):
    return GRID[int(np.argmax(curve >= 0.9 * curve.max()))]


def main():
    raw = load()
    curves = {k: v[0] for k, v in raw.items()}
    nlayer = {m: raw[(m, "bandit")][1] for m in MODELS if (m, "bandit") in raw}
    print(f"loaded {len(curves)} / {len(MODELS)*len(SCENARIOS)} combos\n")
    print("layers per model:", ", ".join(f"{MDISP[m]} {nlayer[m]}" for m in MODELS))

    head = "depth %   " + " ".join(f"{c:5d}" for c in COLS)

    print("\n\n=== A. BY FAMILY (mean over combos) " + "=" * 40)
    print(head)
    for key, name in FAMILIES:
        sel = [c for (m, e), c in curves.items() if SCENARIOS[e][0] == key]
        print(f"{name:38s}\n  n={len(sel):<3d}  {row(np.vstack(sel).mean(axis=0))}")

    print("\n  two-line version for the record (Chicken folded into 'game'):")
    for key, name in [("bayes", "Bayesian"), (None, "Game (all four)")]:
        sel = [c for (m, e), c in curves.items()
               if (SCENARIOS[e][0] == "bayes") == (key == "bayes")]
        print(f"  {name:16s} n={len(sel):<3d} {row(np.vstack(sel).mean(axis=0))}")

    print("\n\n=== B. BY SCENARIO (mean over 9 models) " + "=" * 36)
    print(head + "   D90  final")
    for key, name in FAMILIES:
        print(f"-- {name}")
        for env, (fam, disp) in SCENARIOS.items():
            if fam != key:
                continue
            sel = [curves[(m, env)] for m in MODELS if (m, env) in curves]
            mean = np.vstack(sel).mean(axis=0)
            ds = [d90(c) for c in sel]
            print(f"  {disp:14s}{row(mean)}   {np.median(ds):3.0f}  {mean[-1]:5.1f}")

    print("\n\n=== C. BY MODEL (mean over 8 scenarios) " + "=" * 36)
    print(head)
    for m in MODELS:
        sel = [curves[(m, e)] for e in SCENARIOS if (m, e) in curves]
        print(f"  {MDISP[m]:16s}{row(np.vstack(sel).mean(axis=0))}")

    print("\n\n=== D. SCALE BAND x FAMILY " + "=" * 49)
    bands = [("small 2-7B", ["gemma2b", "llama", "phi", "qwen", "dsqwen7"]),
             ("mid 27-31B", ["qwen27", "gemma31"]),
             ("large 70B",  ["llama70", "dsllama70"])]
    print(head)
    for bname, bmodels in bands:
        print(f"-- {bname}")
        for key, name in FAMILIES:
            sel = [curves[(m, e)] for m in bmodels for e in SCENARIOS
                   if (m, e) in curves and SCENARIOS[e][0] == key]
            print(f"  {name:36s}{row(np.vstack(sel).mean(axis=0))}")

    print("\n\n=== E. FINAL-LAYER SPREAD per scenario (the '83.2 floor') " + "=" * 17)
    print(f"  {'scenario':14s} {'min':>6s} {'median':>7s} {'max':>6s}   worst combo")
    for env, (fam, disp) in SCENARIOS.items():
        fin = {m: curves[(m, env)][-1] for m in MODELS if (m, env) in curves}
        lo = min(fin, key=fin.get)
        print(f"  {disp:14s} {min(fin.values()):6.1f} {np.median(list(fin.values())):7.1f} "
              f"{max(fin.values()):6.1f}   {MDISP[lo]}")

    print("\n\n=== G. BY MODEL x FAMILY (the plot's data, fully broken out) " + "=" * 14)
    print("model              family                 " + " ".join(f"{c:5d}" for c in COLS)
          + "   D90")
    for m in MODELS:
        for i, (key, name) in enumerate(FAMILIES):
            sel = [curves[(m, e)] for e in SCENARIOS
                   if (m, e) in curves and SCENARIOS[e][0] == key]
            mean = np.vstack(sel).mean(axis=0)
            ds = np.median([d90(c) for c in sel])
            lead = MDISP[m] if i == 0 else ""
            print(f"  {lead:16s} {name:22s}{row(mean)}   {ds:3.0f}")

    print("\n\n=== H. BY MODEL x REACTIVE SCENARIO (each of the three) " + "=" * 20)
    react = [e for e in SCENARIOS if SCENARIOS[e][0] == "react"]
    print("model              scenario     " + " ".join(f"{c:5d}" for c in COLS)
          + "   D90  peak  final")
    for m in MODELS:
        for i, env in enumerate(react):
            c = curves[(m, env)]
            lead = MDISP[m] if i == 0 else ""
            print(f"  {lead:16s} {SCENARIOS[env][1]:12s}{row(c)}   "
                  f"{d90(c):3.0f} {c.max():5.1f} {c[-1]:5.1f}")

    print("\n\n=== F. FINAL-LAYER DIPS (peak -> final, combos losing >2 points) " + "=" * 11)
    dips = []
    for (m, env), c in curves.items():
        if c.max() - c[-1] > 2.0:
            dips.append((c.max() - c[-1], MDISP[m], SCENARIOS[env][1], c.max(), c[-1]))
    for drop, m, e, pk, fi in sorted(dips, reverse=True):
        print(f"  {m:16s} {e:12s} {pk:5.1f} -> {fi:5.1f}   (-{drop:.1f})")
    print(f"  {len(dips)} of {len(curves)} combos dip by more than 2 points")


if __name__ == "__main__":
    main()
