#!/usr/bin/env python
"""Belief forms across depth; the action readout throws it away.

LEFT PANEL   balanced accuracy of a TRAINED linear readout of the baseline's
             decision state, against normalized depth (layer / L).
RIGHT STRIP  balanced agreement of the MODEL'S OWN readout at the same final
             position -- the pre-CoT action preference (Probe 2).

The two panels read the same vector at 100% depth. They differ only in whether
the linear readout is fitted to the baseline or is the model's own unembedding
direction. The drop between them is the belief-to-action gap, and drawing it in
one figure is the point: the information is present and is not used.

WHY NORMALIZED DEPTH: models span 26-80 layers, so raw layer index cannot be
overlaid.

WHY BALANCED ACCURACY: raw accuracy at layer 0 sits at each scenario's majority
base rate (51-66%). Balanced pins layer 0 at exactly 50, so "rises from chance"
is visible, and it matches the metric the main table reports.

WHY THREE FAMILIES: Chicken is a game with NO reactive opponent type. Its belief
forms like a Bayesian scenario (98.5) while its action collapses like a game
(66.1) -- the largest gap in the suite. Folding it into a single "game" average
destroys exactly that contrast. Chicken is the control that separates the two
stages; keep it split.

DATA
  belief  results/{P1_v4,P1_v4_layerbal}/probe_{model}_{env}.json
          field `layer_accuracy_bal`, final layer, no layer selection.
  action  results/P1_v4/behavior_by_t.json, pooled over t from the per-class
          counts: 0.5*(correct_act1/n_act1 + correct_act0/n_act0).
IPD is excluded from the suite (horizon- rather than belief-driven).

Usage:  micromamba run -n ftdiag python analysis/plot_belief_layers.py
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle
from paperstyle import SURFACE, INK, INK_MUTED, FIG_WIDE, FIG_COL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "plots", "belief_layers")

# The \caption in paper carries the title -- an in-figure title duplicates it
# and costs vertical space that is scarce at one-column width.
SHOW_TITLE = False
SHOW_COMBOS = True        # thin per-combo lines behind the family means

# --- one-column sizing --------------------------------------------------------
# AUTHORED AT ONE-COLUMN WIDTH. Placed as \begin{figure} with
# \includegraphics[width=\columnwidth], i.e. 3.3 in -- NOT a two-column figure*.
# So every size below is the PRINTED point size, with no LaTeX scaling between.
# This figure was previously authored at 11.0 in with 18-26pt type; at
# \columnwidth that is a 0.30x reduction landing at 5-8pt, which breaks AAAI's
# 9pt in-figure minimum. Author at the width the figure is actually placed at.
#
# 9.0 leaves ZERO margin -- the include must stay 1.0x. [width=0.9\columnwidth]
# would put every label at 8.1pt and violate the rule.
PT_BASE = 9.0       # ticks, axis labels, gap labels, "Chance"
PT_TITLE = 10.0     # the two panel titles -- the only type above 9pt
PT_LEGEND = 9.0     # the one-row legend. 9.0 is the floor, not a default: the
                    # row is measured at draw time and must fit FIG_W at this
                    # size. Do not drop below 9.0 -- that is the AAAI minimum.

FIG_W = FIG_COL     # 3.3 -- AAAI one-column width
AX_H = 1.45         # plotting area height, inches
LEFT_IN = 0.50      # y label + y tick labels
RIGHT_IN = 0.09      # clearance for the "Action (P2)" title and the gap labels
TOP_IN = 0.22       # panel titles
XTICK_IN = 0.16
XLABEL_IN = 0.17
LEGEND_IN = 0.21    # ONE row holding all three families
BOT_IN = 0.03

# Categorical slots 1-3 of the validated palette. Checked with
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
# -> all pass; worst all-pairs CVD dE 9.2 (deutan), normal-vision 24.0. The aqua
# slot is under 3:1 on this surface, so the relief rule applies: direct labels.
C_BAYES, C_NONREACT, C_REACT = "#2a78d6", "#1baf7a", "#eb6834"

MODELS = ["gemma2b", "llama", "phi", "qwen", "dsqwen7",
          "qwen27", "gemma31", "llama70", "dsllama70"]
SCEN = {"bandit": "bayes", "monty": "bayes", "trust_game": "bayes",
        "machine": "bayes", "chicken": "nonreact",
        "stag_hunt": "react", "bos": "react", "pennies": "react"}
# Sentence case, and short enough that all three share ONE legend row at 3.3 in.
# "inference" and "(non-reactive)" are dropped: the caption carries both, and at
# one-column width the row only closes without them.
FAMILIES = [("bayes", "Bayesian", C_BAYES),
            ("nonreact", "Chicken", C_NONREACT),
            ("react", "Reactive games", C_REACT)]

GRID = np.arange(0.0, 100.0 + 1e-9, 2.0)


def split(base):
    for env in SCEN:
        if base.endswith("_" + env):
            return base[: -(len(env) + 1)], env
    return None


def load_belief():
    out = {}
    for tag in ("P1_v4_layerbal", "P1_v4"):        # canonical P1_v4 wins
        for path in glob.glob(os.path.join(ROOT, "results", tag, "probe_*.json")):
            hit = split(os.path.basename(path)[len("probe_"):-len(".json")])
            if hit is None or hit[0] not in MODELS:
                continue
            lab = json.load(open(path)).get("layer_accuracy_bal")
            if not lab:
                continue
            layers = sorted(int(k) for k in lab)
            acc = np.array([lab[str(k)] for k in layers]) * 100.0
            depth = np.array(layers, dtype=float) / layers[-1] * 100.0
            out[hit] = np.interp(GRID, depth, acc)
    return out


def load_action():
    """Pooled balanced pre-CoT agreement, recomputed from per-class counts."""
    beh = json.load(open(os.path.join(ROOT, "results", "P1_v4",
                                      "behavior_by_t.json")))["behavior"]
    out = {}
    for combo, per_t in beh.items():
        hit = split(combo)
        if hit is None or hit[0] not in MODELS:
            continue
        c1 = c0 = n1 = n0 = 0.0
        for rec in per_t.values():
            if rec.get("agree_act1") is not None:
                c1 += rec["agree_act1"] * rec["n_act1"]
                n1 += rec["n_act1"]
            if rec.get("agree_act0") is not None:
                c0 += rec["agree_act0"] * rec["n_act0"]
                n0 += rec["n_act0"]
        if n1 and n0:
            out[hit] = 50.0 * (c1 / n1 + c0 / n0)
    return out


def main():
    paperstyle.apply()
    matplotlib.rcParams.update({
        # authored at final printed size -- the saved box must not be re-cropped
        "savefig.bbox": None,
        "font.size": PT_BASE,
        "axes.labelsize": PT_BASE, "xtick.labelsize": PT_BASE,
        "ytick.labelsize": PT_BASE, "legend.fontsize": PT_BASE,
        "axes.titlesize": PT_TITLE,
    })
    belief, action = load_belief(), load_action()
    combos = [k for k in belief if k in action]
    print(f"{len(belief)} belief curves, {len(action)} action values, "
          f"{len(combos)} paired")

    # Authored at one-column width; the layout is specified in INCHES so the
    # type-to-plot ratio is exact and does not drift when a constant is edited.
    fig_h = TOP_IN + AX_H + XTICK_IN + XLABEL_IN + LEGEND_IN + BOT_IN
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(FIG_W, fig_h), sharey=True,
        gridspec_kw={"width_ratios": [4.2, 1.6], "wspace": 0.10})
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(
        left=LEFT_IN / FIG_W,
        right=1 - RIGHT_IN / FIG_W,
        top=1 - TOP_IN / fig_h,
        bottom=(BOT_IN + LEGEND_IN + XLABEL_IN + XTICK_IN) / fig_h,
    )

    for ax in (axL, axR):
        ax.axhline(50, color=INK, lw=0.7, ls=(0, (2, 3)), zorder=1)
    axL.annotate("Chance", xy=(1.5, 50), xytext=(1.5, 48.6), color=INK,
                 ha="left", va="top", fontsize=PT_BASE)

    rng = np.random.default_rng(0)
    handles = []
    for key, label, color in FAMILIES:
        sel = [k for k in combos if SCEN[k[1]] == key]
        if SHOW_COMBOS:
            for k in sel:
                axL.plot(GRID, belief[k], color=color, lw=0.35, alpha=0.16, zorder=2)
            jit = rng.uniform(-0.15, 0.15, len(sel)) - 0.20
            axR.scatter(jit, [action[k] for k in sel], s=5, color=color,
                        alpha=0.38, linewidth=0, zorder=2)

        mean = np.vstack([belief[k] for k in sel]).mean(axis=0)
        act = float(np.mean([action[k] for k in sel]))
        line, = axL.plot(GRID, mean, color=color, lw=1.5, zorder=4,
                         solid_capstyle="round", label=label)
        handles.append(line)
        axR.scatter([-0.20], [act], s=24, color=color, zorder=5,
                    edgecolor=SURFACE, linewidth=0.7)

        # the drop, drawn across the panel break
        axL.plot([100], [mean[-1]], "o", ms=3.4, color=color, zorder=5,
                 markeredgecolor=SURFACE, markeredgewidth=0.6)
        fig.add_artist(matplotlib.patches.ConnectionPatch(
            xyA=(100, mean[-1]), coordsA=axL.transData,
            xyB=(-0.20, act), coordsB=axR.transData,
            color=color, lw=0.9, ls=(0, (2.2, 1.8)), zorder=3, alpha=0.9))
        # Chicken (66.1) and Reactive (65.2) land on top of each other; nudge
        # their gap labels apart so both stay readable.
        dy = {"bayes": 0.0, "nonreact": 4.4, "react": -4.4}[key]
        axR.annotate(f"$-${mean[-1] - act:.0f}", xy=(0.18, act + dy), color=color,
                     ha="left", va="center", zorder=6, fontsize=PT_BASE,
                     fontweight="bold")

        print(f"{label:30s} n={len(sel):2d}  belief {mean[-1]:5.1f}  "
              f"action {act:5.1f}  gap {mean[-1]-act:5.1f}   " +
              " ".join(f"{p:3.0f}%:{np.interp(p, GRID, mean):.1f}"
                       for p in (10, 30, 50, 70, 100)))

    axL.set_xlim(0, 100)
    axL.set_ylim(43, 103)
    axL.set_xticks([0, 25, 50, 75, 100])
    axL.set_yticks([50, 60, 70, 80, 90, 100])
    axL.set_xlabel("Network depth (layer / $L$, %)", labelpad=2)
    axL.set_ylabel("Balanced accuracy (%)", labelpad=2)
    axL.yaxis.grid(True, zorder=0)
    axL.set_axisbelow(True)

    axR.set_xlim(-0.60, 1.05)
    axR.set_xticks([])
    axR.yaxis.grid(True, zorder=0)
    axR.set_axisbelow(True)

    for side in ("top", "right"):
        axL.spines[side].set_visible(False)
    for side in ("top", "right", "left"):
        axR.spines[side].set_visible(False)
    axR.tick_params(axis="y", length=0)

    # Bold sentence-case panel headers, matching the RQ1 figure. P1/P2 mirror
    # the row labels of the main results table -- these are the same quantities,
    # verified cell-for-cell, so the labels cross-reference it. "Pre-CoT" is
    # dropped from the right header (it does not fit 0.8 in); the \caption says it.
    axL.set_title("Belief (P1)", color=INK, fontsize=PT_TITLE,
                  fontweight="bold", pad=4)
    axR.set_title("Action (P2)", color=INK, fontsize=PT_TITLE,
                  fontweight="bold", pad=4)

    # All three families on ONE row. This only fits 3.3 in with the shortened
    # labels above and a tightened columnspacing; the handles and pads are pulled
    # in too. PT_LEGEND stays at the 9pt in-figure minimum -- see the check below.
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
                     borderaxespad=0.0, fontsize=PT_LEGEND,
                     bbox_to_anchor=(0.5, BOT_IN / fig_h))

    # A single row that silently overruns the figure box would be clipped by
    # LaTeX, so measure it rather than trusting the constants above.
    fig.canvas.draw()
    leg_w = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
    print(f"legend row: {leg_w:.2f} in of {FIG_W:.2f} in "
          f"({'fits' if leg_w <= FIG_W else 'OVERRUNS'}) at {PT_LEGEND:.1f} pt")

    if SHOW_TITLE:
        fig.suptitle("Belief formation across depth and the belief-to-action gap",
                     y=0.995, va="top", fontweight="bold", fontsize=PT_TITLE + 1)

    print(f"\nauthored size: {FIG_W:.2f} x {fig_h:.2f} in "
          f"(printed pt = authored pt at width=\\columnwidth)")

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        # second stem matches the filename the paper's \includegraphics uses
        for stem in ("belief_layers", "Belief-Action"):
            fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=400)
    print(f"wrote {OUT}/{{belief_layers,Belief-Action}}.{{pdf,png}}")


if __name__ == "__main__":
    main()
