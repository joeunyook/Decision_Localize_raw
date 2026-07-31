#!/usr/bin/env python
"""RQ1 figure: post-CoT decision outcomes, decomposed.

Every unambiguous decision step lands in exactly one of three outcomes:
    correct     -- terminated with a parseable answer that matches the baseline
    incorrect   -- terminated with a parseable answer that does not
    no answer   -- hit the generation cap without emitting a parseable answer

Segments are CLASS-BALANCED: each step is weighted 1/(2 * n_class) for its
baseline-optimal class, so the three segments still sum to 100% while matching
the balanced metric used everywhere else in the paper. Terminated accuracy is
correct/(correct+incorrect); coverage is correct+incorrect; effective accuracy
is the correct segment alone -- all three readable off one bar.

Answers are re-parsed from cot_text with the current last-match / \\boxed{}
regex (core.probes.parse_answer_letter); the stored post_valid flag is never
trusted (see DOCUMENTATION.txt, Bug 5).

Usage:  micromamba run -n ftdiag python analysis/plot_rq1_outcomes.py
"""
import glob
import json
import os
import random
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle
from paperstyle import (C_CORRECT, C_WRONG, C_NOANS, SURFACE, INK, INK_MUTED,
                        FIG_WIDE, FIG_COL)

# --- type sizes ---------------------------------------------------------------
# AUTHORED AT ONE-COLUMN WIDTH.  This figure is placed as a single-column
# \begin{figure} at \includegraphics[width=\columnwidth], i.e. ~3.3 in -- NOT as
# a two-column figure*.  So FIG_W must be FIG_COL, and every size below is the
# PRINTED point size with no LaTeX scaling in between.
#
# Do NOT author this at FIG_WIDE (7.0 in) and let width=\columnwidth shrink it:
# that is a 3.3/7.0 = 0.47x reduction, which turned a previous 9-10pt pass into
# 4-5pt on the page and broke AAAI's 9pt in-figure minimum.  (Cost me two
# rebuilds on 2026-07-28 -- author at the width the figure is actually placed
# at, every time.)
#
# Everything internal sits at EXACTLY the 9pt minimum, panel titles at 10pt.
# NOTE: 9.0 leaves ZERO margin. Compliance depends entirely on the include being
# 1.0x -- \includegraphics[width=\columnwidth]{} with this 3.3in canvas. Anything
# that shrinks it breaks the rule immediately: [width=0.9\columnwidth] would put
# every label at 8.1pt. If a fractional width is ever needed, re-author FIG_W to
# the real target width instead of scaling.
PT_BASE = 9.0      # legend, tick labels, band headers, accuracy numerals
PT_NAME = 9.0      # model names
PT_TITLE = 10.0    # family panel titles -- the only type above 9pt
PT_SUP = 11.0      # figure title (unused -- SHOW_TITLE is False, see below)

# --- geometry, in inches (authored = printed) ---------------------------------
# Height is bought back from PITCH, never from type size: the figure was reading
# as too dominant in-column, so the row pitch and the band gaps are tightened
# while every label stays at the 9pt floor.
FIG_W = FIG_COL    # 3.3 -- AAAI ONE-COLUMN width
ROW_IN = 0.145     # vertical pitch of one model row; must clear PT_NAME
LEFT_IN = 0.90     # gutter for the model names at PT_NAME
RIGHT_IN = 0.04
TOP_IN = 0.30      # suptitle strip (only when SHOW_TITLE)
XTICK_IN = 0.16    # x tick labels (bottom panel only -- panels share x)
LEGEND_IN = 0.18   # ONE row of four at one-column width
BOT_IN = 0.03
BAR_H = 0.66       # bar height in row units; also bounds the 50% chance rule

# TITLE PLACEMENT. A panel title must read as belonging to the panel BELOW it.
# These were previously one constant (TITLE_IN) doing two different jobs -- the
# figure's top strip AND the gap between panels -- which forced the
# "Game-theoretic" title to sit ~0.02 in under the last Bayesian bar and
# ~0.08 in above its own, i.e. visually attached to the wrong panel.
# They are now independent:
TITLE_PAD_PT = 2.0     # axes top -> title bottom. Identical on both panels, so
                       # each title sits the same distance above its own bars.
TITLE_STRIP_IN = 0.20  # figure top margin reserved for the first panel title
PANEL_GAP_IN = 0.34    # whitespace between the two panels. Must exceed the
                       # title's own height + TITLE_PAD_PT by enough that the
                       # clearance ABOVE the lower title clearly beats the gap
                       # BELOW it -- that asymmetry is what groups it downward.

# A band header is a text line competing with the 9pt model names on either
# side, so its clearances are set in POINTS, not in row units -- tightening
# ROW_IN does not make a 9pt glyph any shorter. BAND_OFF * ROW_IN must stay
# above ~0.115 in (8.3pt) or the header's descenders hit the row below it.
BAND_GAP = 1.50    # y units inserted before each scale-band header
BAND_OFF = 0.82    # header sits this far above its band's first model row
PAD_TOP = 0.05     # clearance above the first band header, in row units. Kept
                   # near zero so the panel title sits close to its own content
                   # -- this plus TITLE_PAD_PT is the whole title-to-bars gap.
PAD_BOT = 0.35
XMAX = 119         # data x limit; 100..XMAX is the numeral gutter. Must clear
                   # a 3-digit "100" AND the CI cap sitting at x=100.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESPONSES = os.path.join(ROOT, "P3_postcot", "responses")
OUT = os.path.join(ROOT, "plots", "rq1_outcomes")

# --- what counts as a scorable step -----------------------------------------
AMBIG = 0.10          # drop |oracle - 0.5| <= AMBIG (baseline near-indifferent)

# AAAI convention is a LaTeX \caption, not an in-figure title. The caption in
# paper_rq1_results.tex already says "Final decisions made by nine models...",
# so an in-figure title states it twice and costs vertical space we cannot
# spare at one-column width.
SHOW_TITLE = False

# Mirrors core.probes.ANSWER_RE. Duplicated so the figure can be built on the
# login node without importing torch.
ANSWER_RE = re.compile(r"Answer\s*:\s*([AB])\b|\\boxed\{([AB])\}")


def parse_answer_letter(text):
    matches = list(ANSWER_RE.finditer(text or ""))
    if not matches:
        return None
    m = matches[-1]
    return m.group(1) or m.group(2)


# --- layout ------------------------------------------------------------------
# Band headers are compact sentence case, NOT caps: at 9pt an all-caps bold
# header reads visually larger than the 9pt model names it groups (every glyph
# occupies full cap height), which made the scale bands over-dominant in column.
BANDS = [
    ("Small (2–4B)", [
        ("gemma2b", "Gemma-2 2B"),
        ("llama",   "Llama-3.2 3B"),
        ("phi",     "Phi-4-mini 3.8B"),
        ("qwen",    "Qwen-3.5 4B"),
    ]),
    ("Mid/large (27–70B)", [
        ("qwen27",  "Qwen-3.6 27B"),
        ("gemma31", "Gemma-4 31B"),
        ("llama70", "Llama-3.3 70B"),
    ]),
    ("R1-distilled", [
        ("dsqwen7",   "R1-Qwen 7B"),
        ("dsllama70", "R1-Llama 70B"),
    ]),
]

FAMILIES = [
    ("Bayesian inference", ["bandit", "monty", "trust_game", "machine"]),
    ("Game-theoretic",     ["stag_hunt", "chicken", "bos", "pennies"]),
]

# Palette: status-role colors, validated with the dataviz skill's checker
# (light surface, adjacent pairs) -- deutan dE 18.3 correct<->incorrect,
# 16.3 incorrect<->no answer; normal-vision floor clear at 23.9.
# The no-answer slot is deliberately achromatic and carries HATCHING as its
# identity channel: it encodes absence of a decision, not a third outcome
# category. Sub-3:1 fills are relieved by direct value labels, as required.

def seed_tallies(model, envs):
    """Per-seed [class][outcome] counts. Seed is the resampling unit for CIs:
    rounds within a session share a hidden parameter and a history, so they are
    not independent and a round-level interval would be far too narrow."""
    per_seed = defaultdict(lambda: {True: [0, 0, 0], False: [0, 0, 0]})
    for env in envs:
        for path in glob.glob(os.path.join(RESPONSES, f"{model}_{env}_seed*.json")):
            seed = int(re.search(r"_seed(\d+)\.json$", path).group(1))
            with open(path) as fh:
                rounds = json.load(fh)["rounds"]
            for r in rounds:
                p = r["oracle_p_act1"]
                if abs(p - 0.5) <= AMBIG:
                    continue
                gt = p > 0.5
                letter = parse_answer_letter(r.get("cot_text"))
                if letter is None:
                    idx = 2
                elif (letter == "B") == gt:
                    idx = 0
                else:
                    idx = 1
                per_seed[seed][gt][idx] += 1
    return per_seed


def balanced_shares(tallies):
    """Class-balanced (correct, incorrect, no-answer) shares, summing to 1."""
    agg = {True: [0, 0, 0], False: [0, 0, 0]}
    for t in tallies:
        for cls in (True, False):
            for i in range(3):
                agg[cls][i] += t[cls][i]
    shares = [0.0, 0.0, 0.0]
    for cls, counts in agg.items():
        n = sum(counts)
        if not n:
            return None
        for i in range(3):
            shares[i] += 0.5 * counts[i] / n
    return shares


def bootstrap_ci(per_seed, n_boot=2000, seed=0):
    """Percentile CI for the CORRECT share, resampling whole sessions."""
    rng = random.Random(seed)
    seeds = list(per_seed)
    draws = []
    for _ in range(n_boot):
        pick = [per_seed[rng.choice(seeds)] for _ in seeds]
        sh = balanced_shares(pick)
        if sh is not None:
            draws.append(sh[0])
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return lo, hi


def main():
    paperstyle.apply()
    # authored at final printed size, so the saved box must NOT be re-cropped
    matplotlib.rcParams.update({
        "savefig.bbox": None,
        "font.size": PT_BASE,
        "xtick.labelsize": PT_BASE,
        "ytick.labelsize": PT_NAME,
        "legend.fontsize": PT_BASE,
        "axes.titlesize": PT_TITLE,
    })
    os.makedirs(OUT, exist_ok=True)

    rows = []           # (kind, label) where kind in {"band", "model"}
    for band, models in BANDS:
        rows.append(("band", band))
        for key, label in models:
            rows.append(("model", (key, label)))

    y_of = {}
    band_y = {}
    y = 0.0
    for kind, payload in rows:
        if kind == "band":
            y -= BAND_GAP
            band_y[payload] = y
        else:
            y_of[payload[0]] = y
            y -= 1.0

    y_lo = min(y_of.values()) - PAD_BOT
    y_hi = PAD_TOP
    ax_h = (y_hi - y_lo) * ROW_IN
    ax_w = FIG_W - LEFT_IN - RIGHT_IN
    top_in = TOP_IN if SHOW_TITLE else 0.0
    fig_h = (top_in + TITLE_STRIP_IN + ax_h + PANEL_GAP_IN + ax_h
             + XTICK_IN + LEGEND_IN + BOT_IN)

    # The 50% chance rule is drawn as a bounded segment, not an axvline: an
    # axvline spans the whole axes and so shot up past the top bar and through
    # the band-header rows. Clamp it to the bar stack itself.
    bar_top = max(y_of.values()) + BAR_H / 2
    bar_bot = min(y_of.values()) - BAR_H / 2

    # stacked: one family per row, sharing the x axis
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, fig_h), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    data = {}
    for ax, (fam_label, envs) in zip(axes, FAMILIES):
        ax.set_facecolor(SURFACE)
        for kind, payload in rows:
            if kind != "model":
                continue
            key, _ = payload
            if (key, fam_label) not in data:
                per_seed = seed_tallies(key, envs)
                data[(key, fam_label)] = (balanced_shares(per_seed.values()),
                                          bootstrap_ci(per_seed))
            shares, (ci_lo, ci_hi) = data[(key, fam_label)]
            yy = y_of[key]
            left = 0.0
            for share, color, hatch in (
                (shares[0], C_CORRECT, None),
                (shares[1], C_WRONG, None),
                (shares[2], C_NOANS, "////"),
            ):
                if share <= 0:
                    continue
                ax.barh(
                    yy, share * 100, left=left * 100, height=BAR_H,
                    color=color, hatch=hatch,
                    edgecolor=SURFACE if hatch is None else "#6f6e6a",
                    linewidth=1.0 if hatch is None else 0.0,
                )
                left += share
            # 95% CI on the correct share, bootstrapped over sessions
            ax.plot([ci_lo * 100, ci_hi * 100], [yy, yy],
                    color=INK, linewidth=0.9, solid_capstyle="butt", zorder=5)
            for xb in (ci_lo * 100, ci_hi * 100):
                ax.plot([xb, xb], [yy - 0.16, yy + 0.16],
                        color=INK, linewidth=0.9, zorder=5)
            # relief rule: the correct share (= effective accuracy) is labelled
            # directly, outside the bar so it never sits on another segment
            ax.text(
                XMAX - 2, yy, f"{shares[0] * 100:.0f}",
                va="center", ha="right", fontsize=PT_BASE, color=INK,
            )

        ax.set_xlim(0, XMAX)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100"], color=INK)
        # centre the family title on the FIGURE, not on the bar area -- the
        # axes start after the name gutter, so an axes-centred title reads as
        # shifted right relative to the suptitle
        ax.set_title(fam_label, color=INK, pad=TITLE_PAD_PT, fontsize=PT_TITLE,
                     fontweight="bold",
                     x=(0.5 * FIG_W - LEFT_IN) / ax_w, ha="center")
        ax.tick_params(length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        # chance for the balanced metric is exactly 50, not 0. Bounded to the
        # bar stack (see bar_top/bar_bot) so it ends flush with the top bar's
        # upper edge instead of running on up through the band headers.
        ax.plot([50, 50], [bar_bot, bar_top], color=INK_MUTED, linewidth=0.7,
                linestyle=(0, (2, 2)), zorder=4, solid_capstyle="butt")

        # y labels: band headers in caps, model names in the left gutter.
        # Both panels carry them now -- stacked panels cannot share a y axis.
        ticks, labels = [], []
        for kind, payload in rows:
            if kind == "model":
                ticks.append(y_of[payload[0]])
                labels.append(payload[1])
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, color=INK)
        ax.set_ylim(y_lo, y_hi)

        for band, yy in band_y.items():
            ax.text(
                -(LEFT_IN - 0.03) / ax_w, yy + BAND_OFF, band,
                transform=ax.get_yaxis_transform(),
                fontsize=PT_BASE, color=INK, ha="left", va="center",
                weight="bold",
            )

    handles = [
        Patch(facecolor=C_CORRECT, edgecolor=SURFACE, label="correct"),
        Patch(facecolor=C_WRONG, edgecolor=SURFACE, label="incorrect"),
        Patch(facecolor=C_NOANS, hatch="////", edgecolor="#6f6e6a", label="no answer"),
        Line2D([0], [0], color=INK, linewidth=0.9, label="95% CI"),
    ]
    # All four keys on ONE row. At 3.3 in this only closes with the handles and
    # the column gaps pulled in; the type stays at the 9pt floor and the row is
    # measured below rather than assumed.
    leg = fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, BOT_IN / fig_h), labelcolor=INK,
        fontsize=PT_BASE, handlelength=1.1, columnspacing=0.8,
        handletextpad=0.35, borderaxespad=0.0, labelspacing=0.4,
    )
    if SHOW_TITLE:
        fig.suptitle(
            "Final decision accuracy",
            fontsize=PT_SUP, color=INK, y=1 - 0.10 / fig_h, va="top",
            fontweight="bold",
        )
    fig.subplots_adjust(
        left=LEFT_IN / FIG_W,
        right=1 - RIGHT_IN / FIG_W,
        top=1 - (top_in + TITLE_STRIP_IN) / fig_h,
        bottom=(BOT_IN + LEGEND_IN + XTICK_IN) / fig_h,
        hspace=PANEL_GAP_IN / ax_h,
    )
    print(f"authored size: {FIG_W:.2f} x {fig_h:.2f} in "
          f"(printed pt = authored pt at width=\\columnwidth)")

    # A one-row legend that overran the canvas would be clipped by LaTeX, and
    # the lower title's grouping depends on a margin ratio that is easy to
    # break by nudging a constant -- so assert both instead of eyeballing them.
    fig.canvas.draw()
    leg_w = leg.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()).width
    print(f"legend row: {leg_w:.2f} in of {FIG_W:.2f} in "
          f"({'fits' if leg_w <= FIG_W else 'OVERRUNS'}) at {PT_BASE:.1f} pt")

    t_h = (axes[1].title.get_window_extent()
           .transformed(fig.dpi_scale_trans.inverted()).height)
    below = TITLE_PAD_PT / 72 + (PAD_TOP + BAND_OFF) * ROW_IN
    above = PANEL_GAP_IN - TITLE_PAD_PT / 72 - t_h - (PAD_BOT - BAR_H / 2) * ROW_IN
    print(f"lower title: {above:.3f} in clear above / {below:.3f} in to its own "
          f"band header ({above / below:.1f}x -- groups downward)")

    for ext in ("pdf", "png"):
        for stem in ("rq1_outcomes", "Decision_final"):
            path = os.path.join(OUT, f"{stem}.{ext}")
            fig.savefig(path, dpi=400)
            print("wrote", path)

    # table view (the relief rule's companion, and the numbers for the caption)
    print(f"\n{'model':16s} " + "  ".join(f"{f[0][:34]:>36s}" for f in FAMILIES))
    print(f"{'':16s} " + "  ".join(f"{'correct [95% CI] / incorr / no-ans':>36s}"
                                   for _ in FAMILIES))
    for _, models in BANDS:
        for key, label in models:
            cells = []
            for fam_label, _ in FAMILIES:
                (c, w, n), (lo, hi) = data[(key, fam_label)]
                cells.append(f"{100*c:5.1f} [{100*lo:4.1f},{100*hi:5.1f}] /{100*w:5.1f} /{100*n:5.1f}")
            print(f"{label:16s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
