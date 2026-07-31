"""Shared matplotlib settings for every paper figure. Import before plotting.

    from paperstyle import apply, FIG_WIDE, FIG_COL

Why each setting is here (do not drop these without a reason):

  pdf.fonttype = 42   matplotlib's DEFAULT IS 3. Type 3 fonts are rejected by
                      AAAI/IEEE/ACM PDF checks and cannot be searched or copied
                      from the compiled paper. 42 = TrueType, embedded.
  figure sizes        Author each figure at its FINAL printed size. A 11-inch
                      figure dropped into a 7-inch column is scaled 0.64x by
                      LaTeX, so 8.5pt tick labels arrive at 5.4pt. Sizes below
                      are the real AAAI two-column text widths.
  font sizes          Chosen so labels land at 7-9pt AT FINAL SIZE, i.e. close
                      to the 9pt body text, with no \\includegraphics scaling.
  serif family        Liberation Serif is metric-compatible with Times, which is
                      what the AAAI style file sets for body text.
"""
import matplotlib as mpl

# AAAI two-column geometry, in inches
FIG_COL = 3.3      # single column  -> \begin{figure}
FIG_WIDE = 7.0     # full text width -> \begin{figure*}

# Status-role fills, validated for CVD separation on a light surface
# (deutan dE 18.3 correct<->incorrect, 16.3 incorrect<->no answer).
C_CORRECT = "#7fc97f"
C_WRONG = "#d03b3b"
C_NOANS = "#a8a7a2"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"


def apply():
    mpl.rcParams.update({
        # --- output ---------------------------------------------------------
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        # --- type -----------------------------------------------------------
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "STIXGeneral",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 17.0,
        "axes.titlesize": 21.0,
        "axes.labelsize": 17.0,
        "xtick.labelsize": 17.0,
        "ytick.labelsize": 17.0,
        "legend.fontsize": 15.4,
        # --- marks ----------------------------------------------------------
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.edgecolor": INK_MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
    })
