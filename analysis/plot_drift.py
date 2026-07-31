"""Drift trajectory plots: oracle vs pre-CoT vs post-CoT preference per round.

Usage:
  python analysis/plot_drift.py                     # plot every responses/*.json
  python analysis/plot_drift.py llama_monty_seed0   # plot one session
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent.parent


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, float), -30, 30)))


def plot_session(resp_path, out_dir):
    data = json.loads(Path(resp_path).read_text())
    sess, rounds = data["session"], data["rounds"]
    ts = [r["t"] for r in rounds]
    oracle = [r["oracle_p_act1"] for r in rounds]
    pre = sigmoid([r["delta_pre_sem"] for r in rounds])
    post_t = [r["t"] for r in rounds if r["post_valid"]]
    post = sigmoid([r["delta_post_sem"] for r in rounds if r["post_valid"]])
    invalid_t = [r["t"] for r in rounds if not r["post_valid"]]

    pre_raw = [r["delta_pre_sem"] for r in rounds]
    post_raw = [r["delta_post_sem"] for r in rounds if r["post_valid"]]
    bias = [r["letter_bias"] for r in rounds]

    fig, (ax1, ax1b, ax2) = plt.subplots(3, 1, figsize=(9, 10), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 2, 1]})

    ax1.plot(ts, oracle, "k--", lw=2, label=f"oracle P({sess['act1']} optimal)")
    ax1.plot(ts, pre, "o-", color="#2CA02C", label="pre-CoT preference (sigmoid gap)")
    if post_t:
        ax1.plot(post_t, post, "s-", color="#D62728", label="post-CoT preference")
    for t in invalid_t:
        ax1.axvline(t, color="gray", alpha=0.25, ls=":")
    ax1.axhline(0.5, color="gray", lw=0.5)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel(f"P(choose {sess['act1']})")
    ax1.set_title(f"{sess['model_key']} | {sess['env']} | seed {sess['seed']} | "
                  f"{sess['hidden_param']}\n"
                  f"Delta_T={sess['summary']['Delta_T']}  "
                  f"format={sess['summary']['format_rate']:.0%}  "
                  f"drift_corr_pre={sess['summary']['drift_corr_pre']}")
    ax1.legend(loc="best", fontsize=8)

    # raw logit scale: distinguishes a graded +/-2-nat preference (evidence-
    # sized, still responsive) from a pinned +/-10-nat prior (saturated).
    # The sigmoid panel above renders both as ~0.999 / ~0.001.
    ax1b.plot(ts, pre_raw, "o-", color="#2CA02C", label="Delta_pre (raw logit gap)")
    if post_t:
        ax1b.plot(post_t, post_raw, "s-", color="#D62728", label="Delta_post (raw)")
    ax1b.plot(ts, bias, "x:", color="#7F7F7F", label="letter bias (nuisance)")
    ax1b.axhline(0, color="k", lw=0.5)
    ax1b.axhspan(-2, 2, color="#2CA02C", alpha=0.07)
    ax1b.text(ts[0], 2.1, "graded zone (|gap| < 2 nats)", fontsize=7, color="#2CA02C")
    ax1b.set_ylabel("logit gap (nats)")
    ax1b.legend(loc="best", fontsize=8)

    amp = [r["delta_amp"] for r in rounds]
    amp_t = [t for t, a in zip(ts, amp) if a is not None]
    amp_v = [a for a in amp if a is not None]
    ax2.bar(amp_t, amp_v, color="#9467BD", alpha=0.8)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("Delta_amp (post - pre)")
    ax2.set_xlabel("round t")

    out = Path(out_dir) / (Path(resp_path).stem + ".png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def main():
    out_dir = HERE / "plots"
    out_dir.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        paths = [HERE / "responses" / f"{sys.argv[1]}.json"]
    else:
        paths = sorted((HERE / "responses").glob("*.json"))
    for p in paths:
        plot_session(p, out_dir)


if __name__ == "__main__":
    main()
