"""P4 summary: pool steer/{tag}/*.json into rescue / break / random-control rates.

Definitions (unambiguous rounds only, |oracle - 0.5| > 0.1):
  wrong rounds    baseline semantic gap disagrees with the oracle sign
  correct rounds  baseline agrees
  rescue rate     wrong rounds flipped to correct by rescue steering, per alpha
  break rate      correct rounds flipped to wrong by anti steering, per alpha
  random rate     wrong rounds flipped by random-direction steering, per alpha
                  (mean over the random controls)

A causal belief->action account predicts rescue >> random and a monotone
dose-response in alpha; break rate rising symmetrically shows the direction is
two-way, not a ceiling artifact.

Usage:
  python analysis/steer_summary.py --tag P1_v3
  python analysis/steer_summary.py --tag P1_v3 --models llama qwen --envs ipd
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="P1_v3")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--envs", nargs="*", default=None)
    args = ap.parse_args()

    steer_dir = HERE / "steer" / args.tag
    files = sorted(steer_dir.glob("*_seed*.json"))
    if not files:
        raise SystemExit(f"no steering results in {steer_dir}")

    # pools[(model, env)] -> dict of counters
    pools = defaultdict(lambda: {"wrong": 0, "correct": 0, "drift_max": 0.0,
                                 "rescued": defaultdict(int),
                                 "broken": defaultdict(int),
                                 "rand_flip": defaultdict(float),
                                 "alphas": None, "n_random": 0, "n_sessions": 0})
    for f in files:
        d = json.loads(f.read_text())
        m, e = d["model_key"], d["env"]
        if args.models and m not in args.models:
            continue
        if args.envs and e not in args.envs:
            continue
        p = pools[(m, e)]
        p["alphas"] = d["alphas"]
        p["n_random"] = d["n_random"]
        p["n_sessions"] += 1
        for r in d["rounds"]:
            if not r["unambiguous"]:
                continue
            if r["stored_delta_pre_sem"] is not None:
                p["drift_max"] = max(p["drift_max"],
                                     abs(r["gaps"]["baseline"] - r["stored_delta_pre_sem"]))
            base_ok = (r["gaps"]["baseline"] > 0) == r["gt_act1"]
            if base_ok:
                p["correct"] += 1
                for a in d["alphas"]:
                    if (r["gaps"][f"anti_a{a:g}"] > 0) != r["gt_act1"]:
                        p["broken"][a] += 1
            else:
                p["wrong"] += 1
                for a in d["alphas"]:
                    if (r["gaps"][f"rescue_a{a:g}"] > 0) == r["gt_act1"]:
                        p["rescued"][a] += 1
                    flips = sum((r["gaps"][f"rand{j}_a{a:g}"] > 0) == r["gt_act1"]
                                for j in range(d["n_random"]))
                    p["rand_flip"][a] += flips / max(d["n_random"], 1)

    out = []
    hdr = f"{'model':10s} {'env':11s} {'sess':>4s} {'wrong':>5s} {'corr':>5s}  " \
          f"{'alpha':>5s} {'rescue':>7s} {'random':>7s} {'break':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for (m, e), p in sorted(pools.items()):
        for a in p["alphas"]:
            rescue = p["rescued"][a] / p["wrong"] if p["wrong"] else None
            rand = p["rand_flip"][a] / p["wrong"] if p["wrong"] else None
            broke = p["broken"][a] / p["correct"] if p["correct"] else None
            fmt = lambda x: f"{x:6.1%}" if x is not None else "     -"
            print(f"{m:10s} {e:11s} {p['n_sessions']:4d} {p['wrong']:5d} "
                  f"{p['correct']:5d}  {a:5g} {fmt(rescue):>7s} {fmt(rand):>7s} "
                  f"{fmt(broke):>7s}")
            out.append({"model": m, "env": e, "alpha": a,
                        "n_sessions": p["n_sessions"],
                        "n_wrong": p["wrong"], "n_correct": p["correct"],
                        "rescue_rate": rescue, "random_rate": rand,
                        "break_rate": broke,
                        "baseline_replay_max_drift": p["drift_max"]})
        if p["drift_max"] > 0.5:
            print(f"  [warn] {m}/{e}: baseline replay drifts from the stored sweep "
                  f"by up to {p['drift_max']:.2f} nats -- check determinism")
    out_path = HERE / "steer" / args.tag / "steer_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
