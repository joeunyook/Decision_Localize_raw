"""t-stratified analyses: does anything change with history length?

Recomputes, per (model, env) combo and per round index t:

  probe_by_t.json     probe accuracy at the combo's best layer (from the P1
                      probe JSONs), grouped 10-fold CV over seeds, with the
                      held-out predictions binned by t. Tests whether the
                      BELIEF is decodable at every history depth.
  behavior_by_t.json  "behavior": pre-CoT sign agreement with the oracle,
                      mean |delta_pre_sem| (internal confidence), oracle
                      margin, and fraction of unambiguous rounds, per t.
                      Also per-oracle-class counts and BALANCED agreement
                      (macro-average over the two oracle labels): raw
                      agreement equals the label base rate for a
                      constant-sign responder, and several env x t cells
                      are label-degenerate (ipd t=1,2,10 are 100% one
                      class; trust_game is 75/25 from t=5; monty is
                      62-67% act1 throughout), so raw agreement there
                      measures the prior, not the readout. agree_bal = 0.5
                      for any history-blind constant policy.
                      "rescue": among baseline-wrong unambiguous rounds in
                      the steering replays, how many are flipped to correct
                      by rescue steering at --rescue-alpha, per t.

All stats use the sweep convention: unambiguous = |oracle_p_act1 - 0.5| >
AMBIG_EPS; ambiguous rounds are excluded from probe training, agreement, and
confidence (their labels are noise), but still counted in oracle margin and
frac_unambig.

Usage:
  python analysis/by_t.py --tag P1_v3
  python analysis/by_t.py --tag P1_v3 --out-dir /tmp/check   # don't clobber
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).parent.parent

AMBIG_EPS = 0.10
PROBE_C = 0.1

MODEL_ORDER = ["gemma2b", "llama", "qwen", "dsqwen7", "phi",
               "qwen27", "gemma31", "llama70", "dsllama70"]
ENV_ORDER = ["bandit", "monty", "machine", "ipd", "stag_hunt", "bos", "pennies",
             "trust_game", "chicken"]


def find_combos(residuals_dir, models=None, envs=None):
    combos = set()
    for npy in residuals_dir.glob("*_seed*.npy"):
        model_key, env_name = npy.stem.rsplit("_seed", 1)[0].split("_", 1)
        if models and model_key not in models:
            continue
        if envs and env_name not in envs:
            continue
        combos.add((model_key, env_name))
    order_m = {m: i for i, m in enumerate(MODEL_ORDER)}
    order_e = {e: i for i, e in enumerate(ENV_ORDER)}
    return sorted(combos, key=lambda c: (order_m.get(c[0], 99), order_e.get(c[1], 99)))


def probe_by_t_combo(residuals_dir, results_dir, model_key, env_name, n_folds=10):
    """Grouped k-fold CV at the best layer; per-t accuracy of held-out preds."""
    probe_json = results_dir / f"probe_{model_key}_{env_name}.json"
    best_layer = json.loads(probe_json.read_text())["best_layer"]

    X_parts, y_parts, t_parts, g_parts = [], [], [], []
    for npy in sorted(residuals_dir.glob(f"{model_key}_{env_name}_seed*.npy")):
        seed = int(npy.stem.split("seed")[-1])
        labels_path = npy.with_name(npy.stem + "_labels.json")
        if not labels_path.exists():
            continue
        labels = json.loads(labels_path.read_text())["labels"]
        X = np.load(npy, mmap_mode="r")[:, best_layer, :].astype(np.float32)
        keep = np.array([abs(l["oracle_p_act1"] - 0.5) > AMBIG_EPS for l in labels])
        X_parts.append(X[keep])
        y_parts.append(np.array([l["gt_act1"] for l in labels])[keep])
        t_parts.append(np.array([l["t"] for l in labels])[keep])
        g_parts.append(np.full(keep.sum(), seed))
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    t = np.concatenate(t_parts)
    g = np.concatenate(g_parts)

    correct = np.zeros(len(y), dtype=bool)
    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups=g):
        if len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=PROBE_C)
        clf.fit(X[tr], y[tr])
        correct[te] = clf.predict(X[te]) == y[te]

    per_t = {}
    y = y.astype(bool)
    for tt in sorted(set(t.tolist())):
        m = t == tt
        m1, m0 = m & y, m & ~y
        # acc_bal: macro-average over the two oracle-label classes, the
        # artifact-proof twin of the behavior section's agree_bal; None when
        # a class is empty at this t (label-degenerate cell)
        acc_bal = (float((correct[m1].mean() + correct[m0].mean()) / 2.0)
                   if m1.any() and m0.any() else None)
        per_t[str(tt)] = {"n": int(m.sum()), "acc": float(correct[m].mean()),
                          "n_act1": int(m1.sum()), "n_act0": int(m0.sum()),
                          "acc_bal": acc_bal}
    return {"best_layer": int(best_layer), "per_t": per_t}


def behavior_by_t_combo(responses_dir, model_key, env_name):
    """Per-t agreement / confidence / oracle margin from the sweep responses."""
    acc = defaultdict(lambda: {"n_all": 0, "margin": 0.0, "n": 0,
                               "agree": 0, "conf": 0.0,
                               "n1": 0, "ok1": 0, "n0": 0, "ok0": 0})
    T = 0
    for f in sorted(responses_dir.glob(f"{model_key}_{env_name}_seed*.json")):
        d = json.loads(f.read_text())
        T = max(T, d["session"]["T"])
        for r in d["rounds"]:
            a = acc[r["t"]]
            a["n_all"] += 1
            a["margin"] += abs(r["oracle_p_act1"] - 0.5)
            if abs(r["oracle_p_act1"] - 0.5) <= AMBIG_EPS:
                continue
            if r["delta_pre_sem"] is None:
                continue
            gt = r["oracle_p_act1"] > 0.5
            ok = (r["delta_pre_sem"] > 0) == gt
            a["n"] += 1
            a["agree"] += int(ok)
            a["conf"] += abs(r["delta_pre_sem"])
            a["n1" if gt else "n0"] += 1
            a["ok1" if gt else "ok0"] += int(ok)
    out = {}
    for tt in range(1, T + 1):
        a = acc[tt]
        agree_bal = None
        if a["n1"] and a["n0"]:
            agree_bal = (a["ok1"] / a["n1"] + a["ok0"] / a["n0"]) / 2
        out[str(tt)] = {
            "n": a["n"],
            "agree": a["agree"] / a["n"] if a["n"] else None,
            "mean_conf": a["conf"] / a["n"] if a["n"] else None,
            "mean_oracle_margin": a["margin"] / a["n_all"] if a["n_all"] else None,
            "frac_unambig": a["n"] / a["n_all"] if a["n_all"] else None,
            "n_act1": a["n1"], "n_act0": a["n0"],
            "agree_act1": a["ok1"] / a["n1"] if a["n1"] else None,
            "agree_act0": a["ok0"] / a["n0"] if a["n0"] else None,
            "agree_bal": agree_bal,
        }
    return out


def rescue_by_t_combo(steer_dir, model_key, env_name, alpha):
    """Per-t rescue rate among baseline-wrong unambiguous rounds (steer replays),
    plus a per-t integrity check: how far / how often the replayed baseline gap
    drifts from the stored sweep gap (a sign flip there means the round's
    wrong/correct classification itself is replay-sensitive)."""
    counts = defaultdict(lambda: {"wrong": 0, "rescued": 0})
    drift = defaultdict(lambda: {"n": 0, "sign_flips": 0, "max_drift": 0.0})
    for f in sorted(steer_dir.glob(f"{model_key}_{env_name}_seed*.json")):
        d = json.loads(f.read_text())
        for r in d["rounds"]:
            if not r["unambiguous"]:
                continue
            if r["stored_delta_pre_sem"] is not None:
                dr = drift[r["t"]]
                dr["n"] += 1
                dr["sign_flips"] += int((r["gaps"]["baseline"] > 0)
                                        != (r["stored_delta_pre_sem"] > 0))
                dr["max_drift"] = max(dr["max_drift"],
                                      abs(r["gaps"]["baseline"] - r["stored_delta_pre_sem"]))
            if (r["gaps"]["baseline"] > 0) == r["gt_act1"]:
                continue
            c = counts[r["t"]]
            c["wrong"] += 1
            if (r["gaps"][f"rescue_a{alpha:g}"] > 0) == r["gt_act1"]:
                c["rescued"] += 1
    rescue = {str(tt): {"wrong": c["wrong"], "rescued": c["rescued"],
                        "rate": c["rescued"] / c["wrong"]}
              for tt, c in sorted(counts.items())}
    drift_out = {str(tt): {"n": v["n"], "sign_flips": v["sign_flips"],
                           "max_drift": v["max_drift"]}
                 for tt, v in sorted(drift.items())}
    return rescue, drift_out


def main():
    ap = argparse.ArgumentParser(description="t-stratified probe/behavior/rescue")
    ap.add_argument("--tag", default="P1_v3")
    ap.add_argument("--residuals-dir", default="residuals")
    ap.add_argument("--responses-dir", default="responses")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--rescue-alpha", type=float, default=4.0)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default results/{tag}; pass another dir "
                         "to compare against stored results without clobbering)")
    args = ap.parse_args()

    residuals_dir = (HERE / args.residuals_dir).resolve()
    responses_dir = (HERE / args.responses_dir).resolve()
    results_dir = HERE / "results" / args.tag
    steer_dir = HERE / "steer" / args.tag
    out_dir = Path(args.out_dir).resolve() if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = find_combos(residuals_dir, args.models, args.envs)

    probe_out, behavior_out, rescue_out, drift_out = {}, {}, {}, {}
    for model_key, env_name in combos:
        key = f"{model_key}_{env_name}"
        print(f"=== {key} ===")
        probe_out[key] = probe_by_t_combo(residuals_dir, results_dir,
                                          model_key, env_name, args.n_folds)
        behavior_out[key] = behavior_by_t_combo(responses_dir, model_key, env_name)
        rescue_out[key], drift_out[key] = rescue_by_t_combo(
            steer_dir, model_key, env_name, args.rescue_alpha)
        accs = [v["acc"] for v in probe_out[key]["per_t"].values()]
        agrees = [v["agree"] for v in behavior_out[key].values() if v["agree"] is not None]
        print(f"  probe acc min/max {min(accs):.2f}/{max(accs):.2f}  "
              f"agreement min/max {min(agrees):.2f}/{max(agrees):.2f}  "
              f"wrong rounds (steer) {sum(c['wrong'] for c in rescue_out[key].values())}")

    (out_dir / "probe_by_t.json").write_text(json.dumps(probe_out, indent=1))
    (out_dir / "behavior_by_t.json").write_text(
        json.dumps({"behavior": behavior_out, "rescue": rescue_out,
                    "replay_drift": drift_out}, indent=1))
    print(f"\n{len(combos)} combos -> {out_dir}/probe_by_t.json, behavior_by_t.json")


if __name__ == "__main__":
    main()
