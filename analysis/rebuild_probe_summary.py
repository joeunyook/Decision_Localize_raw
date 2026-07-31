"""Rebuild results/{tag}/probe_summary.json from the per-combo probe_*.json files.

WHY THIS EXISTS: probe_train.py writes probe_summary.json from only the combos in
its *current* invocation (probe_train.py:305). Running it for a single model or a
single env therefore REPLACES the summary rather than extending it -- a per-model
launch loop leaves a summary containing only the last model. The per-combo
probe_{model}_{env}.json files are always complete, so the summary is fully
regenerable; run this after any partial probe_train run.

Usage:
  python analysis/rebuild_probe_summary.py --tag P1_v4
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FIELDS = ("model", "env", "majority_baseline", "best_layer", "best_accuracy",
          "best_accuracy_bal", "final_layer_accuracy", "margin_over_baseline",
          "n_rounds_used", "n_seeds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="P1_v4")
    ap.add_argument("--results-root", default=str(HERE / "results"))
    args = ap.parse_args()

    results_dir = Path(args.results_root) / args.tag
    rows = []
    # probe_summary.json (a list) and probe_by_t.json (keyed by model_env) also
    # match probe_*.json but are not per-combo records -- skip anything that
    # isn't a dict carrying a "model" key.
    skipped = []
    for f in sorted(results_dir.glob("probe_*.json")):
        if f.name in ("probe_summary.json", "probe_by_t.json"):
            continue
        d = json.loads(f.read_text())
        if not isinstance(d, dict) or "model" not in d:
            skipped.append(f.name)
            continue
        rows.append({k: d[k] for k in FIELDS if k in d})

    rows.sort(key=lambda r: (r.get("model", ""), r.get("env", "")))
    out = results_dir / "probe_summary.json"
    out.write_text(json.dumps(rows, indent=2))

    models = sorted({r["model"] for r in rows})
    envs = sorted({r["env"] for r in rows})
    missing_bal = [f"{r['model']}/{r['env']}" for r in rows if "best_accuracy_bal" not in r]
    print(f"rebuilt {out}")
    print(f"  combos : {len(rows)}  ({len(models)} models x up to {len(envs)} envs)")
    print(f"  models : {', '.join(models)}")
    print(f"  envs   : {', '.join(envs)}")
    if skipped:
        print(f"  skipped: {len(skipped)} non-combo files")
    if missing_bal:
        print(f"  NOTE   : {len(missing_bal)} combos predate best_accuracy_bal "
              f"(retrain to populate): {', '.join(missing_bal[:6])}"
              + (" ..." if len(missing_bal) > 6 else ""))


if __name__ == "__main__":
    main()
