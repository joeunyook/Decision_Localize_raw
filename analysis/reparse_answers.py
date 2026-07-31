"""Offline re-parse of already-collected P3 post-CoT responses using the
corrected parse_answer_letter() (last-match, not first-match -- see
core/probes.py). Does NOT mutate anything under P3_postcot/ -- recomputes
parsed_letter/decision_act1 from the already-saved cot_text at analysis time,
same pattern already used for the earlier stale-\\boxed{} regex fix.

NOTE on scope: this recovers the correct DECISION (parsed_letter,
decision_act1, hence accuracy/agreement) for every affected round, because
that only depends on the text. It does NOT recover gap_raw/delta_post_sem/
delta_amp/drift_corr_post for rounds where the old first-match letter
differed from the corrected last-match letter -- those were computed by
locating a generation step for the WRONG letter's token, and the raw
per-step logits were never saved to disk, so the true confidence gap at the
real answer step is not reconstructable offline. Any confidence/logit-gap
analysis should treat those specific rounds as suspect (flagged below via
gap_suspect) until/unless they are regenerated.

Usage: ~/bin/micromamba run -n ftdiag python analysis/reparse_answers.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from core.probes import parse_answer_letter  # noqa: E402

RESPONSES_DIR = HERE / "P3_postcot" / "responses"


def reparse_all():
    """Returns list of dicts, one per round, with old and corrected fields."""
    rows = []
    for f in sorted(RESPONSES_DIR.glob("*.json")):
        model, env, seed_part = f.stem.rsplit("_", 2)[0], None, None
        parts = f.stem.split("_")
        model_key = parts[0]
        seed = int(parts[-1].replace("seed", ""))
        env_name = "_".join(parts[1:-1])
        d = json.load(open(f))
        for r in d["rounds"]:
            text = r.get("cot_text")
            if not text or not r.get("post_valid"):
                continue
            old_letter = r["parsed_letter"]
            new_letter = parse_answer_letter(text)
            oracle_p = r["oracle_p_act1"]
            oracle_letter = "B" if oracle_p > 0.5 else "A"
            changed = new_letter != old_letter
            rows.append({
                "model": model_key, "env": env_name, "seed": seed, "t": r["t"],
                "old_letter": old_letter, "new_letter": new_letter,
                "changed": changed,
                "old_decision_act1": r["decision_act1"],
                "new_decision_act1": (new_letter == "B"),
                "oracle_letter": oracle_letter,
                "old_correct": (old_letter == oracle_letter),
                "new_correct": (new_letter == oracle_letter),
                "gap_suspect": changed,  # gap_raw/delta_post_sem read the wrong token step
            })
    return rows


def summarize(rows):
    by_model = defaultdict(lambda: {"n": 0, "changed": 0,
                                     "old_correct": 0, "new_correct": 0})
    for r in rows:
        b = by_model[r["model"]]
        b["n"] += 1
        b["changed"] += r["changed"]
        b["old_correct"] += r["old_correct"]
        b["new_correct"] += r["new_correct"]
    return by_model


if __name__ == "__main__":
    rows = reparse_all()
    by_model = summarize(rows)

    print(f"{'model':12s} {'n_rounds':>9s} {'changed':>8s} {'pct':>6s}  "
          f"{'acc_old':>8s} {'acc_new':>8s}  {'delta':>7s}")
    for model in sorted(by_model, key=lambda m: -by_model[m]["changed"]):
        b = by_model[model]
        acc_old = b["old_correct"] / b["n"] * 100
        acc_new = b["new_correct"] / b["n"] * 100
        print(f"{model:12s} {b['n']:9d} {b['changed']:8d} "
              f"{b['changed']/b['n']*100:5.1f}%  "
              f"{acc_old:7.2f}% {acc_new:7.2f}%  {acc_new-acc_old:+6.2f}pp")

    total = sum(b["n"] for b in by_model.values())
    total_changed = sum(b["changed"] for b in by_model.values())
    print(f"\nTotal: {total} valid rounds scanned, {total_changed} "
          f"({total_changed/total*100:.1f}%) had a corrected parsed_letter.")
    print("All 'changed' rounds have gap_raw/delta_post_sem/drift_corr_post "
          "flagged suspect (not recoverable offline) -- see module docstring.")

    out_path = HERE / "analysis" / "reparse_answers_output.json"
    with open(out_path, "w") as f:
        json.dump(rows, f)
    print(f"\nFull per-round corrected table written to {out_path}")
