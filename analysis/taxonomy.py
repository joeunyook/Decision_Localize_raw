"""Failure-taxonomy classifier.

Reads every responses/*.json session and assigns per-session (and per-round)
failure labels from the five-way taxonomy:

  BELIEF_UPDATE_FAILURE : pre-CoT preference does not track the oracle
                          trajectory although the oracle moves substantially.
                          (drift_corr_pre low / negative while oracle_range large)
  ACTION_MAPPING_FAILURE: pre-CoT preference tracks the oracle (belief present)
                          but the behavioral decision is wrong on unambiguous
                          rounds. The model knows the state, maps it to the
                          wrong action.
  COT_RATIONALIZATION   : pre-CoT gap is weak or wrong-signed, but the post-CoT
                          gap is large in the same wrong direction -- generated
                          reasoning amplified a wrong/absent prior.
  COT_CORRUPTION        : pre-CoT gap is correct-signed on a round, but the
                          post-CoT decision flips to the wrong action -- the
                          reasoning process destroyed a good latent belief.
  FORMAT_ARTIFACT       : the model never generated 'Answer: A/B'; the round
                          carries no valid post-CoT measurement. Reported
                          separately -- this is a measurement artifact, NOT a
                          scientific finding.

Thresholds are deliberately conservative and printed with every report so
reviewers can audit them.
"""

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent.parent

# --- thresholds (audit these) ---
AMBIG_EPS = 0.10        # |oracle - 0.5| <= eps -> round is ambiguous, skipped
ORACLE_RANGE_MIN = 0.25  # oracle must move at least this much for drift claims
CORR_TRACKING_MIN = 0.40  # drift_corr_pre above this counts as 'tracking'
WEAK_PRE = 1.0          # |delta_pre| below this counts as 'weak prior'
STRONG_POST = 3.0       # |delta_post| above this counts as 'confident'


def classify_round(r):
    labels = []
    p = r["oracle_p_act1"]
    if abs(p - 0.5) <= AMBIG_EPS:
        return ["AMBIGUOUS"]
    gt = p > 0.5
    pre_sign_correct = (r["delta_pre_sem"] > 0) == gt

    if not r["post_valid"]:
        labels.append("FORMAT_ARTIFACT")
        return labels

    decision_correct = r["decision_act1"] == gt
    post = r["delta_post_sem"]

    if pre_sign_correct and not decision_correct:
        labels.append("COT_CORRUPTION")
    if (abs(r["delta_pre_sem"]) < WEAK_PRE or not pre_sign_correct) \
            and abs(post) > STRONG_POST and not decision_correct:
        labels.append("COT_RATIONALIZATION")
    if not pre_sign_correct and not decision_correct:
        labels.append("WRONG_BEFORE_AND_AFTER")
    if decision_correct and not labels:
        labels.append("CORRECT")
    return labels or ["OTHER"]


def classify_session(sess, rounds):
    s = sess["summary"]
    labels = []
    if s["oracle_range"] >= ORACLE_RANGE_MIN:
        corr = s["drift_corr_pre"]
        if corr is None or corr < CORR_TRACKING_MIN:
            labels.append("BELIEF_UPDATE_FAILURE")
        elif s["Delta_T"] is not None and s["Delta_T"] > 0.3:
            labels.append("ACTION_MAPPING_FAILURE")
    if s["format_rate"] < 0.7:
        labels.append("FORMAT_UNRELIABLE")
    if not labels:
        labels.append("HEALTHY")
    return labels


def main():
    report = {}
    for path in sorted((HERE / "responses").glob("*.json")):
        data = json.loads(path.read_text())
        sess, rounds = data["session"], data["rounds"]
        round_labels = Counter()
        for r in rounds:
            for lab in classify_round(r):
                round_labels[lab] += 1
        sess_labels = classify_session(sess, rounds)
        key = path.stem
        report[key] = {
            "session_labels": sess_labels,
            "round_labels": dict(round_labels),
            "Delta_T": sess["summary"]["Delta_T"],
            "drift_corr_pre": sess["summary"]["drift_corr_pre"],
            "format_rate": sess["summary"]["format_rate"],
        }
        print(f"{key:40s}  {','.join(sess_labels):30s}  rounds={dict(round_labels)}")

    out = HERE / "results" / "taxonomy_report.json"
    out.write_text(json.dumps({
        "thresholds": {"AMBIG_EPS": AMBIG_EPS, "ORACLE_RANGE_MIN": ORACLE_RANGE_MIN,
                       "CORR_TRACKING_MIN": CORR_TRACKING_MIN, "WEAK_PRE": WEAK_PRE,
                       "STRONG_POST": STRONG_POST},
        "sessions": report,
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
