#!/usr/bin/env python3
"""Narrative-framing ablation: does the decision behaviour depend on story tone?

Every scenario prompt exists in two variants that are numerically identical --
same hidden-type descriptions, same payoffs/rates, same rendered history, same
computed tally block, same answer-format block -- and differ only in narrative
framing:

  canonical  slot machines, hunters, trustees, a game-show host, an opponent
  dry        an abstract probability / decision-theory problem statement

(prompts/{env}_forced.txt vs prompts/{env}_forced_dry.txt, and the matching
*_cot*.txt pair.)  If the belief-to-action failures were an artifact of
anthropomorphic story framing rather than of the decision structure, stripping
the narrative should move the numbers.

Three matched comparisons, each on the SAME seeds in both arms:

  P1 Belief    balanced linear-probe accuracy at the final layer
               canonical results/P1_v4/probe_*.json (100 seeds)
               dry       results/ABL_dry_v1/probe_*.json (30 seeds)
               NOTE the training-set sizes differ; see the caveat printed with
               the table.  The behavioural comparisons below are exactly matched.
  P2 Pre-CoT   balanced agreement of the pre-CoT preference sign, seeds 0-29
               canonical responses/            dry ABL_dry_v1/responses/
  P3 Post-CoT  balanced agreement of the parsed CoT answer, seeds 0-14
               canonical P3_postcot/responses/ dry ABL_dry_cot_v1/responses/

Balanced = macro-average over the two baseline-optimal action classes (chance
50), computed on unambiguous steps only (|p*-0.5| > 0.1), matching the metric
in the main results table.  Post-CoT answers are ALWAYS re-parsed from cot_text
with the current last-match / \\boxed{} regex; the stored post_valid flag
predates that fix for some models and is never trusted (DOCUMENTATION.txt, Bug 5).

IPD is excluded from all totals (dropped from the paper: horizon-dependent
labels), but printed as a separate row for completeness.

  python3 analysis/ablation_dry.py
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = ["gemma2b", "llama", "phi", "qwen", "dsqwen7", "qwen27", "gemma31",
          "llama70", "dsllama70"]
LABEL = {
    "gemma2b": "Gemma-2 2B", "llama": "Llama-3.2 3B", "phi": "Phi-4-mini 3.8B",
    "qwen": "Qwen-3.5 4B", "dsqwen7": "R1-Qwen 7B", "qwen27": "Qwen-3.6 27B",
    "gemma31": "Gemma-4 31B", "llama70": "Llama-3.3 70B", "dsllama70": "R1-Llama 70B",
}
BAYES = ["bandit", "monty", "trust_game", "machine"]
GAME = ["stag_hunt", "chicken", "bos", "pennies"]
ENVS = BAYES + GAME
ENV_LABEL = {"bandit": "Bandit", "monty": "Monty Hall", "trust_game": "Trust Game",
             "machine": "Machine", "stag_hunt": "Stag Hunt", "chicken": "Chicken",
             "bos": "BoS", "pennies": "Pennies", "ipd": "IPD"}

AMBIG = 0.1
ANSWER_RE = re.compile(r"Answer\s*:\s*([AB])\b|\\boxed\{([AB])\}")


def parse_answer_letter(text):
    """Last-match parse, identical to core/probes.parse_answer_letter."""
    if not text:
        return None
    m = list(ANSWER_RE.finditer(text))
    return (m[-1].group(1) or m[-1].group(2)) if m else None


def balanced(pairs):
    """pairs = [(gt_act1, model_act1_or_None)]; non-answers count as wrong.

    Returns (balanced_acc, coverage, n) with balanced_acc in percent, or
    (None, None, 0) when a class is empty.
    """
    acc = {}
    cov_hit = cov_n = 0
    for gt, got in pairs:
        cls = acc.setdefault(bool(gt), [0, 0])
        cls[1] += 1
        if got is not None:
            cov_hit += 1
            if bool(got) == bool(gt):
                cls[0] += 1
        cov_n += 1
    if len(acc) < 2 or any(v[1] == 0 for v in acc.values()):
        return None, None, cov_n
    b = 100.0 * sum(v[0] / v[1] for v in acc.values()) / len(acc)
    return b, 100.0 * cov_hit / max(cov_n, 1), cov_n


def load_pairs(resp_dir, model, env, seeds, mode):
    """mode 'pre' -> sign of delta_pre_sem; 'post' -> re-parsed CoT letter."""
    out = []
    for s in seeds:
        p = os.path.join(resp_dir, f"{model}_{env}_seed{s}.json")
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for r in d.get("rounds", []):
            op = r.get("oracle_p_act1")
            if op is None or abs(op - 0.5) <= AMBIG:
                continue
            gt = op > 0.5
            if mode == "pre":
                g = r.get("delta_pre_sem")
                got = None if g is None else (g > 0)
            else:
                letter = parse_answer_letter(r.get("cot_text") or "")
                got = None if letter is None else (letter == "B")
            out.append((gt, got))
    return out


def probe_bal(tag, model, env):
    """Final-layer BALANCED probe accuracy, percent, or None if unavailable."""
    for t in (tag, "P1_v4_layerbal") if tag == "P1_v4" else (tag,):
        p = os.path.join(ROOT, "results", t, f"probe_{model}_{env}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        lab = d.get("layer_accuracy_bal")
        if lab:
            top = max(int(k) for k in lab)
            return 100.0 * lab[str(top)]
    return None


def fmt(v, w=6, nd=1):
    return f"{v:{w}.{nd}f}" if v is not None else " " * (w - 1) + "-"


def table(title, rows, head):
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)
    print(head)
    for r in rows:
        print(r)


def main():
    pre_seeds = list(range(30))
    post_seeds = list(range(15))

    # ---------------------------------------------------------------- P2
    p2 = {}
    for m in MODELS:
        for e in ENVS + ["ipd"]:
            can = balanced(load_pairs(os.path.join(ROOT, "responses"), m, e, pre_seeds, "pre"))[0]
            dry = balanced(load_pairs(os.path.join(ROOT, "ABL_dry_v1", "responses"), m, e, pre_seeds, "pre"))[0]
            p2[(m, e)] = (can, dry)

    # ---------------------------------------------------------------- P1
    p1 = {}
    for m in MODELS:
        for e in ENVS + ["ipd"]:
            p1[(m, e)] = (probe_bal("P1_v4", m, e), probe_bal("ABL_dry_v1", m, e))

    # ---------------------------------------------------------------- P3
    p3 = {}
    for m in MODELS:
        for e in ENVS + ["ipd"]:
            cb, cc, cn = balanced(load_pairs(os.path.join(ROOT, "P3_postcot", "responses"), m, e, post_seeds, "post"))
            db, dc, dn = balanced(load_pairs(os.path.join(ROOT, "ABL_dry_cot_v1", "responses"), m, e, post_seeds, "post"))
            p3[(m, e)] = (cb, db, cc, dc, cn, dn)

    def famrows(store, idx_can, idx_dry, envs):
        rows = []
        for m in MODELS:
            cs = [store[(m, e)][idx_can] for e in envs if store[(m, e)][idx_can] is not None]
            ds = [store[(m, e)][idx_dry] for e in envs if store[(m, e)][idx_dry] is not None]
            if not cs or not ds:
                rows.append((m, None, None, None))
                continue
            c, d = sum(cs) / len(cs), sum(ds) / len(ds)
            rows.append((m, c, d, d - c))
        return rows

    for name, store, ic, idr in (("P1 Belief (balanced probe, final layer)", p1, 0, 1),
                                 ("P2 Pre-CoT (balanced agreement, seeds 0-29)", p2, 0, 1),
                                 ("P3 Post-CoT (balanced accuracy, seeds 0-14, non-answer = wrong)", p3, 0, 1)):
        head = (f"{'model':<18}{'BAYESIAN':>26}   {'GAME THEORETIC':>26}   {'ALL 8':>26}\n"
                f"{'':<18}{'canon':>8}{'dry':>8}{'delta':>8}   "
                f"{'canon':>8}{'dry':>8}{'delta':>8}   {'canon':>8}{'dry':>8}{'delta':>8}")
        rows = []
        allb, allg, alla = [], [], []
        for m in MODELS:
            line = f"{LABEL[m]:<18}"
            for envs, acc in ((BAYES, allb), (GAME, allg), (ENVS, alla)):
                cs = [store[(m, e)][ic] for e in envs if store[(m, e)][ic] is not None]
                ds = [store[(m, e)][idr] for e in envs if store[(m, e)][idr] is not None]
                if cs and ds and len(cs) == len(ds):
                    c, d = sum(cs) / len(cs), sum(ds) / len(ds)
                    acc.append((c, d))
                    line += f"{c:8.1f}{d:8.1f}{d - c:+8.1f}   "
                else:
                    line += f"{'-':>8}{'-':>8}{'-':>8}   "
            rows.append(line.rstrip())
        line = f"{'MEAN':<18}"
        for acc in (allb, allg, alla):
            if acc:
                c = sum(a for a, _ in acc) / len(acc)
                d = sum(b for _, b in acc) / len(acc)
                line += f"{c:8.1f}{d:8.1f}{d - c:+8.1f}   "
        rows.append("-" * 100)
        rows.append(line.rstrip())
        table(name, rows, head)

    # per-scenario deltas
    print("\n" + "=" * 108)
    print("PER SCENARIO (mean over 9 models; delta = dry - canonical, percentage points)")
    print("=" * 108)
    print(f"{'scenario':<14}{'P1 canon':>10}{'P1 dry':>9}{'d':>7}   "
          f"{'P2 canon':>10}{'P2 dry':>9}{'d':>7}   {'P3 canon':>10}{'P3 dry':>9}{'d':>7}")
    for e in ENVS + ["ipd"]:
        line = f"{ENV_LABEL[e]:<14}"
        for store, ic, idr in ((p1, 0, 1), (p2, 0, 1), (p3, 0, 1)):
            cs = [store[(m, e)][ic] for m in MODELS if store[(m, e)][ic] is not None]
            ds = [store[(m, e)][idr] for m in MODELS if store[(m, e)][idr] is not None]
            if cs and ds:
                c, d = sum(cs) / len(cs), sum(ds) / len(ds)
                line += f"{c:10.1f}{d:9.1f}{d - c:+7.1f}   "
            else:
                line += f"{'-':>10}{'-':>9}{'-':>7}   "
        print(line.rstrip() + ("      (excluded from the paper)" if e == "ipd" else ""))

    # coverage, since a framing change could shift termination rather than accuracy
    print("\n" + "=" * 108)
    print("P3 COVERAGE (% of unambiguous steps that terminate with a parseable answer, 8 scenarios)")
    print("=" * 108)
    print(f"{'model':<18}{'canon':>9}{'dry':>9}{'delta':>9}")
    for m in MODELS:
        cs = [p3[(m, e)][2] for e in ENVS if p3[(m, e)][2] is not None]
        ds = [p3[(m, e)][3] for e in ENVS if p3[(m, e)][3] is not None]
        if cs and ds:
            c, d = sum(cs) / len(cs), sum(ds) / len(ds)
            print(f"{LABEL[m]:<18}{c:9.1f}{d:9.1f}{d - c:+9.1f}")

    # headline scatter of the individual cells
    dl = [store_delta for store, ic, idr in ((p2, 0, 1),)
          for m in MODELS for e in ENVS
          if store[(m, e)][ic] is not None and store[(m, e)][idr] is not None
          for store_delta in [store[(m, e)][idr] - store[(m, e)][ic]]]
    if dl:
        dl_sorted = sorted(dl)
        n = len(dl)
        print("\nP2 cell-level deltas (n=%d model-scenario cells): mean %+.2f pp, "
              "median %+.2f, |mean| %.2f, min %+.1f, max %+.1f"
              % (n, sum(dl) / n, dl_sorted[n // 2], sum(abs(x) for x in dl) / n,
                 dl_sorted[0], dl_sorted[-1]))
        print("cells moving more than 10 pp: %d / %d" % (sum(1 for x in dl if abs(x) > 10), n))

    print("\nCAVEAT: the P1 arms differ in probe training-set size (100 vs 30 seeds), "
          "so P1 deltas conflate framing with sample size; P2 and P3 are seed-matched "
          "and are the comparisons to quote.")


if __name__ == "__main__":
    sys.exit(main())
