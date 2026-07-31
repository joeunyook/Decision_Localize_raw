"""Pre-cluster smoke test: envs, oracles, stochasticity, prompts, metrics.

CPU-only (no torch, no models). Exercises the exact code paths run_all.py uses,
minus the model forward passes. Run before launching the wide seed sweep:

  python smoke_test.py                # default: 40 seeds, T=10
  python smoke_test.py --seeds 20

Exit code 0 = all checks passed. Each check prints PASS/FAIL + diagnostics.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from envs import make_env, ENVS
from core.metrics import summarize_session, sigmoid, spearman

AMBIG_EPS = 0.10
FAILURES = []

# kept in sync with run_all.py's ANSWER_FORMAT_LETTER/ANSWER_FORMAT_BOXED --
# duplicated (not imported) so this file stays torch-free.
ANSWER_FORMAT_LETTER = "Write your final answer — exactly one of:\n     Answer: A\n     Answer: B"


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def simulate(env_name, seed, T):
    """Replicate run_all.py's env loop exactly (no model)."""
    rng = np.random.default_rng(seed)
    env = make_env(env_name, rng, T, seed=seed)
    records, oracle, ctxs, histories = [], [], [], []
    for t in range(1, T + 1):
        ctx = env.new_round(t)
        histories.append(env.format_history(records))
        oracle.append(env.oracle_p_act1(records))
        ctxs.append(ctx)
        _, rec = env.outcome(t, ctx, env.behavior_action(t))
        records.append(rec)
    return env, records, oracle, ctxs, histories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()
    N, T = args.seeds, args.rounds
    seeds = list(range(N))

    print(f"smoke test: {len(ENVS)} envs x {N} seeds x T={T}\n")

    # simulate everything once
    sims = {}  # (env, seed) -> (env_obj, records, oracle, ctxs, histories)
    for env_name in ENVS:
        for s in seeds:
            sims[(env_name, s)] = simulate(env_name, s, T)

    # ---------------------------------------------------------------- 1
    print("1. determinism: same seed twice -> identical session")
    for env_name in ENVS:
        env1, rec1, or1, _, _ = simulate(env_name, 3, T)
        env2, rec2, or2, _, _ = simulate(env_name, 3, T)
        same = (env1.hidden_param() == env2.hidden_param()
                and json.dumps(rec1, default=str) == json.dumps(rec2, default=str)
                and or1 == or2)
        check(f"{env_name}: seed 3 reproducible", same)

    # ---------------------------------------------------------------- 2
    print("\n2. stochasticity: hidden params + histories vary across seeds")
    for env_name in ENVS:
        params = [sims[(env_name, s)][0].hidden_param() for s in seeds]
        n_distinct = len(set(params))
        check(f"{env_name}: >=2 distinct hidden params over {N} seeds",
              n_distinct >= 2, f"{n_distinct} distinct: {sorted(set(params))}")
        # two seeds sharing a hidden param must still differ in history
        by_param = {}
        for s, p in zip(seeds, params):
            by_param.setdefault(p, []).append(s)
        # a duplicated param whose histories differ shows per-session
        # stochasticity beyond the hidden draw. Some params are legitimately
        # deterministic generators (trust_game RECIPROCATOR/EXPLOITER always
        # repay/withhold), so pass if ANY duplicated group differs.
        dups = [v for v in by_param.values() if len(v) >= 2]
        if dups:
            diff = [v for v in dups
                    if json.dumps(sims[(env_name, v[0])][1], default=str)
                    != json.dumps(sims[(env_name, v[1])][1], default=str)]
            check(f"{env_name}: same param, different history "
                  f"({len(diff)}/{len(dups)} duplicated params differ)",
                  bool(diff))

    # ---------------------------------------------------------------- 3
    print("\n3. P1 class census: sessions contributing to each gt class")
    for env_name in ENVS:
        sess_true, sess_false = 0, 0
        for s in seeds:
            oracle = sims[(env_name, s)][2]
            kept = [p > 0.5 for p in oracle if abs(p - 0.5) > AMBIG_EPS]
            sess_true += any(kept)
            sess_false += any(not k for k in kept)
        ok = sess_true >= 3 and sess_false >= 3
        check(f"{env_name}: >=3 sessions per class",
              ok, f"act1-class in {sess_true}/{N} sessions, act0-class in {sess_false}/{N}")

    # ---------------------------------------------------------------- 4
    print("\n4. oracle exact values at t=1 (uninformed prior)")
    m = sims[("monty", 0)][2][0]
    check("monty t=1 == 5/9", abs(m - 5 / 9) < 1e-9, f"{m:.6f}")
    tg = sims[("trust_game", 0)][2][0]
    check("trust_game t=1 == sigmoid(-2.5) ~ 0.0759",
          abs(tg - float(sigmoid(-2.5))) < 1e-6, f"{tg:.6f}")
    mc = sims[("machine", 0)][2][0]
    check("machine t=1 == sigmoid(-2.5) ~ 0.0759 (twin of trust_game)",
          abs(mc - float(sigmoid(-2.5))) < 1e-6, f"{mc:.6f}")
    ck = sims[("chicken", 0)][2][0]
    check("chicken t=1 == 0.5 exactly (prior at indifference)",
          abs(ck - 0.5) < 1e-9, f"{ck:.6f}")
    b = sims[("bandit", 0)][2][0]
    check("bandit t=1 ~ 0.5 (MC)", abs(b - 0.5) < 0.02, f"{b:.4f}")

    # ---------------------------------------------------------------- 5
    print("\n5. oracle sanity across all sessions: finite, bounded, informative")
    for env_name in ENVS:
        allp = np.array([p for s in seeds for p in sims[(env_name, s)][2]])
        check(f"{env_name}: all oracle in [0,1], no NaN",
              bool(np.isfinite(allp).all() and (allp >= 0).all() and (allp <= 1).all()))
        ranges = [max(sims[(env_name, s)][2]) - min(sims[(env_name, s)][2]) for s in seeds]
        n_inf = sum(r >= 0.25 for r in ranges)
        # flat sessions are legitimate (e.g. trust_game EXPLOITER: prior already
        # correct, evidence only confirms) but drift claims need moving oracles:
        # require enough informative sessions for the oracle_range-gated analyses
        check(f"{env_name}: >=3 informative sessions (oracle_range >= 0.25)",
              n_inf >= 3, f"{n_inf}/{N} informative ({n_inf / N:.0%})")

    # ---------------------------------------------------------------- 6
    print("\n6. env-specific dynamics signatures")
    ipd_tft = [s for s in seeds if "TIT_FOR_TAT" in sims[("ipd", s)][0].hidden_param()]
    if ipd_tft:
        # horizon signature: cooperation was optimal at some point (oracle > 0.9)
        # but the endgame kills the continuation bonus -> final round favors DEFECT
        drops = [max(sims[("ipd", s)][2]) > 0.9 and sims[("ipd", s)][2][-1] < 0.5
                 for s in ipd_tft]
        check("ipd: TFT endgame reversal (peak > 0.9, final round < 0.5)",
              all(drops), f"{sum(drops)}/{len(drops)} TFT sessions")
    # trust_game v2: the noisy mid-types must separate -- GENEROUS sessions
    # should mostly end on the INVEST side (oracle > 0.5), STINGY on KEEP.
    # "mostly", not "all": an unlucky GENEROUS run legitimately ends KEEP.
    for ty, side_invest in (("GENEROUS", True), ("STINGY", False)):
        ss = [s for s in seeds
              if sims[("trust_game", s)][0].hidden_param() == f"trustee={ty}"]
        if ss:
            good = sum((sims[("trust_game", s)][2][-1] > 0.5) == side_invest
                       for s in ss)
            check(f"trust_game: most {ty} sessions end oracle on the "
                  f"{'INVEST' if side_invest else 'KEEP'} side",
                  good * 2 > len(ss), f"{good}/{len(ss)}")
    # machine = paired twin of trust_game: same seed must yield the
    # index-aligned hidden type, the identical 15/0 outcome sequence, and the
    # identical oracle trajectory (only the narrative framing differs)
    from envs.trust_game import TrustGameEnv
    from envs.machine import MachineEnv
    pair_ok = True
    for s in seeds:
        te, trec, tor, _, _ = sims[("trust_game", s)]
        me, mrec, mor, _, _ = sims[("machine", s)]
        if (TrustGameEnv.TYPES.index(te.trustee_type)
                != MachineEnv.TYPES.index(me.trustee_type)
                or [r["returned"] for r in trec] != [r["paid"] for r in mrec]
                or tor != mor):
            pair_ok = False
            break
    check(f"machine: paired-seed twin of trust_game over all {N} seeds "
          "(type index, outcomes, oracle identical)", pair_ok)
    # chicken: yield-leaning types must end on the DARE-optimal side and
    # dare-leaning types on the YIELD-optimal side (mostly; unlucky runs ok)
    for ty, side_dare in (("CAUTIOUS", True), ("WARY", True),
                          ("BOLD", False), ("RECKLESS", False)):
        ss = [s for s in seeds
              if sims[("chicken", s)][0].hidden_param() == f"opponent={ty}"]
        if ss:
            good = sum((sims[("chicken", s)][2][-1] > 0.5) == side_dare
                       for s in ss)
            check(f"chicken: most {ty} sessions end oracle on the "
                  f"{'DARE' if side_dare else 'YIELD'} side",
                  good * 2 > len(ss), f"{good}/{len(ss)}")
    stag_tft = [s for s in seeds if "TIT_FOR_TAT" in sims[("stag_hunt", s)][0].hidden_param()]
    if stag_tft:
        oscs = []
        for s in stag_tft:
            sgn = np.sign(np.array(sims[("stag_hunt", s)][2]) - 0.5)
            oscs.append(int(np.sum(sgn[1:] != sgn[:-1])))
        check("stag_hunt: TFT oracle oscillates (>=2 sign flips in some session)",
              max(oscs) >= 2, f"flips per TFT session: {oscs}")

    # ---------------------------------------------------------------- 7
    print("\n7. prompt templates: render cleanly for both kinds x both mappings")
    for env_name in ENVS:
        env, records, _, ctxs, histories = sims[(env_name, 0)]
        for kind in ("forced", "cot"):
            tmpl = (HERE / "prompts" / f"{env_name}_{kind}.txt").read_text()
            for t_idx in (0, T - 1):          # t=1 (empty history) and t=T
                for mapping in (0, 1):
                    a, b = ((env.act0_desc, env.act1_desc) if mapping == 0
                            else (env.act1_desc, env.act0_desc))
                    try:
                        out = tmpl.format(t=t_idx + 1, T=T, history=histories[t_idx],
                                          option_A_desc=a, option_B_desc=b,
                                          answer_format=ANSWER_FORMAT_LETTER,
                                          **ctxs[t_idx])
                    except KeyError as e:
                        check(f"{env_name}/{kind} t={t_idx+1} m{mapping}: renders",
                              False, f"missing field {e}")
                        continue
                    # NOTE: '{' legitimately appears in the rendered history
                    # (Python dict-list format), so only check the option descs;
                    # missing fields are already caught by the KeyError above
                    ok = (a in out) and (b in out)
                    check(f"{env_name}/{kind} t={t_idx+1} m{mapping}: complete", ok,
                          "" if ok else "missing option description in render")
        forced = (HERE / "prompts" / f"{env_name}_forced.txt").read_text()
        check(f"{env_name}: 'Answer:' NOT in forced template (prefill-injected)",
              "Answer:" not in forced)
        cot = (HERE / "prompts" / f"{env_name}_cot.txt").read_text()
        check(f"{env_name}: answer-format placeholder present in cot template",
              "{answer_format}" in cot)

    # ---------------------------------------------------------------- 8
    print("\n8. history integrity: grows by 1/round, tally present at t>1")
    for env_name in ENVS:
        env, records, _, _, histories = sims[(env_name, 0)]
        check(f"{env_name}: {T} history records after {T} rounds", len(records) == T)
        check(f"{env_name}: tally line in mid-session history",
              "Tally" in histories[T // 2])

    # ---------------------------------------------------------------- 9
    print("\n9. metrics: summarize_session on synthetic perfect/broken sessions")
    def synth(pre_fn, decision_fn):
        return [{"t": t, "oracle_p_act1": 0.5 + 0.04 * t,
                 "delta_pre_sem": pre_fn(t), "delta_post_sem": None,
                 "delta_amp": None, "post_valid": False,
                 "decision_act1": decision_fn(t)} for t in range(1, 11)]
    s_perfect = summarize_session(synth(lambda t: 0.3 * t, lambda t: True))
    check("perfect tracker -> drift_corr_pre ~ 1.0",
          np.isclose(s_perfect["drift_corr_pre"], 1.0), str(s_perfect["drift_corr_pre"]))
    check("perfect decisions -> Delta_T == 0.0",
          s_perfect["Delta_T"] == 0.0, str(s_perfect["Delta_T"]))
    s_anti = summarize_session(synth(lambda t: -0.3 * t, lambda t: False))
    check("anti-tracker -> drift_corr_pre ~ -1.0",
          np.isclose(s_anti["drift_corr_pre"], -1.0), str(s_anti["drift_corr_pre"]))
    check("all-wrong decisions -> Delta_T == 1.0",
          s_anti["Delta_T"] == 1.0, str(s_anti["Delta_T"]))
    s_flat = summarize_session(synth(lambda t: 0.0, lambda t: None))
    check("flat preference -> drift_corr_pre is None (constant series)",
          s_flat["drift_corr_pre"] is None)
    check("spearman guard: len<3 -> None", spearman([1, 2], [1, 2]) is None)

    # ---------------------------------------------------------------- summary
    print(f"\n{'=' * 60}")
    if FAILURES:
        print(f"SMOKE TEST FAILED: {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST PASSED: all checks green. Safe to launch the sweep.")


if __name__ == "__main__":
    main()
