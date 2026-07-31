"""Cross-verify bos against stag_hunt: label balance, clock-dependence, prompts."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "/users/joeunyook/logit-probe")
from envs import make_env

HERE = Path("/users/joeunyook/logit-probe")
T, NSEED = 10, 100


def sweep(env_name):
    lab = np.zeros((NSEED, T), dtype=bool)
    amb = np.zeros((NSEED, T), dtype=bool)
    types = []
    for s in range(NSEED):
        rng = np.random.default_rng(s)
        env = make_env(env_name, rng, T, seed=s)
        types.append(env.opp_type)
        records = []
        for t in range(1, T + 1):
            ctx = env.new_round(t)
            p = env.oracle_p_act1(records)
            lab[s, t - 1] = p > 0.5
            amb[s, t - 1] = abs(p - 0.5) <= 0.10
            beh = env.behavior_action(t)
            _, rec = env.outcome(t, ctx, beh)
            records.append(rec)
    return lab, amb, types


for name in ("stag_hunt", "bos", "ipd"):
    lab, amb, types = sweep(name)
    maj = lab.mean(axis=0) > 0.5
    clock = np.tile(maj, (NSEED, 1))
    census = {ty: types.count(ty) for ty in sorted(set(types))}
    print(f"\n=== {name} ===")
    print(f"  type census        : {census}")
    print(f"  % act1 by t        : " + " ".join(f"{100*lab[:, t].mean():3.0f}" for t in range(T)))
    print(f"  % act1 overall     : {100*lab.mean():.1f}%")
    print(f"  degenerate rounds  : {sum(1 for t in range(T) if lab[:, t].mean() in (0.0, 1.0))}/10")
    print(f"  clock-only accuracy: {100*(clock == lab).mean():.1f}%   <-- lower is better")
    print(f"  ambiguous (<=0.10) : {100*amb.mean():.1f}% of rounds dropped")

# ---- prompt render check (all 4 templates, both mappings) ----
print("\n=== bos prompt render check ===")
rng = np.random.default_rng(7)
env = make_env("bos", rng, T, seed=7)
env.opp_type = "TIT_FOR_TAT"
records = []
for t in range(1, 7):
    ctx = env.new_round(t)
    if t == 6:
        hist = env.format_history(records)
        for kind in ("forced", "cot", "forced_dry", "cot_dry"):
            tmpl = (HERE / "prompts" / f"bos_{kind}.txt").read_text()
            for mapping in (0, 1):
                a, b = ((env.act0_desc, env.act1_desc) if mapping == 0
                        else (env.act1_desc, env.act0_desc))
                try:
                    out = tmpl.format(t=t, T=T, history=hist, option_A_desc=a,
                                      option_B_desc=b, answer_format="<AF>", **ctx)
                except KeyError as e:
                    print(f"  FAIL {kind} mapping{mapping}: missing field {e}")
                    break
            else:
                print(f"  OK   bos_{kind}.txt   ({len(out)} chars)")
                continue
            break
        print("\n----- rendered bos_forced.txt @ t=6, mapping 0 -----")
        tmpl = (HERE / "prompts" / "bos_forced.txt").read_text()
        print(tmpl.format(t=6, T=T, history=hist,
                          option_A_desc=env.act0_desc, option_B_desc=env.act1_desc))
    beh = env.behavior_action(t)
    _, rec = env.outcome(t, ctx, beh)
    records.append(rec)
