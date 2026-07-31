"""IPD oracle label distribution by round, seeds 0-99, under three bonus rules.

  current  : bonus = 2*(T-t)*P(TFT|h)        (what the paper used)
  onestep  : bonus = 2*P(TFT|h)              (behavior-policy-consistent)
  none     : bonus = 0                       (pure myopic stage EU)

Replays the exact scripted histories of the sweep (same rng seeding as run_all.py).
"""
import sys, numpy as np
sys.path.insert(0, "/users/joeunyook/logit-probe")
from envs import make_env
from envs.ipd import TYPES

T, NSEED = 10, 100
IT = TYPES.index("TIT_FOR_TAT")

def label(env, records, rule):
    post = env.posterior(records)
    my_prev = records[-1]["my"] if records else None
    p_c = sum(post[i] * env._p_opp_coop(ty, my_prev) for i, ty in enumerate(TYPES))
    t = len(records) + 1
    p_tft = post[IT]
    eu_c, eu_d = 3 * p_c, 5 * p_c + 1 * (1 - p_c)
    bonus = {"current": 2 * max(T - t, 0) * p_tft,
             "onestep": 2 * p_tft,
             "none":    0.0}[rule]
    return (eu_c + bonus - eu_d) > 0        # True = COOPERATE

rules = ["current", "onestep", "none"]
coop = {r: np.zeros((NSEED, T), dtype=bool) for r in rules}
types = []

for s in range(NSEED):
    rng = np.random.default_rng(s)
    env = make_env("ipd", rng, T, seed=s)
    types.append(env.opp_type)
    records = []
    for t in range(1, T + 1):
        ctx = env.new_round(t)
        for r in rules:
            coop[r][s, t - 1] = label(env, records, r)
        beh = env.behavior_action(t)
        _, rec = env.outcome(t, ctx, beh)
        records.append(rec)

print(f"opponent types over 100 seeds: "
      f"{ {ty: types.count(ty) for ty in TYPES} }\n")
print("% of seeds labelled COOPERATE, by round")
print(f"{'t':>3} | {'current':>8} {'onestep':>8} {'none':>6}")
print("-" * 32)
for t in range(T):
    print(f"{t+1:>3} | " + " ".join(f"{100*coop[r][:, t].mean():7.0f}%" for r in rules))
print("-" * 32)
print("all | " + " ".join(f"{100*coop[r].mean():7.0f}%" for r in rules))

print("\nlabel agreement with the current rule (over all 1000 rounds):")
for r in rules[1:]:
    print(f"  {r:8s}: {100*(coop[r] == coop['current']).mean():5.1f}% identical")

# how much of the current label is explained by the clock alone?
clock = np.tile(np.arange(1, T + 1), (NSEED, 1))
maj_by_t = coop["current"].mean(axis=0) > 0.5
pred = np.tile(maj_by_t, (NSEED, 1))
print(f"\nA clock-only predictor (per-round majority, ignores all history) "
      f"reproduces the current oracle on {100*(pred == coop['current']).mean():.1f}% of rounds.")
