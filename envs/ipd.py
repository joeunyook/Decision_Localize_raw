"""Iterated Prisoner's Dilemma with opponent-type uncertainty.

Payoffs (row = you): T=5, R=3, P=1, S=0
                 opp C      opp D
    you C         3           0
    you D         5           1

Hidden parameter: opponent type, uniform over
    ALWAYS_COOP : cooperates w.p. 1-eps
    ALWAYS_DEF  : defects w.p. 1-eps
    TIT_FOR_TAT : copies YOUR previous move (C on round 1), w.p. 1-eps
eps = 0.10.

Oracle (documented simplification -- horizon-aware, not full POMDP planning):
    p_c            = posterior-predictive P(opp cooperates this round)
    EU(C) = 3*p_c + 0*(1-p_c)
    EU(D) = 5*p_c + 1*(1-p_c)
    Myopically D dominates. Cooperation only pays through TFT reciprocity:
    continuation bonus for C = (T - t) * (R - P) * P(TFT | h) = 2*(T-t)*P(TFT)
    (each remaining round yields R=3 under mutual cooperation vs P=1 under
    mutual defection when facing TFT).
    P(C optimal) = smooth sigmoid of [EU(C) + bonus - EU(D)].
"""

import numpy as np

from .base import SequentialEnv

TYPES = ("ALWAYS_COOP", "ALWAYS_DEF", "TIT_FOR_TAT")
EPS = 0.10


class IPDEnv(SequentialEnv):
    name = "ipd"
    act0, act1 = "DEFECT", "COOPERATE"
    act0_desc = "DEFECT (5 points if they cooperate, 1 if they defect)"
    act1_desc = "COOPERATE (3 points if they cooperate, 0 if they defect)"

    def __init__(self, rng, T, opp_type=None):
        super().__init__(rng, T)
        self.opp_type = opp_type if opp_type is not None else str(rng.choice(TYPES))
        self._records = []

    def hidden_param(self):
        return f"opponent={self.opp_type}"

    def _p_opp_coop(self, opp_type, my_prev):
        if opp_type == "ALWAYS_COOP":
            return 1 - EPS
        if opp_type == "ALWAYS_DEF":
            return EPS
        if my_prev is None or my_prev == "COOPERATE":
            return 1 - EPS
        return EPS

    def new_round(self, t):
        return {"rounds_left": self.T - t}

    def behavior_action(self, t):
        return int(self.rng.integers(0, 2))

    def outcome(self, t, ctx, action_idx):
        my = self.act1 if action_idx == 1 else self.act0
        my_prev = self._records[-1]["my"] if self._records else None
        p_c = self._p_opp_coop(self.opp_type, my_prev)
        opp = "COOPERATE" if self.rng.random() < p_c else "DEFECT"
        payoff = {("COOPERATE", "COOPERATE"): 3, ("COOPERATE", "DEFECT"): 0,
                  ("DEFECT", "COOPERATE"): 5, ("DEFECT", "DEFECT"): 1}[(my, opp)]
        record = {"t": t, "action": my, "my": my, "opp": opp,
                  "win": payoff >= 3, "payoff": payoff,
                  "result": f"opp played {opp}, you scored {payoff}"}
        self._records.append(record)
        return float(payoff), record

    def format_history(self, records):
        if not records:
            return "[]  # no completed rounds yet"
        lines = ["["]
        for r in records:
            lines.append(f'    {{"you": "{r["my"]}", "opponent": "{r["opp"]}", "your_payoff": {r["payoff"]}}},')
        lines.append("]")
        opp_c = sum(1 for r in records if r["opp"] == "COOPERATE")
        lines.append(f"# Tally  ->  opponent COOPERATED {opp_c}/{len(records)} rounds")
        return "\n".join(lines)

    def posterior(self, records):
        logp = np.zeros(len(TYPES))
        my_prev = None
        for r in records:
            for i, ty in enumerate(TYPES):
                p = self._p_opp_coop(ty, my_prev)
                lik = p if r["opp"] == "COOPERATE" else 1 - p
                logp[i] += np.log(max(lik, 1e-12))
            my_prev = r["my"]
        w = np.exp(logp - logp.max())
        return w / w.sum()

    def oracle_p_act1(self, records):
        post = self.posterior(records)
        my_prev = records[-1]["my"] if records else None
        p_c = sum(post[i] * self._p_opp_coop(ty, my_prev)
                  for i, ty in enumerate(TYPES))
        t = len(records) + 1
        p_tft = post[TYPES.index("TIT_FOR_TAT")]
        eu_c = 3 * p_c
        eu_d = 5 * p_c + 1 * (1 - p_c)
        bonus = 2 * max(self.T - t, 0) * p_tft
        return float(1.0 / (1.0 + np.exp(-1.5 * (eu_c + bonus - eu_d))))
