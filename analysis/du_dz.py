"""
DeltaU vs Deltaz : does the pre-CoT action logit margin track the baseline's
expected-utility margin?

DeltaU_t = E[U(a1,theta)|h_t] - E[U(a0,theta)|h_t]   in NATIVE PAYOFF UNITS
Deltaz_t = z(a1|h_t) - z(a0|h_t)                     = delta_pre_sem (nats)

Every scenario's oracle_p_act1 is a known, invertible function of DeltaU, so
DeltaU is recovered exactly per round (bandit is recomputed from the Beta
posterior, which is where its utility margin actually lives).

Reported per (model, scenario):
  rho          Spearman rank correlation (monotone tracking, scale free)
  slope_std    OLS slope in nats per SD of DeltaU (cross-scenario comparable)
  slope_raw    OLS slope in nats per payoff unit
  sign_agree   fraction of UNAMBIGUOUS rounds with sign(Deltaz) == sign(DeltaU)
CIs are bootstrapped over SEEDS (sessions), not rounds: rounds within a session
share a hidden parameter and a history prefix.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = "/users/joeunyook/logit-probe"
RESP = os.path.join(ROOT, "responses")
RES = os.path.join(ROOT, "results")
OUT = os.path.dirname(os.path.abspath(__file__))

AMBIG_EPS = 0.10
N_BOOT = 2000
RNG = np.random.default_rng(20260726)

MODELS = ["gemma2b", "llama", "phi", "qwen", "dsqwen7",
          "qwen27", "gemma31", "llama70", "dsllama70"]

# scenario -> (family, DeltaU range in native payoff units)
FAMILY = {
    "bandit":     ("Bayes-pure", (-1.0, 1.0)),
    "monty":      ("Bayes-pure", (-1.0, 1.0)),
    "trust_game": ("Bayes-EU",   (-10.0, 5.0)),
    "machine":    ("Bayes-EU",   (-10.0, 5.0)),
    "stag_hunt":  ("Game",       (-2.0, 1.0)),
    "chicken":    ("Game",       (-1.0, 1.0)),
    "bos":        ("Game",       (-1.0, 2.0)),
    "pennies":    ("Game",       (-1.0, 1.0)),
    "ipd":        ("IPD",        (-9.0, 9.0)),   # reported separately
}
ORDER = ["bandit", "monty", "trust_game", "machine",
         "stag_hunt", "chicken", "bos", "pennies", "ipd"]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def delta_u(env, oracle, records=None):
    """Invert each scenario's oracle to the expected-utility margin."""
    lo, hi = FAMILY[env][1]
    if env == "monty":
        # P(STAY wins) = 1 - P(SWITCH wins) exactly (win probs sum to 1 per rule)
        du = 2.0 * np.asarray(oracle, dtype=float) - 1.0
    elif env == "bandit":
        du = np.asarray(records, dtype=float)          # E[p_R] - E[p_L]
    elif env in ("trust_game", "machine"):
        du = logit(oracle) / 1.0                        # sigmoid(EU - 10)
    elif env == "stag_hunt":
        du = 3.0 * (logit(oracle) / 20.0)               # 3*(p - 2/3)
    elif env == "bos":
        du = 3.0 * (logit(oracle) / 20.0)               # 3*(p - 1/3)
    elif env in ("chicken", "pennies"):
        du = logit(oracle) / 5.0                        # 2p - 1
    elif env == "ipd":
        du = logit(oracle) / 1.5
    else:
        raise KeyError(env)
    return np.clip(du, lo, hi)


def bandit_margin(history_records, T):
    """E[p_RIGHT|h_t] - E[p_LEFT|h_t] under Beta(1,1) priors, per round t."""
    out, aL, bL, aR, bR = [], 1, 1, 1, 1
    for t in range(T):
        out.append(aR / (aR + bR) - aL / (aL + bL))     # BEFORE round t's outcome
        if t < len(history_records):
            r = history_records[t]
            if r["action"] == "LEFT":
                aL, bL = (aL + 1, bL) if r["win"] else (aL, bL + 1)
            else:
                aR, bR = (aR + 1, bR) if r["win"] else (aR, bR + 1)
    return out


def rankdata(a):
    """Average ranks, fully vectorised (ties matter: DeltaU saturates)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return ranks


def ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    vx = x.var()
    if vx == 0 or len(x) < 3:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / vx)


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def load():
    """rows[(model, env)] = list of (seed, du_array, dz_array, oracle_array)"""
    rows = defaultdict(list)
    files = sorted(os.listdir(RESP))
    for fn in files:
        if not fn.endswith(".json"):
            continue
        stem = fn[:-5]
        parts = stem.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        head, seed = parts
        model = next((m for m in MODELS if head.startswith(m + "_")), None)
        if model is None:
            continue
        env = head[len(model) + 1:]
        if env not in FAMILY:
            continue
        try:
            d = json.load(open(os.path.join(RESP, fn)))
        except Exception:
            continue
        rounds = d.get("rounds") or []
        if not rounds:
            continue
        dz = np.array([r.get("delta_pre_sem", np.nan) for r in rounds], float)
        orc = np.array([r.get("oracle_p_act1", np.nan) for r in rounds], float)
        if env == "bandit":
            rp = os.path.join(RES, fn)
            if not os.path.exists(rp):
                continue
            rd = json.load(open(rp))
            marg = bandit_margin(rd.get("history_records", []), len(rounds))
            du = delta_u(env, orc, records=marg)
        else:
            du = delta_u(env, orc)
        ok = np.isfinite(dz) & np.isfinite(du) & np.isfinite(orc)
        if ok.sum() == 0:
            continue
        rows[(model, env)].append((int(seed), du[ok], dz[ok], orc[ok]))
    return rows


def summarize(sessions):
    du = np.concatenate([s[1] for s in sessions])
    dz = np.concatenate([s[2] for s in sessions])
    orc = np.concatenate([s[3] for s in sessions])

    rho = spearman(du, dz)
    sd = np.std(du)
    slope_raw = ols_slope(du, dz) if sd > 0 else float("nan")
    slope_std = slope_raw * sd

    unamb = np.abs(orc - 0.5) > AMBIG_EPS
    if unamb.sum() > 0:
        sign_agree = float(np.mean(np.sign(dz[unamb]) == np.sign(du[unamb])))
    else:
        sign_agree = float("nan")

    # bootstrap over sessions
    boots_r, boots_s = [], []
    n = len(sessions)
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        bdu = np.concatenate([sessions[i][1] for i in idx])
        bdz = np.concatenate([sessions[i][2] for i in idx])
        boots_r.append(spearman(bdu, bdz))
        s = np.std(bdu)
        boots_s.append(ols_slope(bdu, bdz) * s if s > 0 else np.nan)
    boots_r = np.array(boots_r, float); boots_s = np.array(boots_s, float)
    br = boots_r[np.isfinite(boots_r)]; bs = boots_s[np.isfinite(boots_s)]

    return dict(
        n_rounds=int(len(du)), n_sessions=n,
        rho=rho,
        rho_lo=float(np.percentile(br, 2.5)) if len(br) else float("nan"),
        rho_hi=float(np.percentile(br, 97.5)) if len(br) else float("nan"),
        slope_std=slope_std,
        slope_std_lo=float(np.percentile(bs, 2.5)) if len(bs) else float("nan"),
        slope_std_hi=float(np.percentile(bs, 97.5)) if len(bs) else float("nan"),
        slope_raw=slope_raw,
        sign_agree=sign_agree,
        du_sd=float(sd),
    )


def main():
    rows = load()
    out = {}
    for (model, env), sess in sorted(rows.items()):
        out[f"{model}|{env}"] = summarize(sess)
        out[f"{model}|{env}"]["model"] = model
        out[f"{model}|{env}"]["env"] = env
        out[f"{model}|{env}"]["family"] = FAMILY[env][0]

    with open(os.path.join(OUT, "du_dz.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ---- per model x scenario ----
    print("\nSpearman rho( DeltaU , Deltaz )   [95% CI bootstrapped over seeds]\n")
    hdr = "scenario   " + "".join(f"{m[:9]:>10}" for m in MODELS)
    print(hdr); print("-" * len(hdr))
    for env in ORDER:
        line = f"{env:<11}"
        for m in MODELS:
            k = f"{m}|{env}"
            line += f"{out[k]['rho']:>10.2f}" if k in out else f"{'--':>10}"
        print(line)

    print("\nOLS slope, nats per SD of DeltaU\n")
    print(hdr); print("-" * len(hdr))
    for env in ORDER:
        line = f"{env:<11}"
        for m in MODELS:
            k = f"{m}|{env}"
            line += f"{out[k]['slope_std']:>10.2f}" if k in out else f"{'--':>10}"
        print(line)

    print("\nSign agreement, unambiguous rounds\n")
    print(hdr); print("-" * len(hdr))
    for env in ORDER:
        line = f"{env:<11}"
        for m in MODELS:
            k = f"{m}|{env}"
            line += f"{out[k]['sign_agree']*100:>10.0f}" if k in out else f"{'--':>10}"
        print(line)

    # ---- pooled by family (per model, then across) ----
    print("\n\nPOOLED BY FAMILY (per model)\n")
    fams = ["Bayes-pure", "Bayes-EU", "Game"]
    print(f"{'model':<11}" + "".join(f"{f:>26}" for f in fams))
    print(f"{'':<11}" + "".join(f"{'rho   slope  sign':>26}" for _ in fams))
    print("-" * (11 + 26 * 3))
    for m in MODELS:
        line = f"{m:<11}"
        for fam in fams:
            ks = [k for k in out if out[k]["model"] == m and out[k]["family"] == fam]
            if not ks:
                line += f"{'--':>26}"; continue
            r = np.mean([out[k]["rho"] for k in ks])
            s = np.mean([out[k]["slope_std"] for k in ks])
            g = np.mean([out[k]["sign_agree"] for k in ks]) * 100
            line += f"{r:>10.2f}{s:>8.2f}{g:>8.0f}"
        print(line)

    print("\nFAMILY MEANS (over 9 models)\n")
    print(f"{'family':<12}{'rho':>8}{'slope_std':>12}{'sign%':>8}{'n_combo':>9}")
    print("-" * 49)
    for fam in fams + ["IPD"]:
        ks = [k for k in out if out[k]["family"] == fam]
        if not ks:
            continue
        print(f"{fam:<12}"
              f"{np.mean([out[k]['rho'] for k in ks]):>8.2f}"
              f"{np.mean([out[k]['slope_std'] for k in ks]):>12.2f}"
              f"{np.mean([out[k]['sign_agree'] for k in ks])*100:>8.0f}"
              f"{len(ks):>9}")
    print("\nwrote du_dz.json")


if __name__ == "__main__":
    main()
