"""
CONTROL: how much of the posterior is recoverable from the PRINTED TALLY alone?

The prompt shows a computed outcome tally, and each scenario's log-posterior is
linear in those counts (the posterior is a softmax of a linear function of them).
So "the posterior is linearly decodable from the residual stream" is only
interesting if a linear map from the counts themselves does NOT already recover
it -- i.e. if the network is supplying the nonlinear normalization.

Same targets, same grouped CV, same DeltaU partial-out as posterior_probe.py.
Two count baselines:
  linear     the raw sufficient counts             (like-for-like with the probe)
  quadratic  counts + all pairwise products        (generous upper bound for the
                                                    skeptic's position)
Histories depend only on (env, seed), not on the model, so this is computed once
per scenario.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/users/joeunyook/logit-probe")
from envs import make_env  # noqa: E402

ROOT = "/users/joeunyook/logit-probe"
RES = os.path.join(ROOT, "results")
RESP = os.path.join(ROOT, "responses")
OUT = os.path.dirname(os.path.abspath(__file__))

MODELS = ["gemma2b", "llama", "phi", "qwen", "dsqwen7",
          "qwen27", "gemma31", "llama70", "dsllama70"]
ENVS = ["monty", "trust_game", "machine", "stag_hunt", "chicken", "bos", "pennies"]
SEEDS = range(100)
T = 10
N_FOLDS = 10
N_BINS = 20
RNG = np.random.default_rng(20260726)

DU = {
    "monty":      lambda o: 2.0 * o - 1.0,
    "trust_game": lambda o: _logit(o) / 1.0,
    "machine":    lambda o: _logit(o) / 1.0,
    "stag_hunt":  lambda o: 3.0 * _logit(o) / 20.0,
    "bos":        lambda o: 3.0 * _logit(o) / 20.0,
    "chicken":    lambda o: _logit(o) / 5.0,
    "pennies":    lambda o: _logit(o) / 5.0,
}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def count_cells(env_name, records, act1, act0, outcome_key=None):
    """Sufficient counts the likelihood actually reads, as a fixed-length vector."""
    if env_name == "monty":
        cells = {(a, w): 0 for a in ("STAY", "SWITCH") for w in (True, False)}
        for r in records:
            cells[(r["action"], bool(r["win"]))] += 1
    elif env_name in ("trust_game", "machine"):
        cells = {True: 0, False: 0}
        for r in records:
            if r["action"] != act1:
                continue
            cells[r[outcome_key] == 15] += 1
    elif env_name == "chicken":
        cells = {"YIELD": 0, "DARE": 0}
        for r in records:
            cells[r["opp"]] += 1
    else:  # stag_hunt, bos, pennies -- reactive: (my_prev, opp) cross-tab
        opts = (act1, act0)
        cells = {(p, o): 0 for p in ("NONE",) + opts for o in opts}
        my_prev = "NONE"
        for r in records:
            cells[(my_prev, r["opp"])] += 1
            my_prev = r["my"]
    return np.array([cells[k] for k in sorted(cells, key=str)], dtype=float)


def load_env(env_name):
    env = make_env(env_name, np.random.default_rng(0), T, seed=0)
    act1, act0 = env.act1, env.act0
    okey = getattr(env, "OUTCOME_KEY", None)
    C, Y, DUv, G = [], [], [], []
    for seed in SEEDS:
        stem = f"{env_name}_seed{seed}.json"
        fj = fp = None
        for m in MODELS:                      # histories are model-independent
            a, b = os.path.join(RES, m + "_" + stem), os.path.join(RESP, m + "_" + stem)
            if os.path.exists(a) and os.path.exists(b):
                fj, fp = a, b
                break
        if fj is None:
            continue
        hist = json.load(open(fj))["history_records"]
        rounds = json.load(open(fp))["rounds"]
        n = min(len(rounds), T)
        for t in range(n):
            C.append(count_cells(env_name, hist[:t], act1, act0, okey))
            Y.append(env.posterior(hist[:t]))
        orc = np.array([rounds[i].get("oracle_p_act1", np.nan) for i in range(n)], float)
        DUv.append(DU[env_name](orc))
        G.append(np.full(n, seed))
    if not C:
        return None
    return (np.array(C), np.array(Y), np.concatenate(DUv), np.concatenate(G))


def expand(C):
    """counts + all pairwise products (incl. squares)."""
    cols = [C]
    for i in range(C.shape[1]):
        for j in range(i, C.shape[1]):
            cols.append((C[:, i] * C[:, j])[:, None])
    return np.hstack(cols)


def partial_out(y_tr, du_tr, y_te, du_te):
    edges = np.unique(np.quantile(du_tr, np.linspace(0, 1, N_BINS + 1)))
    if len(edges) < 3:
        return y_tr - y_tr.mean(0), y_te - y_tr.mean(0)
    b_tr = np.clip(np.digitize(du_tr, edges[1:-1]), 0, len(edges) - 2)
    b_te = np.clip(np.digitize(du_te, edges[1:-1]), 0, len(edges) - 2)
    means = np.tile(y_tr.mean(0), (len(edges) - 1, 1))
    for b in range(len(edges) - 1):
        m = b_tr == b
        if m.sum() >= 5:
            means[b] = y_tr[m].mean(0)
    return y_tr - means[b_tr], y_te - means[b_te]


def cv_r2(X, Y, groups, du=None):
    uniq = np.unique(groups)
    folds = np.array_split(np.random.default_rng(7).permutation(uniq), N_FOLDS)
    num = np.zeros(Y.shape[1]); den = np.zeros(Y.shape[1])
    for f in folds:
        te = np.isin(groups, f); tr = ~te
        Xtr, Xte = X[tr], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Xtr = np.hstack([(Xtr - mu) / sd, np.ones((tr.sum(), 1))])
        Xte = np.hstack([(Xte - mu) / sd, np.ones((te.sum(), 1))])
        ytr, yte = Y[tr], Y[te]
        if du is not None:
            ytr, yte = partial_out(ytr, du[tr], yte, du[te])
        lam = 1e-6 * len(Xtr)
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
        pred = Xte @ W
        num += ((yte - pred) ** 2).sum(0)
        den += ((yte - ytr.mean(0)) ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(1.0 - num / den))


def main():
    probe = json.load(open(os.path.join(OUT, "posterior_probe.json")))
    rows = {}
    print(f"{'scenario':<12}{'cells':>6}"
          f"{'countsLIN':>11}{'countsQUAD':>12}{'residual':>10}"
          f"{'| cLIN_p':>10}{'cQUAD_p':>9}{'resid_p':>9}")
    print("-" * 80)
    for env_name in ENVS:
        d = load_env(env_name)
        if d is None:
            continue
        C, Y, du, G = d
        Q = expand(C)
        r_lin = cv_r2(C, Y, G)
        r_quad = cv_r2(Q, Y, G)
        r_lin_p = cv_r2(C, Y, G, du=du)
        r_quad_p = cv_r2(Q, Y, G, du=du)
        ks = [k for k in probe if probe[k]["env"] == env_name]
        r_res = float(np.mean([probe[k]["r2_full"] for k in ks]))
        r_res_p = float(np.mean([probe[k]["r2_partial"] for k in ks]))
        rows[env_name] = dict(cells=int(C.shape[1]),
                              counts_lin=r_lin, counts_quad=r_quad,
                              residual=r_res, counts_lin_partial=r_lin_p,
                              counts_quad_partial=r_quad_p, residual_partial=r_res_p)
        print(f"{env_name:<12}{C.shape[1]:>6}{r_lin:>11.3f}{r_quad:>12.3f}"
              f"{r_res:>10.3f}{r_lin_p:>10.3f}{r_quad_p:>9.3f}{r_res_p:>9.3f}")

    m = lambda k: np.mean([v[k] for v in rows.values()])
    print("-" * 80)
    print(f"{'MEAN':<12}{'':>6}{m('counts_lin'):>11.3f}{m('counts_quad'):>12.3f}"
          f"{m('residual'):>10.3f}{m('counts_lin_partial'):>10.3f}"
          f"{m('counts_quad_partial'):>9.3f}{m('residual_partial'):>9.3f}")
    json.dump(rows, open(os.path.join(OUT, "tally_control.json"), "w"), indent=1)
    print("\nwrote tally_control.json")


if __name__ == "__main__":
    main()
