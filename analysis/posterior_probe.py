"""
Does the residual stream carry the POSTERIOR OVER HIDDEN TYPES p(theta|h_t),
or only its decision-relevant projection DeltaU?

For each (model, scenario) we regress the final-layer residual onto the exact
posterior p(theta|h_t), replayed from the saved history_records.

Three numbers per combo, all grouped-CV by seed (no session leakage):
  R2_full     predicting p(theta|h_t) itself.  Uninformative on its own:
              DeltaU is a function of the posterior, so a probe that only
              tracked DeltaU would already score here.
  R2_partial  THE TEST. Predicting p(theta|h_t) after nonparametrically
              removing E[p(theta|h_t) | DeltaU]. Anything above zero is
              posterior information the decision margin does not contain.
  R2_shuffled Same pipeline, targets permuted across rounds. Fixes the zero.

bandit is excluded by construction: its hidden parameter is binary, so its
posterior is one-dimensional and monotone in DeltaU -- there is no residual
information for the partial test to find.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/users/joeunyook/logit-probe")
from envs import make_env  # noqa: E402

ROOT = "/users/joeunyook/logit-probe"
RESID = os.path.join(ROOT, "residuals")
RES = os.path.join(ROOT, "results")
RESP = os.path.join(ROOT, "responses")
OUT = os.path.dirname(os.path.abspath(__file__))

MODELS = ["gemma2b", "llama", "phi", "qwen", "dsqwen7",
          "qwen27", "gemma31", "llama70", "dsllama70"]
ENVS = ["monty", "trust_game", "machine", "stag_hunt", "chicken", "bos", "pennies"]
SEEDS = range(100)
T = 10
N_FOLDS = 10          # grouped by seed; 10 folds, not 100, for runtime
N_BINS = 20           # quantile bins for the nonparametric DeltaU partial-out
ALPHA = 1.0           # ridge, scaled to trace(K)/n so it is unit-free
RNG = np.random.default_rng(20260726)

DU = {  # invert oracle_p_act1 -> expected-utility margin (native payoff units)
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


def env_posteriors(env, history_records, n_rounds):
    """p(theta | h_t) for t = 1..n_rounds, using only records BEFORE round t."""
    return np.array([env.posterior(history_records[:t]) for t in range(n_rounds)],
                    dtype=float)


def load_combo(model, env_name):
    env = make_env(env_name, np.random.default_rng(0), T, seed=0)
    X, Y, DUv, G = [], [], [], []
    for seed in SEEDS:
        stem = f"{model}_{env_name}_seed{seed}"
        fr = os.path.join(RESID, stem + ".npy")
        fj = os.path.join(RES, stem + ".json")
        fp = os.path.join(RESP, stem + ".json")
        if not (os.path.exists(fr) and os.path.exists(fj) and os.path.exists(fp)):
            continue
        try:
            arr = np.load(fr, mmap_mode="r")
            h = np.asarray(arr[:, -1, :], dtype=np.float32)     # final layer
            hist = json.load(open(fj))["history_records"]
            rounds = json.load(open(fp))["rounds"]
        except Exception:
            continue
        n = min(len(h), len(rounds), T)
        if n == 0:
            continue
        post = env_posteriors(env, hist, n)
        orc = np.array([rounds[i].get("oracle_p_act1", np.nan) for i in range(n)], float)
        du = DU[env_name](orc)
        ok = np.isfinite(du) & np.isfinite(h[:n]).all(axis=1) & np.isfinite(post).all(axis=1)
        if ok.sum() == 0:
            continue
        X.append(h[:n][ok]); Y.append(post[ok]); DUv.append(du[ok])
        G.append(np.full(int(ok.sum()), seed))
    if not X:
        return None
    return (np.concatenate(X), np.concatenate(Y),
            np.concatenate(DUv), np.concatenate(G))


def partial_out(y_tr, du_tr, y_te, du_te, n_bins=N_BINS):
    """Remove E[y | DeltaU] estimated on TRAIN only (bin means, no leakage)."""
    edges = np.quantile(du_tr, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return y_tr - y_tr.mean(0), y_te - y_tr.mean(0)
    b_tr = np.clip(np.digitize(du_tr, edges[1:-1]), 0, len(edges) - 2)
    b_te = np.clip(np.digitize(du_te, edges[1:-1]), 0, len(edges) - 2)
    gm = y_tr.mean(0)
    means = np.tile(gm, (len(edges) - 1, 1))
    for b in range(len(edges) - 1):
        m = b_tr == b
        if m.sum() >= 5:
            means[b] = y_tr[m].mean(0)
    return y_tr - means[b_tr], y_te - means[b_te]


def ridge_cv(X, Y, groups, du=None, shuffle=False):
    """Grouped-by-seed CV; dual-form ridge (n << d). Returns pooled R^2."""
    if shuffle:
        Y = Y[RNG.permutation(len(Y))]
    uniq = np.unique(groups)
    folds = np.array_split(RNG.permutation(uniq), N_FOLDS)
    num = np.zeros(Y.shape[1]); den = np.zeros(Y.shape[1])
    for f in folds:
        te = np.isin(groups, f); tr = ~te
        if tr.sum() < 20 or te.sum() < 2:
            continue
        Xtr, Xte = X[tr], X[te]
        mu = Xtr.mean(0)
        Xtr = Xtr - mu; Xte = Xte - mu
        sc = np.sqrt((Xtr ** 2).sum(1).mean()) or 1.0
        Xtr = Xtr / sc; Xte = Xte / sc

        ytr, yte = Y[tr], Y[te]
        if du is not None:
            ytr, yte = partial_out(ytr, du[tr], yte, du[te])
        ym = ytr.mean(0)
        ytr_c = ytr - ym

        K = Xtr @ Xtr.T
        lam = ALPHA * np.trace(K) / len(K)
        A = np.linalg.solve(K + lam * np.eye(len(K)), ytr_c)
        pred = (Xte @ Xtr.T) @ A + ym

        num += ((yte - pred) ** 2).sum(0)
        den += ((yte - ytr.mean(0)) ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - num / den
    return float(np.nanmean(r2))


def main():
    out = {}
    print(f"{'model':<11}{'scenario':<12}{'R2_full':>9}{'R2_partial':>12}"
          f"{'R2_shuf':>9}{'n':>7}{'K':>4}")
    print("-" * 64)
    for env_name in ENVS:
        for model in MODELS:
            d = load_combo(model, env_name)
            if d is None:
                continue
            X, Y, du, G = d
            r_full = ridge_cv(X, Y, G)
            r_part = ridge_cv(X, Y, G, du=du)
            r_shuf = ridge_cv(X, Y, G, du=du, shuffle=True)
            out[f"{model}|{env_name}"] = dict(
                model=model, env=env_name, r2_full=r_full,
                r2_partial=r_part, r2_shuffled=r_shuf,
                n=int(len(X)), k=int(Y.shape[1]))
            print(f"{model:<11}{env_name:<12}{r_full:>9.3f}{r_part:>12.3f}"
                  f"{r_shuf:>9.3f}{len(X):>7}{Y.shape[1]:>4}")
        print()

    with open(os.path.join(OUT, "posterior_probe.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("\nSCENARIO MEANS")
    print(f"{'scenario':<12}{'R2_full':>9}{'R2_partial':>12}{'R2_shuf':>9}{'combos':>8}")
    print("-" * 50)
    for env_name in ENVS:
        ks = [k for k in out if out[k]["env"] == env_name]
        if not ks:
            continue
        print(f"{env_name:<12}"
              f"{np.mean([out[k]['r2_full'] for k in ks]):>9.3f}"
              f"{np.mean([out[k]['r2_partial'] for k in ks]):>12.3f}"
              f"{np.mean([out[k]['r2_shuffled'] for k in ks]):>9.3f}"
              f"{len(ks):>8}")
    allk = list(out)
    if allk:
        print(f"\n{'OVERALL':<12}"
              f"{np.mean([out[k]['r2_full'] for k in allk]):>9.3f}"
              f"{np.mean([out[k]['r2_partial'] for k in allk]):>12.3f}"
              f"{np.mean([out[k]['r2_shuffled'] for k in allk]):>9.3f}"
              f"{len(allk):>8}")
    print("\nwrote posterior_probe.json")


if __name__ == "__main__":
    main()
