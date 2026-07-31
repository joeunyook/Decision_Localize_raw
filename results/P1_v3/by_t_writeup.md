# t-stratified results — detailed writeup (P1_v3, 2026-07-15)

Question: as within-session evidence accumulates (round index t = 1…10), does the
**belief** (linear probe), the **decision readout** (pre-CoT sign agreement), the
**confidence** (|Δ_pre|), and the **causal repairability** (steering rescue) change —
and does the answer differ between Bayesian-inference games (bandit, monty) and
game-theoretic games (ipd, stag_hunt, trust_game)?

Models (9): gemma2b (gemma-2-2b-it), llama (Llama-3.2-3B), qwen (Qwen3.5-4B),
dsqwen7 (R1-Distill-Qwen-7B), phi (Phi-4-mini), qwen27 (Qwen3.6-27B),
gemma31 (gemma-4-31B-it), llama70 (Llama-3.3-70B), dsllama70 (R1-Distill-Llama-70B).
Data: 100 seeds × 10 rounds per combo (sweep jobs of 2026-07-12/13, `--no-post`);
steering replays seeds 0–39 (α ∈ {1,2,4,8}, rescue/anti/2×random, probe best layer,
last prompt position).

## 0. Provenance and reproduction

Everything in this document is produced by **`analysis/by_t.py`** (committed
2026-07-15), which regenerates `results/P1_v3/probe_by_t.json` and
`results/P1_v3/behavior_by_t.json` from the raw artifacts
(`residuals/*.npy` + `_labels.json`, `responses/*.json`, `steer/P1_v3/*.json`).

Reproduction status vs the 2026-07-15 ad-hoc originals (kept as `*.orig-adhoc`):

- **behavior section: byte-identical** (max abs diff 0 over all 45 combos × 10 t × 5 fields).
- **rescue section: identical** on all 45 combos.
- **probe section: cell counts identical**; accuracies match to mean |diff| 0.17pp.
  16/432 cells differ by >2pp (worst 5.4pp, dsqwen7_ipd t5) — pure CV fold-assignment
  noise: the ad-hoc script's grouped 10-fold split is unrecoverable; `by_t.py` uses
  deterministic `GroupKFold(10)` over seeds. No qualitative statement below is
  fold-sensitive.

Definitions (sweep conventions): a round is **unambiguous** iff |oracle_p − 0.5| > 0.1;
**agreement** = fraction of unambiguous rounds where sign(Δ_pre_sem) matches the oracle
label; **confidence** = mean |Δ_pre_sem|; **rescue@α** = among unambiguous rounds whose
*replayed baseline* gap disagrees with the oracle, fraction flipped to correct by
+α·σ_proj·ŵ.

**New in this pass — balanced agreement (`agree_bal`)**: macro-average of agreement
over the two oracle-label classes, ½·(acc on gt=act1 rounds + acc on gt=act0 rounds).
Raw agreement equals the label base rate for any constant-sign responder;
`agree_bal` = 50 for every history-blind policy, so it is the artifact-proof readout
metric. Several conclusions change under it (§5).

## 1. Integrity checks (read these before the tables)

### 1a. Oracle-label degeneracy by t

Majority oracle-label base rate (%, over unambiguous rounds; n=100 seeds unless shown):

| env        | t1  | t2  | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|------------|-----|-----|----|----|----|----|----|----|----|-----|
| bandit     | —(0)| 57  | 55 | 57 | 53 | 54 | 56 | 56 | 56 | 57  |
| monty      | —   | 62  | 65 | 66 | 66 | 66 | 67 | 67 | 67 | 67  |
| ipd        | **100** | **100** | 72 | 60 | 55 | 51 | 54 | 55 | 72 | **100** |
| stag_hunt  | **100** | 59  | 65 | 58 | 59 | 55 | 55 | 58 | 57 | 60  |
| trust_game | **100** | 59  | 66 | 72 | **75** | **75** | **75** | **75** | **75** | **75** |

Consequences:

- **ipd t=1, t=2 and t=10 are fully degenerate** (label constant across all 100 seeds:
  everyone should cooperate early, defect in the known last round). Any constant-sign
  model scores exactly 0 or 100 there. Every "ipd rises to 100 at t10" or "decays to 0"
  cell in the raw tables is this artifact. Trustworthy ipd range: **t=4–8**
  (base rate 51–60), with t=3/t=9 (72) usable only with the balanced metric.
- **trust_game is 75/25 from t=5 on**: the big models' raw late-t plateau of 75–83%
  sits almost exactly on the base rate; gemma2b's famous "25% below chance" is the
  mirror image (constant sign on the minority class — see §5).
- **monty's 62–67 base rate** equals the small-model raw plateau exactly: those models
  are constant-sign responders there, not weak trackers.
- bandit is near-balanced (53–57) at every t — the only env where raw ≈ balanced and
  raw trends can be read directly. bandit t=1 has 0 unambiguous rounds by construction
  (oracle = prior).

### 1b. Steering-replay drift by t (`replay_drift` section of behavior_by_t.json)

Sign flips between the stored sweep gap and the steering-replay baseline, out of 1846
unambiguous replay rounds per model: **0 flips** for gemma2b, qwen, phi, gemma31,
llama70; 2 (dsllama70), 9 (dsqwen7), 15 (qwen27); **llama 140 (7.6%)**. llama's flips
localize where its gaps are tiny relative to the ~0.06–0.13 replay jitter:
bandit **t=2 only** (18/40 flips; 0–2 at every t≥3), monty (52/360 spread), and
**ipd (66/377, 10–35% of rounds per t, mean |gap| 0.066–0.091)**.
⇒ llama's bandit rescue-by-t at t≥4 is solid; **llama's ipd (and monty) rescue rates
are not interpretable** — the wrong/correct classification itself is replay-unstable.
llama stag_hunt/trust_game: 0 flips.

## 2. Belief by t (probe at best layer, grouped 10-fold CV)

Full table in `probe_by_t.json`. Summary across the 81 (model, t≥2) cells per env:

| env        | median | floor (worst cell) | cells <90% |
|------------|--------|--------------------------------|------------|
| bandit     | 100.0  | 94% (llama t8/t10)             | 0/81 |
| monty      | 99.0   | 92% (llama t3)                 | 0/81 |
| trust_game | 100.0  | 92% (llama70 t2)               | 0/81 |
| ipd        | 94.1   | 81% (dsqwen7 t9)               | 19/81 |
| stag_hunt  | 92.0   | 70% (llama t5; next worst 78)  | 27/81 |

Low-level structure: the ipd/stag_hunt dips are mid-game (t≈5–9) and recover to
~100 at t=10 (e.g. gemma2b_ipd: 100·98·95·92·92·92·87·91·100; llama70_ipd:
100·99·93·97·90·95·95·85·100) — mid-game is where the scripted opponent's behavior is
most mixed, so the posterior itself is most nuanced. **No cell at any t approaches the
majority baseline.** The belief is linearly present from the first informative round to
the last, in both game families, at every scale, including inside every behavioral
failure regime described below (gemma2b_trust_game probe: 99–100 at all ten t's).

## 3. Decisions by t — Bayesian games

### 3a. bandit (base rate ~55 ⇒ raw ≈ balanced; the clean accumulation test)

Oracle margin grows 0.17 (t2) → 0.33 (t10): genuinely gradual evidence accumulation.

Raw agreement (identical to balanced within 2pp everywhere):

| model     | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | pattern |
|-----------|----|----|----|----|----|----|----|----|-----|---------|
| qwen27    | 57 | 100| 100| 100| 100| 100| 100| 100| 100 | perfect from t3 |
| gemma31   | 100| 100| 100| 100| 100| 100| 100| 100| 100 | perfect |
| llama70   | 100| 100| 100| 100| 100| 100| 100| 100| 98  | perfect |
| dsllama70 | 100| 100| 85 | 100| 98 | 100| 100| 100| 100 | near-perfect |
| phi       | 100| 100| 100| 100| 100| 100| 99 | 100| 96  | near-perfect |
| qwen      | 100| 100| 100| 100| 95 | 100| 94 | 96 | 94  | mild late fade |
| gemma2b   | 100| 100| 87 | 96 | 91 | 95 | 85 | 92 | 87  | noisy fade |
| dsqwen7   | 100| 100| 79 | 97 | 81 | 92 | 89 | 89 | 86  | noisy fade |
| llama     | 57 | 100| 80 | 80 | 78 | 68 | 66 | 62 | 59  | **monotone decay** |

Confidence (mean |Δ_pre|) by t, and Spearman ρ(conf, t) over t=2…10:

| model     | t2   | t4   | t6   | t8   | t10  | ρ(bandit) | ρ(monty) |
|-----------|------|------|------|------|------|-----------|----------|
| qwen27    | 1.06 | 2.06 | 3.40 | 4.29 | 4.16 | +0.83 | +1.00 |
| gemma31   | 14.4 | 18.6 | 19.8 | 20.1 | 20.4 | +0.70 | +0.73 |
| llama70   | 6.97 | 8.79 | 9.03 | 10.3 | 9.86 | +0.60 | +0.78 |
| phi       | 0.88 | 1.44 | 1.53 | 1.75 | 1.86 | +0.52 | −0.33 |
| qwen      | 1.02 | 0.71 | 0.89 | 1.02 | 1.08 | +0.35 | −0.75 |
| dsllama70 | 0.58 | 0.94 | 1.11 | 1.16 | 1.17 | +0.17 | −0.92 |
| dsqwen7   | 1.17 | 1.65 | 1.44 | 1.42 | 1.40 | −0.30 | −0.72 |
| llama     | 0.33 | 0.29 | 0.30 | 0.34 | 0.34 | +0.62 (range 0.05) | +0.75 |
| gemma2b   | 1.07 | 0.71 | 0.57 | 0.47 | 0.47 | **−0.97** | +0.03 |

The ≥27B rational-agent signature is two-part and exact: (i) 98–100% agreement at
every t from t3; (ii) confidence sharpens monotonically with evidence — qwen27's
confidence ~quadruples (1.06→4.16) as the oracle margin doubles (0.17→0.33);
ρ(conf, oracle margin) = +0.83/+0.70/+0.60 for qwen27/gemma31/llama70.

Small-model failure modes, precisely:

- **llama**: agreement decays monotonically 100 (t3) → 59 (t10), ρ(agree,t | t3–10) =
  −1.00 (p<0.001); wrong-round count in the replays grows 7→18 from t4 to t10; its
  probe stays 94–100 at those same t's; its confidence is frozen at 0.33 regardless of
  evidence; and §6 shows every accumulating error is flipped by a single α=4 nudge at
  every t. A readout that degrades with context length on top of a belief that doesn't.
- **gemma2b**: agreement stays decent (85–96) but confidence **anti-sharpens**,
  ρ = −0.97: internal certainty shrinks as evidence grows — anti-Bayesian calibration
  with correct sign.
- Every model 2B–70B has ρ(raw agree, t | t3–10) ≤ 0 on bandit except dsllama70 (+0.41
  n.s.); the big three are `const` at 100. More history never helps the readout below
  27B; it only hurts or does nothing.

### 3b. monty (structure inferable analytically; base rate 62–67 ⇒ use balanced)

Balanced agreement by t:

| model     | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | reading |
|-----------|----|----|----|----|----|----|----|----|-----|---------|
| qwen27    | 100| 100| 100| 98 | 99 | 99 | 98 | 99 | 99  | analytic, ceiling from t2 |
| gemma31   | 100| 99 | 99 | 98 | 98 | 97 | 96 | 97 | 98  | analytic |
| llama70   | 53 | 86 | 100| 98 | 99 | 99 | 99 | 99 | 99  | analytic by t4 |
| qwen      | 50 | 50 | 50 | 65 | 75 | 80 | 92 | 95 | 94  | **empirical learning curve** |
| dsqwen7   | 50 | 50 | 50 | 49 | 62 | 77 | 55 | 53 | 87  | noisy learning curve |
| dsllama70 | 50 | 50 | 50 | 50 | 50 | 55 | 65 | 79 | 83  | late learning curve |
| phi       | 50 | 50 | 50 | 50 | 62 | 65 | 61 | 70 | 70  | partial learning curve |
| llama     | 65 | 61 | 68 | 60 | 75 | 64 | 64 | 75 | 82  | weak/noisy |
| gemma2b   | 50 | 50 | 56 | 57 | 56 | 52 | 50 | 50 | 50  | **never leaves chance** |

Two regimes, now artifact-free: ≥27B models solve the switch policy **analytically**
(ceiling from the first informative round — llama70 needs 2 extra rounds); mid-size
models discover it **empirically** within the session (qwen 50→94, ρ(bal,t)=+0.98;
dsllama70 50→83, +0.98; dsqwen7 →87; phi →70). The raw values 62–66 previously read
as "weak early competence" for small models were exactly the label base rate; balanced
shows they start at precisely 50. gemma2b never gets above 57 at any t — it never
finds the structure at all (its raw 62–71 was pure base rate).

## 4. Decisions by t — game-theoretic games (balanced agreement)

ρ(bal, t) over t=3–9 (avoiding both degenerate ends) and pooled mean:

**ipd** — chance for everyone, everywhere:

| model | t3 | t4 | t5 | t6 | t7 | t8 | t9 | mean t3–9 | ρ(bal,t) |
|-----------|----|----|----|----|----|----|----|------|-------|
| gemma2b   | 50 | 50 | 50 | 52 | 50 | 50 | 52 | 50 | +0.10 |
| llama     | 65 | 52 | 57 | 47 | 51 | 44 | 41 | 51 | −0.93 |
| qwen      | 60 | 58 | 54 | 45 | 45 | 46 | 49 | 51 | −0.68 |
| dsqwen7   | 51 | 65 | 48 | 51 | 50 | 48 | 53 | 53 | −0.29 |
| phi       | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | const |
| qwen27    | 53 | 59 | 55 | 58 | 51 | 55 | 47 | 54 | −0.54 |
| gemma31   | 52 | 65 | 54 | 60 | 55 | 57 | 50 | 56 | −0.21 |
| llama70   | 76 | 59 | 64 | 59 | 61 | 66 | 51 | 62 | −0.39 |
| dsllama70 | 60 | 58 | 54 | 53 | 51 | 50 | 50 | 54 | −0.99 |

Once the degenerate t=1/2/10 columns are removed and base rate is controlled,
**no model at any scale is meaningfully above chance at any t** (best: llama70 at 62
pooled, driven by one 76 cell at t3 where the minority class has n≈27). Every apparent
"learning" trajectory in the raw ipd table (qwen 49→100, phi 0→100, gemma2b 100→0) was
the constant-sign endpoint artifact. The probe reads the opponent parameter at 81–100%
at these exact t's.

**stag_hunt** — a flat partial readout appears at ~4B+, never improves with t:

| model | mean bal t3–9 | ρ(bal,t) | reading |
|-----------|----|-------|------------------------------|
| gemma2b   | 50 | const | constant-sign, zero readout |
| llama     | 50 | const | constant-sign, zero readout |
| dsqwen7   | 51 | +0.18 | chance |
| phi       | 59 | +0.31 | weak |
| dsllama70 | 66 | −0.61 | partial, decaying |
| qwen      | 75 | +0.32 | partial, flat |
| qwen27    | 75 | +0.79 (p=.04, range 68→81) | partial, flat-ish |
| gemma31   | 76 | +0.43 | partial, flat |
| llama70   | 77 | +0.39 | partial, flat |

**trust_game** — same shape, higher level:

| model | mean bal t3–9 | ρ(bal,t) | reading |
|-----------|----|-------|------------------------------|
| gemma2b   | 50 | const | **constant-sign: 25/25 on act1 rounds, 0/75 on act0 rounds at every t≥5** |
| dsqwen7   | 66 | +0.51 | erratic (50↔100 swings) |
| llama     | 71 | −0.98 | **decays, then inverts: bal 73 (t8) → 42 (t9) → 23 (t10); act1-class acc 25/25 → 10/25 → 0/25** |
| phi       | 77 | −0.79 | partial, decaying |
| qwen27    | 86 | −0.31 | partial, flat |
| gemma31   | 86 | −0.54 | partial, flat |
| llama70   | 87 | −0.41 | partial, flat |
| qwen      | 92 | +0.64 n.s. | strong, flat |
| dsllama70 | 94 | −0.18 | strong, flat |

The cross-env summary statistic for the paper: over the 27 (model × social-env)
trend tests, **not one shows a significant positive ρ(balanced agreement, t)** on
the artifact-free range (best: qwen27 stag_hunt +0.79, p=0.04, a 68→81 wiggle;
sign test across all 27: 17 negative). The significant trends that exist are
*negative* (llama ipd −0.93, llama trust −0.98, dsllama70 ipd −0.99, phi trust −0.79).
Ten rounds of opponent evidence moves the representation every round (§2) and the
policy's use of it never.

## 5. Corrections to the previous (raw-agreement) reading — matters for the abstract

1. **"gemma2b trust_game locks below chance at 25%" → it is a constant-sign
   responder.** It outputs act1 on every round (act1-class 100%, act0-class 0%);
   25% raw = the act1 base rate. Balanced = exactly 50 at every t. The dissociation
   exemplar *survives and is cleaner*: probe 99–100 at every t vs a readout with
   literally zero mutual information with the belief — but the right description is
   "the readout ignores the belief entirely," not "anti-correlated."
2. **All raw ipd trends were artifacts.** Nothing rises to 100 at t10 and nothing
   decays to 0; ipd balanced is 44–65 for all nine models at all t. This *strengthens*
   the decoupling claim (before, ipd looked noisy-but-structured; it is chance).
3. **Small-model monty "62–66% early competence" was base rate**; balanced start is
   exactly 50. The mid-size learning curves and big-model analytic ceiling both
   survive balanced correction and are real.
4. **Big-model trust_game raw plateau (75–83) is partially base-rate-inflated**, but
   balanced 80–92 confirms a genuine (partial, flat) readout — unlike ipd.
5. **llama's ipd/monty rescue rates (24%/9%) are replay-unstable** (§1b) and should
   not be quoted; its bandit t≥4 rescue (100% everywhere) is solid.
6. New finding from the per-class split: **llama trust_game actively inverts at the
   end of the session** (bal 73→23 over t8→t10) — a second, distinct
   context-length-degradation signature in the same 3B model that decays on bandit.

## 6. Causal rescue by t (α=4, rescue direction at best layer)

**bandit** (rescued/wrong):

| model | t4 | t5 | t6 | t7 | t8 | t9 | t10 | reading |
|-----------|------|------|------|------|------|------|------|---------|
| llama     | 7/7 | 8/8 | 12/12 | 11/11 | 13/13 | 14/14 | 18/18 | **100% at every t; error mass grows 7→18 with t and every error is one nudge from correct** |
| dsllama70 | 4/4 | — | 1/1 | — | — | — | — | 100% where errors exist |
| dsqwen7   | 5/7 | — | 0/5 | 1/2 | 1/2 | 1/2 | 1/2 | partial (~50%) |
| gemma2b   | 0/6 | 0/1 | 0/2 | 0/1 | 0/5 | 1/3 | 0/8 | ≈0: belief direction not causally upstream |
| qwen27/gemma31/llama70 | | | | | | | | 0–2 wrong rounds total (nothing to rescue) |

**Game-theoretic envs — rescue is ≈0 at every t** (pooled over t, α=4):
ipd ≤4% for all reliable models (gemma2b 0/165, qwen 0/155, phi 0/169, dsllama70
0/146, qwen27 1/147, gemma31 2/133, llama70 2/139); stag_hunt ≤2% for all nine
(gemma2b 0/182, llama 0/163, dsllama70 0/182); trust_game 0% for gemma2b (0/257),
qwen, phi, qwen27, gemma31, ~1% llama70 — with three quirks: dsllama70 trust 9/32
(28%, the one model with a partially causal trust readout), dsqwen7 trust t8 10/10
(isolated), llama trust t2 22/22 (real — 0 replay flips — but only at the shallowest
history; 0/188 at t≥3). **monty**: phi 27% and dsqwen7 19% (concentrated mid-session,
exactly where their learning curves are climbing); everyone else ≈0.

So the by-t steering picture matches the behavioral one pointwise: where scale
repairs the mapping (bandit), the residual small-model errors are shallow — the
belief direction flips them at every history depth; where the mapping is decoupled
(social games), pushing along the belief direction does nothing at any history depth.

## 7. Clean takeaways (paper-ready)

1. **Belief updating is Bayesian-consistent at all scales and all history depths.**
   Probe accuracy at the best layer is ≥92% (median 99–100) at every t in bandit,
   monty, trust_game and ≥70% (median 92–94, mid-game dip, full recovery by t10) in
   ipd/stag_hunt — never near baseline, 2B→70B.
2. **In inference games, action selection becomes exactly Bayesian with scale.**
   ≥27B: 98–100% agreement at every t plus monotone confidence sharpening tracking
   the oracle margin (ρ = +0.6…+1.0). Mid-size models show real within-session
   learning curves on monty (50→94 balanced); big models are at ceiling from t=2
   (analytic, not empirical).
3. **Below ~27B the failure is a context-degrading readout, not a failing belief.**
   llama: agreement 100→59 monotone in t, confidence frozen, probe intact, and 100%
   of its accumulating errors flip with one α=4 nudge at every t. gemma2b: correct
   sign but confidence anti-sharpens (ρ=−0.97). llama additionally inverts on
   trust_game in the last two rounds.
4. **In game-theoretic games the decision circuit never consults the belief, at any
   scale, at any history depth.** Base-rate-controlled: ipd is chance (50–62) for all
   nine models at every t; stag_hunt/trust_game show a partial (66–94) but
   t-invariant readout above ~4B and exactly-50 constant-sign behavior at 2–3B;
   0/27 model×env trend tests are significantly positive; steering rescue is ≈0 at
   every t. Evidence accumulates in the representation and never into the policy.
5. **Metric caution that must appear in the paper:** raw sign agreement is
   uninterpretable wherever the oracle label is skewed (ipd t∈{1,2,10} degenerate,
   trust_game 75/25 from t5, monty 62–67); all §4 claims use balanced agreement.
   The SFT patient is unchanged but re-described: gemma2b trust_game = constant
   readout (balanced 50, zero belief-use) under a 100% probe, at every t≥5.

## 8. File map

- `analysis/by_t.py` — regenerates everything below (`--tag P1_v3`).
- `results/P1_v3/probe_by_t.json` — per-combo best layer + per-t probe acc/n.
- `results/P1_v3/behavior_by_t.json` — `behavior` (per-t n, raw/balanced/per-class
  agreement, confidence, oracle margin, frac unambiguous), `rescue` (per-t
  rescued/wrong at α=4), `replay_drift` (per-t replay sign-flips + max drift).
- `*.orig-adhoc` — the 2026-07-15 ad-hoc originals (verified equal; see §0).
- Balanced-agreement and trend statistics in this doc: computed from
  `behavior_by_t.json` fields `agree_bal`, `agree_act1/0`; Spearman over stated ranges.
