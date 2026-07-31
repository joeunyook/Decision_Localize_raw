# Supplementary Material — Assembly and Working Notes

Built 2026-07-31 from `DOCUMENTATION.txt`, `AnonymousSubmission2027.tex`, the
`paper_*.tex` fragments, and live re-runs of the analysis scripts. Every number
below is either (a) recomputed in this session and marked **[V]**, or (b) carried
from a documented earlier session and marked **[D]** with its provenance. Nothing
is quoted from memory alone.

---

## 0. Where things stand

**Main paper** (`AnonymousSubmission2027.tex`, 8 scenarios, 9 models) is written
through Results / Discussion. It cites four appendices:

| Label | Cited at | Status |
|---|---|---|
| `app:posterior` | Results, belief paragraph (l. 988) | **exists** — `paper_appendix_posterior.tex` (180 l., formal, with table) |
| `app:steering` | *never cited in main text*; the 07-29 replacement paragraph cites it | **exists** — `paper_appendix_causality.tex` (400 l., two tables, threats-to-validity) |
| `app:qualitative-examples` | Discussion (l. 1092) | **exists** — inline in the main `.tex`, 3 boxed CoT transcripts |
| `app:models` | Setup, Model Selection (l. 659) | **MISSING — must be written** |
| `app:prompts` | Setup, Prompts and Decoding (l. 673) | **MISSING — must be written** |

**The supplementary draft you pasted is behind the repo.** Its posterior and
steering sections are condensed prose versions; the repo files are longer, have
the tables, and are more defensible. Start from the repo files, not the draft.
The draft's steering section in particular still contains the wording the
2026-07-29 audit flagged (§5 below).

---

## 1. Verified core numbers (re-run this session)

### 1.1 Steering — `python3 analysis/steer_defense.py --tag P1_v4` **[V]**

Reproduces the 2026-07-29 audit exactly. Population: 26,334 unambiguous steps
(\|p\*−0.5\|>0.1), 9 models × 8 scenarios × 40 seeds. IPD adds 3,393 more and is
excluded. All CIs are 95% percentile bootstrap resampling **whole sessions**.

**Partition A — task type** (what the main text uses)

| Family | n_wrong | rescue α=4 | random | diff | dose (α=1/2/4/8) |
|---|---|---|---|---|---|
| Bayesian (4) | 2387 | 27.6% | 7.8% | **+19.8 [+16.8,+22.7]** | 11.1 / 16.5 / 27.6 / 38.1 |
| Game (4) | 4651 | 2.9% | 2.0% | **+0.9 [+0.2,+1.5]** | 1.7 / 2.3 / 2.9 / 4.6 |

Interaction +18.9 pp [+15.8,+21.9].

**Partition B — opponent reactivity** (what both appendices use)

| Family | n_wrong | rescue α=4 | random | diff | dose |
|---|---|---|---|---|---|
| Nonreactive (5) | 3458 | 21.7% | 6.4% | **+15.3 [+13.0,+17.6]** | 8.9 / 13.3 / 21.7 / 30.8 |
| Reactive (3) | 3580 | 1.2% | 1.7% | **−0.5 [−0.9,−0.1]** | 1.0 / 1.1 / 1.2 / 1.6 |

Interaction +15.8 pp [+13.5,+18.2].

**Per scenario (α=4)** — bandit 61.0/8.2 (+52.8), machine 34.0/9.9 (+24.1),
trust 24.4/6.1 (+18.3), monty 10.4/6.4 (+3.9), chicken 8.7/3.3 (+5.4),
pennies 1.6/1.8 (−0.2), bos 1.3/1.7 (−0.4), stag 0.7/1.6 (−0.9).
IPD (dropped) 3.0/5.0 (−2.0).

**Damage on already-correct steps (nonreactive / reactive), α=1/2/4/8**

| Condition | Nonreactive | Reactive |
|---|---|---|
| anti-direction | 1.4 / 2.6 / 5.0 / 10.5 | 0.8 / 0.8 / 0.9 / 1.3 |
| random | 0.9 / 0.9 / 1.1 / 1.5 | 0.8 / 0.9 / 0.9 / 1.2 |
| rescue-direction | 0.5 / 0.5 / 0.6 / 0.5 | 0.7 / 0.7 / 0.6 / 0.9 |

Bidirectional dose-response dissociation. Anti applied to *wrong* rounds does
not rescue them (nonreactive 4.7/3.7/3.8/5.0; reactive 1.1/1.1/1.1/1.3).

**Matched-margin control** (rescue% / random%, n) — answers "your Bayesian errors
were just hairline":

| \|gap\| bin (nats) | Nonreactive (5) | Reactive (3) |
|---|---|---|
| [0.00,0.25) | 41.6 / 18.5, n=1191 | 6.6 / 7.4, n=579 |
| **[0.25,0.50)** | **19.3 / 0.3, n=716** | **0.3 / 0.3, n=698** |
| [0.50,1.00) | 5.8 / 0.0, n=773 | 0.2 / 0.4, n=1315 |
| [1.00,2.00) | 21.0 / 0.0, n=352 | 0.0 / 2.4, n=359 |
| [2.00,∞) | 0.0 / 0.0, n=426 | 0.0 / 0.2, n=629 |

The [0.25,0.50) row is the one to print: matched n, matched confidence, identical
random controls, ~60× difference in rescue.

**Per-model, reactive (3):** 0 of 9 models beat their own random control; two are
significantly negative (gemma2b −1.0 [−1.8,−0.2]; gemma31 −3.9 [−6.9,−1.0]).
**Per-model, nonreactive (5):** 6 of 9 positive with CI excluding zero
(llama +25.4, qwen27 +24.7, gemma2b +22.8, phi +16.4, dsqwen7 +13.8,
dsllama70 +5.4); gemma31 +1.1 marginal; llama70 −0.4 and qwen −0.5 null.

**Single-model exposure:** Chicken's +5.4 is entirely Phi-4-mini
(phi 54.5/5.8 = +48.7 on n=112). Chicken minus phi = +0.3 [−0.6,+1.1], null.
Everything else survives leave-one-out (Bayesian minus phi +22.1 [+18.8,+25.3]).

### 1.2 CoT amplification — `python3 analysis/cot_amplification.py` **[V]**

72 combos, 26,334 scored steps, answers re-parsed with the current last-match /
`\boxed{}` regex.

Game-theoretic family, per model: P2 → P3 effective, coverage, salvage, break —
r(P2, effective gain) = **0.72** (n=7 non-reasoning models). Qwen-3.5 4B is the
selection trap (P3 terminated 91.8 vs effective 66.3 at 72.1% coverage).

Termination by scenario (reasoning-distilled) — the cleanest monotone in the
suite: R1-Qwen 7B bandit 51% → stag_hunt 2%; R1-Llama 70B bandit 97% →
pennies 12%.

### 1.3 Belief, final layer, balanced **[V]** (recomputed from `results/P1_v4` + `P1_v4_layerbal`)

Mean over 9 models: **Bayesian 99.4, game 92.3, all 8 = 95.8** — matches the main
text's 95.8 / 99.4 / 92.3 exactly. Per scenario: Bandit 99.4, Monty 98.9,
Trust 99.6, Machine 99.6, Chicken 98.5, BoS 90.5, Stag 90.3, Pennies 89.8.

### 1.4 Perturbation magnitude — **[V], and this one changes a conclusion**

See §5.1. Pooled ratio (probe \|Δgap\| / random \|Δgap\|) is 0.48 nonreactive /
0.12 reactive; **excluding Gemma-4 31B it is 2.90 / 1.07.** Per model, α=4:

| Model | nonreactive probe/random | ratio | reactive ratio |
|---|---|---|---|
| gemma2b | 0.272 / 0.052 | 5.27 | 1.11 |
| llama | 0.233 / 0.046 | 5.09 | 1.12 |
| qwen27 | 0.257 / 0.066 | 3.89 | 1.22 |
| phi | 0.358 / 0.100 | 3.59 | 1.00 |
| dsqwen7 | 0.153 / 0.059 | 2.58 | 1.00 |
| llama70 | 0.589 / 0.283 | 2.08 | 1.15 |
| dsllama70 | 0.099 / 0.052 | 1.90 | 0.97 |
| qwen | 0.038 / 0.032 | 1.21 | 1.06 |
| **gemma31** | **0.326 / 4.148** | **0.08** | **0.05** |

### 1.5 Narrative-framing ablation — **NEW, `python3 analysis/ablation_dry.py`** **[V]**

Data has been on disk since 2026-07-25 and had **never been aggregated**. Both
arms complete: `ABL_dry_v1` 9×9×30 pre-CoT sessions, `ABL_dry_cot_v1` 9×9×15
post-CoT sessions. Prompts are numerically identical and differ only in story
framing (`prompts/{env}_forced.txt` vs `{env}_forced_dry.txt`).

| Checkpoint | Bayesian Δ | Game Δ | All 8 Δ |
|---|---|---|---|
| P2 pre-CoT (seeds 0–29, exactly matched) | +0.2 | −0.1 | **+0.1** |
| P3 post-CoT (seeds 0–14, exactly matched) | −5.3 | −4.4 | **−4.8** |
| P1 belief probe (100 vs 30 seeds — **confounded**) | −1.5 | −6.2 | −3.9 |

**Reading.** The pre-CoT belief-to-action gap is *not* an artifact of
anthropomorphic framing: stripping every narrative cue moves mean agreement by
0.1 pp. Cell-level scatter is real though (72 cells: mean +0.08, median +0.55,
mean \|Δ\| 8.28, range −41.1…+28.9, 25/72 cells move >10 pp) — 30 seeds is thin;
present the mean, disclose the scatter.

Post-CoT is a different story and is a **finding, not noise**: dry framing costs
4.8 pp of end-to-end accuracy and 6.1 pp of coverage on average
(gemma2b −14.8 pp coverage, qwen27 −8.1). Narrative scaffolding appears to help
models *terminate*, not to help them decide. Gemma-4 31B and Llama-3.3 70B are
unaffected on both (Δ ≈ 0, coverage 100% in both arms).

⚠️ **The P1 row is not yet reportable** — canonical probes are trained on 100
seeds, dry probes on 30, so the −3.9 pp conflates framing with training-set size.
Fix = retrain canonical probes on seeds 0–29 only (CPU, `probe_train.py`,
thread-pinned fan-out on a compute node, ~20 min). Decide before this goes in.

### 1.6 Tally ablation — **[D]** `DOCUMENTATION.txt` 2026-07-29, data in `results/ABL_notally_v1`

6 models × 8 scenarios = 48 pairs, 15 seeds, `--no-tally`. Mean Δ **+0.8 pp**,
mean \|Δ\| 1.3, worst single drop −4.4 (qwen×bandit), floor 74.0%, 0/48 drop more
than 5 pp. Paper sentence already drafted in the log. This closes "the probe just
reads the printed tally".

---

## 2. Results that exist and are documented but NOT recomputed this session

Quote-safe, but re-run before submission if a number is load-bearing.

| Result | Numbers | Source / regenerate |
|---|---|---|
| **Posterior decodability** | R²_full 0.787, R²_partial 0.451, R²_shuf −0.011; 63/63 combos positive; per-scenario table | `analysis/posterior_probe.py` **[D 07-26]** — already written up in `paper_appendix_posterior.tex` |
| **Tally-counts baseline** (linear + quadratic) | residual 0.451 vs counts 0.259 (linear) and 0.369 (quadratic); **counts beat the residual in all three reactive scenarios** (0.601/0.600/0.557 vs 0.460/0.453/0.382) | `analysis/tally_control.py` **[D 07-26]**. Already conceded in `paper_discussion.tex` Limitations. Belongs in the supplementary verbatim — volunteering it is worth more than hiding it |
| **ΔU vs Δz transfer function** | ρ: Bayes-pure 0.84, Bayes-EU 0.79, game 0.51, IPD 0.24. Within-model game/Bayes slope ratio: all 9 below 1, median 0.29 | `analysis/du_dz.py` **[D 07-26]**. Never used in the paper — strong unused supplementary section |
| **Graded-belief control** | 63/63 combos, Spearman(probe projection, continuous oracle posterior) ρ = 0.586–0.981, mean 0.84 | **[D 07-22]**, computed on the then-63-combo set; needs a re-run to cover all 81 (needs `residuals/`, which is a symlink to scratch) |
| **Layer-wise belief curves / D90** | Bayesian 50→99.4, Chicken 50→98.5, reactive 50→90.2 at 0→100% depth; D90 nonreactive 2–50, bos/pennies 32–61, dsllama70×stag 78 | `analysis/plot_belief_layers.py`, figure at `plots/belief_layers/` **[D 07-25]** |
| **RQ2 evidence tracking** | pre ρ 0.542 Bayesian / 0.293 game; post 0.305 / 0.259 — CoT *degrades* expressed confidence calibration while improving accuracy | **[D 07-25]** |
| **Final vs best layer** | best 98.6, final 97.8, median-layer 94.3; 41% of all layers within 1 pp of peak | **[D 07-25]** — the "you cherry-picked a layer" answer |
| **Replay fidelity** | mean \|Δ\| 0.019 nats, median 0.000, p99 0.188, sign agreement 99.16% over 28,800 steps | already in `paper_appendix_causality.tex` |
| **Letter-bias audit** | gemma2b mean \|letter_bias\| 3.08 = 4–10× every other model, sign constant across all 7 scenarios | **[D 07-22]**, script was scratch-only and **lost** — must be rewritten if used (the field `letter_bias` is saved in every `responses/*.json` round) |
| **Reasoning-model non-termination** | per-model per-scenario concludes/loops; 100% of non-terminated rounds hit the 2048 cap; Qwen-3.5's stuck rounds are budget exhaustion scaling with t (6%→40%), R1 models' are history-independent (flat 88–98%) | **[D 07-24 / 07-26]** |
| **IPD exclusion evidence** | clock-only predictor reproduces the oracle on 70.5% of steps vs 56.3% base rate; DEFECT strictly dominates (EU gap ≥ 1 for every posterior); IPD belief probe is the suite's lowest | **[D 07-24/25]**, `analysis/ipd_label_diagnostics.py` |
| **Scenario design guards** | per-scenario clock-only gain: pennies +0.3, stag +0.0, chicken +1.1, bos +3.6 vs **ipd +16.6**; oracle calibration on the diagonal at every bin; realized-payoff optimality 0.835 vs 0.165 vs clairvoyant 1.0 | **[D 07-25]**, `analysis/verify_bos.py` and the pennies guards |

---

## 3. Assets on disk for the two missing appendices

**`app:models`** — everything needed is in `core/model_loader.py`: HF ids, labels,
and the three non-uniform configs that must be disclosed (R1-distill sampling
temp 0.6 / top-p 0.95 per DeepSeek's card; `<think>\n\n</think>` prefill for the
pre-CoT forced read; the `dsllama70` tokenizer patch — its published
`tokenizer.json` ships with no functioning pre_tokenizer, repaired at load time
from the Llama-3.3-70B sibling). Layer counts are recoverable from the probe
JSONs (gemma2b 27 … llama70 81). **The tokenizer patch must be disclosed** — it
is a deviation from the released artifact.

**`app:prompts`** — 36 files in `prompts/` (9 envs × {forced, cot} × {canonical,
dry}). The paper uses 8 envs → 32 files. Include one full rendered example
(prompt + history + tally + answer-format block) plus the dry twin, and state
that the tally line is computed by `format_history()`/`tally_block()`.

---

## 4. Proposed supplementary structure

Ordered so each section answers the objection raised by the previous one.

- **A. Models** *(new)* — table, decoding config, the three deviations above.
- **B. Scenarios and baseline** — full spec per scenario; the corrected Eq. (1)–(2)
  for reactive opponents (see §5.3); design guards (clock-only gain, oracle
  calibration, realized-payoff optimality, ambiguity band in EU units).
- **C. Prompts** *(new)* — canonical and dry renderings, tally block, answer format.
- **D. Probe 1: construction and posterior decodability** — from
  `paper_appendix_posterior.tex`; **add** the tally-counts baseline (§2) and the
  no-tally ablation (§1.6).
- **E. Causal intervention: protocol and validation** — from
  `paper_appendix_causality.tex`; **add** the matched-margin table, the full
  damage table, and the disclosure items in §5.2.
- **F. Chain of thought** — amplification/salvage/break table, coverage, the
  termination gradient, and the two non-termination mechanisms.
- **G. Belief-to-action transfer function** — ΔU vs Δz (unused, strong).
- **H. Ablations** — narrative framing (§1.5), tally removal (§1.6), and the
  layer-selection / final-vs-best-layer analysis.
- **I. A scenario we designed out** — IPD, with the clock-only number.
- **J. Representative CoT failure cases** — the three boxed transcripts, kept as is.

---

## 5. Issues that must be resolved before writing prose

### 5.1 The perturbation-magnitude claim is a one-model artifact — **do not print it as drafted**

`DOCUMENTATION.txt` (07-29, §7 "MOST DANGEROUS UNFORCED ERROR") concludes that
the probe direction moves the action logits *less* than a norm-matched random
vector, and its recommended appendix insert says *"the mean absolute change …
is 0.48 times that under norm matched random directions … so the control is the
larger perturbation in both cases."*

Recomputed this session: that is true **only because Gemma-4 31B's random
directions displace the gap by 4.15 nats** (against 0.33 for its probe
direction), and pooling raw nats across models with 100× different logit scales
lets that one model set the mean. Drop gemma31 and the pooled ratio flips from
0.48 → **2.90** (nonreactive) and 0.12 → **1.07** (reactive); 8 of 9 models have
probe > random in the nonreactive family (1.2–5.3×) and ≈1.0 in the reactive one.

`paper_appendix_causality.tex` already handles this correctly — it reports ρ per
model and explicitly excludes gemma31 from any ρ-based claim. **Use the per-model
version. Do not paste the pooled sentence from the 07-29 log into the
supplementary**; it inverts the conclusion for eight of nine models and a referee
who bins by model will find it. If a pooled number is wanted, report it excluding
gemma31 and say so.

Also still open from that log: `analysis/steer_causality.py`'s docstring and
printed legend still say a ratio near 1.0 means the belief "is decodable but not
read by the answer-token circuit" — i.e. a shipped script whose legend appears to
refute the paper. Fix before release.

### 5.2 The two partitions still disagree in print

Main text uses 4 Bayesian / 4 game (27.6 vs 2.9). Both appendices use
5 nonreactive / 3 reactive (21.7 vs 1.2). Both are arithmetically correct; the
difference is Chicken. The paper's own taxonomy (belief figure caption, belief
prose, both appendices) treats Chicken separately, so only the causal paragraph
is out of step. Two things follow:

1. The main text's *"indistinguishable from a random perturbation"* is **false as
   written** — games(4) is +0.9 pp with CI [+0.2,+1.5], which excludes zero. Under
   the reactivity partition it becomes true and stronger (−0.5 [−0.9,−0.1]).
   The 07-29 log has a full replacement paragraph (§13) ready to paste.
2. Whichever way it lands, the supplementary needs one sentence saying the two
   numbers are two cuts of one experiment.

Undisclosed in the current appendix and cheap to add: the direction is refit on
all rounds including the evaluated ones (legitimate — no held-out accuracy claim
is made — but say so); ℓ\* is an argmax over layers; "two isotropic random
directions" should read "two directions per (model, scenario), each evaluated on
every round, generated from a fixed per-combo seed"; **the random controls carry
the same oracle sign as the rescue condition** (`run_steer.py:203`) — spell this
out or referees will assume a 50% handicap and discount the comparison; and the
intervention is oracle-signed, so the rescue rate is an upper bound on
recoverable error, not a deployable correction.

The user's pasted draft also still says *"with no increasing dose-response in
either sign"*, which is a hair too absolute: reactive rescue does rise
1.0→1.1→1.2→1.6%, but the random control rises faster (1.1→1.2→1.7→2.2%).
Correct wording: *"rescue does not exceed the random control at any intervention
strength."*

### 5.3 Method-level debt the supplementary inherits

- **Eq. (1)–(2) are too narrow for a reactive opponent.** The observation and
  payoff depend on the agent's *previous move*, which lives in h_t, not in θ.
  This already applies to Tit-for-Tat in Stag Hunt and BoS in the submitted
  version. Exact three-line repair (still exact Bayes) is in the 07-25 log §2e.
- **Scenario count.** Abstract and parts of the Method say *seven*; the suite is
  *eight* and the table has eight rows with no IPD. Must be settled.
- **Abstract drift.** The `.tex` abstract contains *"unlike in Bayesian inference
  tasks, where mid-sized models rationally map beliefs to actions"*, reportedly
  not in the OpenReview submission. AAAI may reject on substantial abstract
  change. Reconcile down to the submitted text.
- **`references.bib` does not exist**, so the bibliography cannot build in either
  document. Every `\citep` in `paper_intro_related.tex` is a `VERIFY_*`
  placeholder.
- Main `.tex` housekeeping: stray `IPD` at line 1693, duplicate
  `\label{tab:main-results}` (l. 922 and 1755), and a whole leftover draft
  (old RQ subsections + an 8-column IPD-containing table + empty
  Discussion/Ablations/Conclusion) sitting between the qualitative appendix and
  `\end{document}` at ~l. 1790 — it currently compiles into the appendix.

### 5.4 Claims to soften

- *"The action is not a function of the represented belief"* is too strong as a
  general claim. The game null is **graded**: ΔU–Δz ρ = 0.51 and climbs with
  scale (gemma2b 0.17 → gemma31 0.80). It holds in the limiting cases
  (gemma2b × pennies ρ = −0.00 with belief probe 89%). Write "substantially less
  sensitive" and use the limiting cases as illustration.
- *"Amplification does not help because there is no strongly weighted pathway to
  amplify"* asserts an unmeasured mechanism. Replace with the functional-form
  argument (threshold on a scalar vs fixed point) — it predicts the *symmetric*
  anti-direction null, which the pathway-strength story does not. Full text in
  the 07-29 log §10.
- The old *"hairline errors in Bayesian vs confidently wrong in game-theoretic"*
  framing is **contradicted** under P1_v4: the two families' error-magnitude
  distributions largely overlap (median 0.44 vs 0.56 nats). Grep the draft for
  surviving instances.
- Belief is at ceiling and has ~30× less variance than action across models
  (SD 0.4–2.1 vs 12–17). Report belief once as a precondition with its chance
  level; make **the gap** the reported quantity.

### 5.5 Positioning

Sobotka, Karabag & Topcu, *"Why Do LLMs Struggle in Strategic Play? Broken Links
Between Observations, Beliefs, and Actions"* already owns the term
"belief-action gap", the finding that internal belief exceeds belief use, and a
prompt-level belief intervention. Must be cited in Related Work **and** in a
clause inside the causal paragraph. What is ours: activation-level rather than
prompt-level causality; the reactivity dissociation with matched controls
(Machine = Trust's de-socialized twin on paired seeds; Chicken = a game with a
non-reactive opponent); a Bayes-optimal oracle in every scenario; nine models
2B–70B; the CoT stage decomposition.

---

## 6. Open compute (all cheap, all CPU unless noted)

1. **Matched-seed canonical probes** (seeds 0–29) so the framing ablation's P1 row
   is interpretable. `probe_train.py`, thread-pinned (`OMP/MKL/OPENBLAS=8`), on a
   compute node — never the login node.
2. **Graded-belief ρ over all 81 combos** (currently 63). Needs `residuals/`,
   which is a symlink to `/capstor/scratch` — check it survived the 30-day purge
   before planning on it.
3. **Re-run `posterior_probe.py` / `tally_control.py` / `du_dz.py`** to confirm the
   07-26 numbers still reproduce on the current tree before they are typeset.
4. **Letter-bias aggregation** — script was lost; the `letter_bias` field is saved
   per round, so it is a ~20-line rewrite if the section is wanted.
5. *(GPU, optional but high value)* cosine similarity between each probe direction
   and the A/B unembedding difference `W_U[A] − W_U[B]`. Closes "you steered the
   output direction, not a belief" outright. Load 9 models, read two rows, dot
   with 72 saved directions.

---

## 7. Scripts (all in `analysis/`, all regenerate what is quoted above)

```
steer_defense.py       --tag P1_v4    both partitions + all four referee checks
steer_stats.py                        McNemar + bootstrap on the steering rates
steer_causality.py     --tag P1_v4    |Δgap| ratios  (LEGEND IS WRONG — see §5.1)
cot_amplification.py                  P2/P3 term/P3 eff/coverage/salvage/break
ablation_dry.py                       NEW — narrative-framing ablation (§1.5)
posterior_probe.py                    residual → p(θ|h_t) ridge with ΔU partial-out
tally_control.py                      counts baseline, linear and quadratic
du_dz.py                              ΔU vs Δz transfer function
plot_belief_layers.py                 layer-wise belief figure
plot_rq1_outcomes.py                  RQ1 outcome-decomposition figure
paperstyle.py                         shared rcParams — import before any figure
rebuild_probe_summary.py --tag P1_v4  run after ANY partial probe_train
reparse_answers.py                    offline re-parse of stored cot_text
```
