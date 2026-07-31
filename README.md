# Decision_Localize_raw

Complete working tree for the project **"Localizing LLM Failures in Decision
Making with Bayesian Optimal Baselines"** — code, prompts, all analysis
outputs, figures, paper sources, and the full session log.

This repository is a **recovery archive**, not a packaged release. It is a
faithful snapshot of the live working directory `/users/joeunyook/logit-probe`
on the CSCS Clariden cluster, minus bulk binary activations that do not fit in
git (see [What is NOT in this repo](#what-is-not-in-this-repo)).

Everything excluded is still on CSCS. **Nothing was deleted to create this
repository.**

---

## Table of contents

1. [What the project is](#what-the-project-is)
2. [Repository layout](#repository-layout)
3. [What is NOT in this repo](#what-is-not-in-this-repo)
4. [Restoring on CSCS](#restoring-on-cscs)
5. [Restoring off CSCS](#restoring-off-cscs)
6. [Environment](#environment)
7. [Regenerating each paper number](#regenerating-each-paper-number)
8. [Known traps](#known-traps)

---

## What the project is

A mechanistic pipeline that localizes *where* LLM decision-making fails, by
comparing model behaviour against an exact Bayesian-optimal baseline at three
probing points:

| Probe | What it reads | Metric name |
|---|---|---|
| **P1** Belief representation | residual stream, all layers, final prompt token | Belief (P1) |
| **P2** Belief→action mapping | vocabulary logits, same forward pass | Pre-CoT (P2) |
| **P3** Post-CoT commitment | logits at the committed answer, after generation | Post-CoT (P3) |

**Scale.** 9 models (2.6B–70B), 8 scenarios, 100 seeds × T=10 steps for P1/P2,
40 seeds for P3.

**Scenarios.** Bayesian inference family: `bandit`, `monty`, `trust_game`,
`machine`. Game-theoretic family: `stag_hunt`, `bos`, `pennies`, `chicken`.

**IPD is dropped from the paper** but all its code and data are retained here.
The exclusion is justified by a clock-only predictor reproducing its oracle on
70.5% of steps; `analysis/ipd_label_diagnostics.py` regenerates that number
from the committed `results/` and `responses/` JSONs. Do not delete the IPD
data — it is the evidence for the exclusion.

---

## Repository layout

```
.
├── core/                    model loading, probe read-out primitives
│   ├── model_loader.py      9-model registry; tokenizer patch; sampling config
│   ├── probes.py            forced-choice read, post_cot_generate, ANSWER_RE
│   ├── metrics.py  io_utils.py
│
├── envs/                    scenario definitions (posterior + oracle per env)
│   ├── base.py              shared contract: format_history(), tally_block()
│   ├── bandit.py  monty.py  trust_game.py  machine.py
│   ├── stag_hunt.py  bos.py  pennies.py  chicken.py
│   └── ipd.py               dropped from paper, retained
│
├── prompts/                 46 templates
│   └── {env}_{forced,cot}[_dry].txt
│                            forced = P1/P2 (no generation)
│                            cot    = P3
│                            _dry   = narrative-framing ablation
│
├── analysis/                every number in the paper comes from here
│   ├── probe_train.py       P1: LOSO, C=0.1, layer sweep, balanced accuracy
│   ├── posterior_probe.py   Appendix C.2: ridge + ΔU partial-out
│   ├── steer_defense.py     Appendix D: partitions, session bootstrap, bins
│   ├── steer_causality.py  steer_stats.py  steer_summary.py
│   ├── by_t.py              per-timestep aggregation tables
│   ├── cot_amplification.py Result 3: CoT as commitment amplifier
│   ├── ablation_dry.py      narrative-framing ablation
│   ├── tally_control.py     tally-removal ablation
│   ├── cotfail_*.py  extract_cot_failures.py  taxonomy.py   Appendix E
│   ├── reparse_answers.py   Bug 5 fix: last-match answer parsing
│   ├── rebuild_probe_summary.py    ALWAYS run after a partial probe_train
│   ├── belief_layers_table.py  plot_belief_layers.py  plot_rq1_outcomes.py
│   ├── paperstyle.py        figure style (Type-42 fonts, final-size authoring)
│   └── ipd_label_diagnostics.py  verify_bos.py  bos_assert_report.py
│
├── results/                 8,649 files — per-session oracle + history records
│   ├── P1_v4/               CANONICAL trained probes + steering directions
│   ├── P1_v4_layerbal/      balanced-accuracy backfill for older probes
│   ├── paper_stats/         posterior_probe.json  ← all Appendix C numbers
│   ├── ABL_dry_v1/  ABL_notally_v1/     ablation probe outputs
│   ├── BOS_ASSERT/  BOS_ASSERT_partial/ BoS scenario verification
│   └── P1_v1/ P1_v2/ P1_v3/ P2_v1/ base30/ TIMING_TEST/   superseded, kept
│
├── responses/               8,133 files — per-ROUND records (key: "rounds")
│                            NB: results/ has "session" + "history_records";
│                            responses/ has "rounds". Reading the wrong one
│                            is a KeyError.
│
├── steer/                   4,681 files — activation-steering replays
│   ├── P1_v4/               CANONICAL
│   └── P1_v3/               superseded
│
├── plots/                   422 files
│   ├── belief_layers/       Belief-Action.pdf/png  (paper figure)
│   ├── rq1_outcomes/        Decision_final        (paper figure)
│   └── <tag>/               per-campaign diagnostic plots
│
├── P3_postcot/              6,480 files — post-CoT campaign, seeds 0–39
├── ABL_dry_cot_v1/          2,430 files — dry-framing CoT ablation
├── ABL_dry_v1_snapshot/     4,860 files — responses/ + results/ copied out
│                            from behind the ABL_dry_v1 symlink so the
│                            dry ablation stays reproducible from this repo
├── ABL_notally_v1/          tally ablation (residuals excluded)
├── BOS_ASSERT/              BoS verification (residuals excluded)
├── SMOKE_{distill,bos,pennies}/  P3_smoke/   smoke tests (residuals excluded)
│
├── MANIFESTS/               file listings of the three off-repo trees
│   ├── residuals.manifest   16,262 entries
│   ├── ABL_dry_v1.manifest   9,720 entries
│   └── archive.manifest      9,515 entries
│
├── *.sbatch  (16)           Slurm job scripts
├── *.sh      (7)            launchers: launch_bos.sh, migrate_trust_v2.sh, …
├── run_all.py               main sweep driver; render(), ANSWER_FORMAT_*
├── run_steer.py             steering driver
├── smoke_test.py
├── *.log     (488)          job logs — full provenance of every run
│
├── AnonymousSubmission2027.tex     main paper (READ-ONLY by convention)
├── paper_*.tex                     section drafts and appendix fragments
├── DOCUMENTATION.txt               317 KB session log: every job ID, every bug
├── DATA_LOCATIONS.md               storage map (see correction below)
└── SUPPLEMENTARY_ASSEMBLY.md       supplementary working notes
```

---

## What is NOT in this repo

Four things are excluded. All still exist on CSCS.

### 1. Residual-stream activations — ~73.5 GB

Raw per-layer activations. **Only needed to re-train probes or re-run
`posterior_probe.py`.** Every trained output derived from them *is* committed
(`results/P1_v4/`, `results/paper_stats/posterior_probe.json`), so all reported
numbers can be quoted and inspected without them.

| Tree | Size | Files | Location | Purge risk |
|---|---|---|---|---|
| `residuals/` (canonical P1_v4) | 41 GB | 16,262 | `/capstor/scratch/cscs/joeunyook/logit-probe/residuals` | **YES — 30 day** |
| `ABL_dry_v1/residuals` | 13 GB | (of 9,720) | `/capstor/scratch/cscs/joeunyook/logit-probe/ABL_dry_v1` | **YES — 30 day** |
| `ABL_notally_v1/residuals` | 6.0 GB | | in home, `~/logit-probe/ABL_notally_v1/residuals` | no |
| `archive/*/residuals` | 13.3 GB | (of 9,515) | a0133 store | no |
| `BOS_ASSERT`, `SMOKE_*` residuals | ~225 MB | | in home | no |

`MANIFESTS/*.manifest` lists every excluded file with its byte size, so a
restore can be verified exactly.

### 2. Model weights — 132 GB (re-downloadable)

`meta-llama/Llama-3.3-70B-Instruct` was loaded from a local path,
`/capstor/scratch/cscs/joeunyook/models/Llama-3.3-70B-Instruct` (**purge risk**).
All other models load from the Hugging Face cache at
`HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache`.
Re-download from the identifiers in `core/model_loader.py`.

### 3. `archive/` — 14 GB of superseded and known-bad artifacts

At `/capstor/store/cscs/swissai/a0133/joeunyook/logit-probe-archive` (no purge).
Contains `trust_v1/` (pre-redesign Trust Game),
`dsllama70_broken_tokenizer/` (data collected through the broken tokenizer —
**invalid, never analyse**), `dsqwen7_greedy_old_format/`, and
`answer_parse_bug_2026-07-23/` (the source of the dsqwen7 P3 backfill).

### 4. A 5.74 GB core dump

`core_clariden-ln004_118572` — a crash artifact, not project data. Left in
place on CSCS, gitignored here.

---

## Restoring on CSCS

```bash
git clone git@github.com:joeunyook/Decision_Localize_raw.git logit-probe
cd logit-probe
```

Then, in order:

**1. Recreate the three symlinks.** They are gitignored, so a fresh clone has
no `residuals/`, `archive/`, or `ABL_dry_v1/` at all. Every script expects
those paths to exist:

```bash
ln -s /capstor/scratch/cscs/joeunyook/logit-probe/residuals   residuals
ln -s /capstor/scratch/cscs/joeunyook/logit-probe/ABL_dry_v1  ABL_dry_v1
ln -s /capstor/store/cscs/swissai/a0133/joeunyook/logit-probe-archive  archive
```

**2. Check the scratch trees still exist.** Scratch purges anything untouched
for 30 days. If `residuals/` and `ABL_dry_v1/` are gone, probes cannot be
re-trained and `analysis/posterior_probe.py` cannot run; everything else works.
Verify against the manifests:

```bash
find residuals -type f -printf '%s %p\n' | sort -k2 > /tmp/now.txt
diff <(grep -v '^#' MANIFESTS/residuals.manifest) /tmp/now.txt
```

**3. Re-download the 70B weights** if the scratch copy was purged, and point
`core/model_loader.py`'s `llama70` entry at the new location (it currently
holds an absolute path, not a HF identifier).

**4. Rebuild the environment** — see below.

**5. `ABL_dry_v1_snapshot/`** already holds the dry ablation's `responses/` and
`results/`. If the scratch tree is gone, either point `analysis/ablation_dry.py`
at the snapshot or copy it back into place.

---

## Restoring off CSCS

Everything except probe re-training works with no CSCS access at all:
all trained probes, all steering replays, all per-round records, all figures,
and all paper sources are committed. You can re-run every aggregation and
plotting script.

What you cannot do without the residuals: re-train P1 probes at a different
`C` or layer, re-run `posterior_probe.py` with different bin counts, or
recompute steering directions. Those need the 41 GB tree.

---

## Environment

Managed with micromamba. **Not version-pinned** — the env was rebuilt from an
unpinned recipe on 2026-07-24 after a scratch purge destroyed the original, so
library versions already differ from the earliest runs.

| Item | Location |
|---|---|
| `micromamba` binary | `~/bin/micromamba` |
| `MAMBA_ROOT_PREFIX` | `/capstor/store/cscs/swissai/a0133/joeunyook/micromamba` (set in `~/.bashrc` line 37) |
| env name | `ftdiag` |
| recipe | `~/FTdiagnosis/setup_env.sh` |
| rebuild script | `~/rebuild_ftdiag.sh` |

Run anything with:

```bash
~/bin/micromamba run -n ftdiag python analysis/<script>.py
```

Verified working versions at time of archiving: Python 3.11.15,
torch 2.13.0+cu130 (CUDA available), transformers 5.14.1.

> **`DATA_LOCATIONS.md` is stale on one row.** It states the micromamba env
> lives on `/capstor/scratch`. It was relocated to the a0133 store on
> 2026-07-24 precisely because scratch purged it. Trust this README.

---

## Regenerating each paper number

| Paper element | Command | Reads |
|---|---|---|
| Belief (P1) table column | `analysis/probe_train.py --tag P1_v4` | `residuals/` |
| Appendix C.2 R² figures | `analysis/posterior_probe.py` | `residuals/` |
| Appendix D steering | `analysis/steer_defense.py` | `steer/P1_v4/`, `responses/` |
| Result 3 (CoT amplifier) | `analysis/cot_amplification.py` | `P3_postcot/`, `responses/` |
| Dry-framing ablation | `analysis/ablation_dry.py` | `ABL_dry_v1_snapshot/`, `ABL_dry_cot_v1/` |
| Tally ablation | `analysis/tally_control.py` | `ABL_notally_v1/` |
| Per-timestep tables | `analysis/by_t.py --tag P1_v4` | `results/`, `responses/` |
| IPD exclusion justification | `analysis/ipd_label_diagnostics.py` | `results/`, `responses/` |
| Belief-Action figure | `analysis/plot_belief_layers.py` | `results/P1_v4*` |
| Appendix E CoT failures | `analysis/extract_cot_failures.py`, `cotfail_*.py` | `P3_postcot/responses/` |

Cached output already committed: `results/paper_stats/posterior_probe.json`
holds all 63 combos' `R2_full` / `R2_partial` / `R2_shuffled`.

---

## Known traps

Full detail in `DOCUMENTATION.txt`. The ones that will bite you:

1. **`archive/archive` is a symlink to its own parent.** Any symlink-following
   copy (`cp -aL`, `tar -h`, `rsync -L`) recurses forever. Always copy that
   tree without dereferencing.

2. **Scratch purges at 30 days.** It has already destroyed the conda env once
   (2026-07-24). `residuals/`, `ABL_dry_v1/` and the 70B weights all sit on
   scratch. Touch them or move them to a0133 store (10 TB, 0.5% used, no purge).

3. **Never trust the stored `post_valid` / `decision_act1` flags** in P3 data
   collected before 2026-07-21. Two parsing bugs affected them: a missing
   `\boxed{}` pattern, and first-match instead of last-match answer parsing
   (837 rounds, 4.3%, almost all one-directional). Always re-parse `cot_text`
   with the current `core/probes.ANSWER_RE` / `parse_answer_letter`.

4. **`probe_train.py` rewrites `results/<tag>/probe_summary.json` from only the
   combos in its current invocation.** A per-model loop leaves a summary with
   one entry. Run `analysis/rebuild_probe_summary.py --tag P1_v4` after any
   partial run. Per-combo `probe_*.json` files are always complete.

5. **`run_sweep_100seeds.sbatch` and `run_steer.sbatch` default to the stale
   `TAG=P1_v3`.** Always pass `TAG=` explicitly or you will write new results
   into the frozen v1 snapshot.

6. **`--skip-existing` checks only file existence and JSON-parseability, not
   correctness.** Resubmitting over corrupted data silently preserves it.
   Archive first, then relaunch.

7. **DeepSeek-R1-Distill-Llama-70B ships a tokenizer with no pre-tokenizer.**
   `core/model_loader.py` transplants one from Llama-3.3-70B-Instruct at load
   time. Without that patch every prompt is a run-on string and all results for
   that model are invalid.

8. **`/tmp` is not shared between login and compute nodes.** An
   `sbatch --output=/tmp/...` job silently produces no log.
