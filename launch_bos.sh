#!/bin/bash
# =============================================================================
# Launch the full `bos` (Bach or Stravinsky / Battle of the Sexes) campaign.
# =============================================================================
# Every flag that has ever silently bitten this project is passed EXPLICITLY
# here, never inherited from an sbatch default:
#
#   ENVS=bos   -- run_sweep_100seeds.sbatch and run_steer.sbatch still default
#                 to the stale five-env list (monty bandit stag_hunt ipd
#                 trust_game). Inheriting that default is what gave dsllama70
#                 zero machine/chicken sessions and cost a full re-run.
#   TAG=P1_v4  -- run_steer.sbatch defaults to the OLD tag P1_v3.
#
# A preflight check runs first and ABORTS before submitting anything if the env
# is not correctly registered. --skip-existing (hardcoded in the sweep/post
# scripts) is safe here only because bos is brand new; the preflight also
# confirms no stale bos artifacts exist that could be silently skipped.
#
# Usage:
#   ./launch_bos.sh precot     # Probe 1 + Probe 2 (100 seeds) -- cheap, ~1h
#   ./launch_bos.sh cot        # Probe 3 (40 seeds, 2 shards of 20 per model)
#   ./launch_bos.sh probes     # OPTIONAL: precot already trains Probe 1
#   ./launch_bos.sh dry        # dry-tone prompt-framing ablation (pre-CoT)
# =============================================================================
set -euo pipefail
cd /users/joeunyook/logit-probe

MODE=${1:-}
# 12h walls with FEW, LARGE shards -- deliberately not many short jobs. Shards
# are sized against the WORST observed CoT throughput in this project
# (~100 s/round: dsllama70 x monty, 653 min for 40 seeds), not per-model
# averages (which range down to 15 s/round at qwen x stag_hunt). 20 seeds =
# 200 rounds ~= 5.6h worst case, i.e. ~2.1x margin inside a 12h wall.
# A straggler is free to resubmit: run_post.sbatch hardcodes --skip-existing.
WALL=${WALL:-12:00:00}
A="--account=a0133 --qos=normal --time=${WALL}"
SMALL="gemma2b llama phi qwen"
MID="qwen27 gemma31"
ONE_GPU="gemma2b llama phi qwen qwen27 gemma31 dsqwen7"
TWO_GPU="llama70 dsllama70"
COT_SHARDS="0:19 20:39"

# ---------------------------------------------------------------- preflight
echo "=== preflight ==="
~/bin/micromamba run -n ftdiag python - <<'PY'
import sys, numpy as np
sys.path.insert(0, "/users/joeunyook/logit-probe")
from pathlib import Path
from envs import ENVS, make_env

ok = True
def chk(label, cond):
    global ok
    print(f"  [{'OK ' if cond else 'FAIL'}] {label}")
    ok = ok and cond

chk("bos registered in envs.ENVS", "bos" in ENVS)
for kind in ("forced", "cot", "forced_dry", "cot_dry"):
    p = Path(f"/users/joeunyook/logit-probe/prompts/bos_{kind}.txt")
    chk(f"prompts/bos_{kind}.txt exists", p.exists())

# oracle must produce BOTH classes, and must not depend on the round counter
labels = np.zeros((40, 10), dtype=bool)
for s in range(40):
    env = make_env("bos", np.random.default_rng(s), 10, seed=s)
    rec = []
    for t in range(1, 11):
        ctx = env.new_round(t)
        labels[s, t - 1] = env.oracle_p_act1(rec) > 0.5
        _, r = env.outcome(t, ctx, env.behavior_action(t))
        rec.append(r)
chk(f"oracle spans both classes (act1={100*labels.mean():.0f}%)",
    0.15 < labels.mean() < 0.85)
maj = labels.mean(axis=0) > 0.5
clock = np.tile(maj, (40, 1))
base = max(labels.mean(), 1 - labels.mean())
gain = (clock == labels).mean() - base
chk(f"clock-only gain over base rate = {100*gain:+.1f}pp (must be < 10pp)", gain < 0.10)

import analysis.by_t as byt
chk("bos present in by_t.ENV_ORDER", "bos" in byt.ENV_ORDER)

sys.exit(0 if ok else 1)
PY
echo "preflight passed"
echo ""

# guard: --skip-existing is hardcoded downstream, so stale artifacts would be
# silently skipped rather than recomputed.
# NB: `|| true` is required -- under `set -euo pipefail` a glob with no matches
# makes ls exit non-zero and pipefail kills the whole script right here.
stale=$( { ls results/*_bos_seed*.json P3_postcot/results/*_bos_seed*.json 2>/dev/null || true; } | wc -l )
if [ "$stale" -ne 0 ] && [ "$MODE" != "probes" ]; then
    if [ "${ALLOW_EXISTING:-0}" = "1" ]; then
        echo "WARNING: $stale existing bos artifact(s) will be SKIPPED (--skip-existing):"
        { ls results/*_bos_seed*.json P3_postcot/results/*_bos_seed*.json 2>/dev/null || true; } | sed 's/^/         /'
        echo "         proceeding because ALLOW_EXISTING=1"
    else
        echo "ABORT: $stale existing bos artifacts found; --skip-existing would skip them."
        echo "       Re-run with ALLOW_EXISTING=1 if they are known-good, or move them aside."
        exit 1
    fi
fi

case "$MODE" in
# --------------------------------------------------------------- pre-CoT
precot)
  echo "=== Probe 1 + Probe 2: bos, 100 seeds, all 9 models ==="
  for m in $SMALL $MID; do
    sbatch $A --job-name=bos-pre-$m \
      --export=ALL,MODEL=$m,ENVS=bos,SEED_FROM=0,SEED_TO=99,TAG=P1_v4 \
      run_sweep_100seeds.sbatch
  done
  sbatch $A --job-name=bos-pre-dsqwen7 \
    --export=ALL,MODEL=dsqwen7,ENVS=bos,SEED_FROM=0,SEED_TO=99,TAG=P1_v4 \
    run_sweep_100seeds.sbatch
  for m in $TWO_GPU; do
    sbatch $A --gpus-per-node=2 --job-name=bos-pre-$m \
      --export=ALL,MODEL=$m,ENVS=bos,SEED_FROM=0,SEED_TO=99,TAG=P1_v4 \
      run_sweep_100seeds.sbatch
  done
  ;;

# ------------------------------------------------------------------- CoT
cot)
  echo "=== Probe 3: bos, 40 seeds, 2 shards of 20, ${WALL} wall ==="
  # Uniform 20-seed shards for EVERY model. Per-round CoT cost in this project
  # spans 15-100 s/round depending on how often a model loops to the token cap,
  # and that is not predictable from model size -- so every model is planned at
  # the worst observed rate rather than its own average. A straggler resubmits
  # cleanly: run_post.sbatch hardcodes --skip-existing, so nothing is redone.
  n=0
  for shard in $COT_SHARDS; do
    f=${shard%%:*}; t=${shard##*:}
    for m in $ONE_GPU; do
      sbatch $A --job-name=bos-cot-$m-$f \
        --export=ALL,MODEL=$m,ENVS=bos,SEED_FROM=$f,SEED_TO=$t run_post.sbatch
      n=$((n+1))
    done
    for m in $TWO_GPU; do
      sbatch $A --gpus-per-node=2 --job-name=bos-cot-$m-$f \
        --export=ALL,MODEL=$m,ENVS=bos,SEED_FROM=$f,SEED_TO=$t run_post.sbatch
      n=$((n+1))
    done
  done
  echo "submitted ${n} CoT jobs (9 models x 2 shards)"
  ;;

# ---------------------------------------------------------------- probes
probes)
  echo "=== OPTIONAL re-train of Probe 1 (the precot sweep already trains it) ==="
  for m in $SMALL $MID dsqwen7 $TWO_GPU; do
    sbatch $A --job-name=bos-probe-$m \
      --export=ALL,MODEL=$m,ENVS=bos,TAG=P1_v4 run_probe_train_tag.sbatch
  done
  ;;

# ------------------------------------------------------------- ablation
dry)
  echo "=== dry-tone ablation: bos, pre-CoT, seeds 0-29 ==="
  for m in $SMALL $MID dsqwen7; do
    sbatch $A --job-name=bos-dry-$m \
      --export=ALL,MODEL=$m,ENVS=bos,SEED_FROM=0,SEED_TO=29 run_ablation_dry.sbatch
  done
  for m in $TWO_GPU; do
    sbatch $A --gpus-per-node=2 --job-name=bos-dry-$m \
      --export=ALL,MODEL=$m,ENVS=bos,SEED_FROM=0,SEED_TO=29 run_ablation_dry.sbatch
  done
  ;;

*)
  echo "usage: $0 {precot|cot|probes|dry}"; exit 1;;
esac

echo ""
squeue -u joeunyook --name="$(squeue -u joeunyook -h -o %j | grep '^bos-' | paste -sd, -)" 2>/dev/null || squeue -u joeunyook
