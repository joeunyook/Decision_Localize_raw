#!/bin/bash
# =============================================================================
# logit-probe — 100-seed expansion sweep, one model per node (submit 5x)
# =============================================================================
# Existing 3 small models already have seeds 0-39 on disk -> run 40-99 only.
# New midscale models (qwen27, gemma31) run the full 0-99.
#
# Usage:
#   cd ~/logit-probe
#   sbatch --job-name=sweep-llama   --export=ALL,MODEL=llama,SEED_FROM=40,SEED_TO=99 run_sweep_100seeds.sbatch
#   sbatch --job-name=sweep-phi     --export=ALL,MODEL=phi,SEED_FROM=40,SEED_TO=99   run_sweep_100seeds.sbatch
#   sbatch --job-name=sweep-qwen    --export=ALL,MODEL=qwen,SEED_FROM=40,SEED_TO=99  run_sweep_100seeds.sbatch
#   sbatch --job-name=sweep-qwen27  --export=ALL,MODEL=qwen27,SEED_FROM=0,SEED_TO=99 run_sweep_100seeds.sbatch
#   sbatch --job-name=sweep-gemma31 --export=ALL,MODEL=gemma31,SEED_FROM=0,SEED_TO=99 run_sweep_100seeds.sbatch
#
# NOTE: the login environment exports SBATCH_ACCOUNT=a0107 (dead account), which
# overrides the directives below -- always pass --account=a0133 --qos=normal on
# the command line.
#
# 2026-07-13 additions (download via download_new_models.sh first):
#   sbatch --account=a0133 --qos=normal --job-name=sweep-gemma2b --export=ALL,MODEL=gemma2b,SEED_FROM=0,SEED_TO=99 run_sweep_100seeds.sbatch
#   sbatch --account=a0133 --qos=normal --job-name=sweep-dsqwen7 --export=ALL,MODEL=dsqwen7,SEED_FROM=0,SEED_TO=99 run_sweep_100seeds.sbatch
#   sbatch --account=a0133 --qos=normal --gpus-per-node=2 --job-name=sweep-llama70   --export=ALL,MODEL=llama70,SEED_FROM=0,SEED_TO=99   run_sweep_100seeds.sbatch
#   sbatch --account=a0133 --qos=normal --gpus-per-node=2 --job-name=sweep-dsllama70 --export=ALL,MODEL=dsllama70,SEED_FROM=0,SEED_TO=99 run_sweep_100seeds.sbatch
#
# Each job ends by training the P1 probes for its model (tag P1_v3) so the
# per-combo JSONs + plots appear as soon as the model's sweep finishes.
# NOTE: results/P1_v3/probe_summary.json is overwritten by whichever job
# finishes last — rebuild the merged 25-combo summary from the per-combo
# JSONs afterwards (they have unique names and never collide).
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/users/joeunyook/logit-probe/sweep100_%x_%j.log

set -euo pipefail
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
export HF_HUB_OFFLINE=1

cd /users/joeunyook/logit-probe

# optional env overrides (defaults preserve the original 5-env P1_v3 behavior).
# Multi-env values use ':' as separator because --export splits on commas,
# e.g. ENVS=trust_game:machine:chicken.
# trust_game v2 + machine + chicken run (2026-07-19):
#   --export=ALL,MODEL=<m>,SEED_FROM=0,SEED_TO=99,ENVS=trust_game:machine:chicken,TAG=P1_v4
# and run ./migrate_trust_v2.sh FIRST, or --skip-existing silently keeps
# v1 trust_game data (machine/chicken have no old artifacts and just run).
ENVS=${ENVS:-"monty bandit stag_hunt ipd trust_game"}
ENVS=${ENVS//:/ }
TAG=${TAG:-P1_v3}

echo "============================================="
echo "logit-probe 100-seed sweep — model=${MODEL} envs=${ENVS} seeds=${SEED_FROM}-${SEED_TO} tag=${TAG}"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Started : $(date)"
echo "============================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

~/bin/micromamba run -n ftdiag python run_all.py \
    --models "${MODEL}" \
    --envs ${ENVS} \
    --seeds $(seq "${SEED_FROM}" "${SEED_TO}") --rounds 10 --no-post \
    --skip-existing

echo ""
echo "--- sweep done, training P1 probes for ${MODEL} (tag ${TAG}) ---"
~/bin/micromamba run -n ftdiag python analysis/probe_train.py \
    --models "${MODEL}" \
    --envs ${ENVS} \
    --residuals-dir residuals \
    --tag "${TAG}" --save-weights

echo ""
echo "============================================="
echo "Finished : $(date)"
echo "Residuals: $(ls residuals/${MODEL}_*.npy 2>/dev/null | wc -l) files for ${MODEL}"
echo "Plots    : $(ls plots/${TAG}/probe_${MODEL}_*.png 2>/dev/null | wc -l)"
echo "============================================="
