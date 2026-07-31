#!/bin/bash
# =============================================================================
# logit-probe — P4 steering / rescue experiment, one model per node
# =============================================================================
# Directions must exist first:
#   micromamba run -n ftdiag python analysis/probe_train.py \
#       --residuals-dir residuals --tag P1_v3 --weights-only
#
# NOTE: the login environment exports SBATCH_ACCOUNT=a0107 (dead account), which
# overrides the directives below -- always pass --account=a0133 --qos=normal on
# the command line (CLI beats env beats file).
#
# Usage (small models, 1 GPU):
#   cd ~/logit-probe
#   sbatch --account=a0133 --qos=normal --job-name=steer-llama   --export=ALL,MODEL=llama   run_steer.sbatch
#   sbatch --account=a0133 --qos=normal --job-name=steer-phi     --export=ALL,MODEL=phi     run_steer.sbatch
#   sbatch --account=a0133 --qos=normal --job-name=steer-qwen    --export=ALL,MODEL=qwen    run_steer.sbatch
#   sbatch --account=a0133 --qos=normal --job-name=steer-qwen27  --export=ALL,MODEL=qwen27  run_steer.sbatch
#   sbatch --account=a0133 --qos=normal --job-name=steer-gemma31 --export=ALL,MODEL=gemma31 run_steer.sbatch
# 70B models (after their sweep + probe training):
#   sbatch --account=a0133 --qos=normal --gpus-per-node=2 --job-name=steer-llama70 --export=ALL,MODEL=llama70 run_steer.sbatch
#
# Optional env overrides: SEED_FROM/SEED_TO (default 0-39), TAG (default P1_v3),
# ENVS (default all five).
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/users/joeunyook/logit-probe/steer_%x_%j.log

set -euo pipefail
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
export HF_HUB_OFFLINE=1

cd /users/joeunyook/logit-probe

SEED_FROM=${SEED_FROM:-0}
SEED_TO=${SEED_TO:-39}
TAG=${TAG:-P1_v3}
# multi-env values use ':' (--export splits on commas), e.g.
# ENVS=trust_game:machine:chicken
ENVS=${ENVS:-"monty bandit stag_hunt ipd trust_game"}
ENVS=${ENVS//:/ }

echo "============================================="
echo "logit-probe steering — model=${MODEL} envs=${ENVS} seeds=${SEED_FROM}-${SEED_TO} tag=${TAG}"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Started : $(date)"
echo "============================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

~/bin/micromamba run -n ftdiag python run_steer.py \
    --models "${MODEL}" \
    --envs ${ENVS} \
    --seeds $(seq "${SEED_FROM}" "${SEED_TO}") --rounds 10 \
    --tag "${TAG}" \
    --skip-existing

echo ""
echo "============================================="
echo "Finished : $(date)"
echo "Sessions : $(ls steer/${TAG}/${MODEL}_*.json 2>/dev/null | wc -l) files for ${MODEL}"
echo "============================================="
