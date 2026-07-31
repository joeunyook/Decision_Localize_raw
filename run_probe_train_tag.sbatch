#!/bin/bash
# =============================================================================
# logit-probe — generic P1 probe training, one model per job, parameterized by
# TAG + RESID_DIR (so it works for any residuals root, e.g. the ABL_dry_v1
# ablation sweep, not just the canonical residuals/ dir).
# =============================================================================
# Usage:
#   sbatch --job-name=probe-abldry-gemma2b \
#          --export=ALL,MODEL=gemma2b,TAG=ABL_dry_v1,RESID_DIR=ABL_dry_v1/residuals,ENVS="monty bandit stag_hunt ipd trust_game machine chicken" \
#          run_probe_train_tag.sbatch
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/users/joeunyook/logit-probe/probetrain_%x_%j.log

set -euo pipefail
cd /users/joeunyook/logit-probe

TAG=${TAG:-P1_v4}
RESID_DIR=${RESID_DIR:-residuals}
ENVS=${ENVS:-"monty bandit stag_hunt ipd trust_game machine chicken"}

echo "============================================="
echo "P1 probe training — model=${MODEL} tag=${TAG} residuals-dir=${RESID_DIR}"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Started : $(date)"
echo "============================================="

~/bin/micromamba run -n ftdiag python analysis/probe_train.py \
    --models "${MODEL}" \
    --envs ${ENVS} \
    --residuals-dir "${RESID_DIR}" \
    --tag "${TAG}"

echo ""
echo "============================================="
echo "Finished : $(date)"
echo "Plots    : $(ls plots/${TAG}/probe_${MODEL}_*.png 2>/dev/null | wc -l)"
echo "============================================="
