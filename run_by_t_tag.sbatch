#!/bin/bash
# =============================================================================
# logit-probe — generic by_t.py t-stratified aggregation, parameterized by
# TAG + RESID_DIR + RESP_DIR so it works for any sweep root (e.g. ABL_dry_v1),
# not just the canonical residuals/responses dirs.
# =============================================================================
# Usage:
#   sbatch --job-name=byt-abldry \
#          --export=ALL,TAG=ABL_dry_v1,RESID_DIR=ABL_dry_v1/residuals,RESP_DIR=ABL_dry_v1/responses \
#          run_by_t_tag.sbatch
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/users/joeunyook/logit-probe/byt_%x_%j.log

set -euo pipefail
cd /users/joeunyook/logit-probe

TAG=${TAG:-P1_v4}
RESID_DIR=${RESID_DIR:-residuals}
RESP_DIR=${RESP_DIR:-responses}

echo "============================================="
echo "by_t.py aggregation — tag=${TAG} residuals-dir=${RESID_DIR} responses-dir=${RESP_DIR}"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Started : $(date)"
echo "============================================="

~/bin/micromamba run -n ftdiag python analysis/by_t.py \
    --tag "${TAG}" \
    --residuals-dir "${RESID_DIR}" \
    --responses-dir "${RESP_DIR}"

echo ""
echo "============================================="
echo "Finished : $(date)"
echo "============================================="
