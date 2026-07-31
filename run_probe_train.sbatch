#!/bin/bash
# =============================================================================
# logit-probe — P1 probe training, one model per job (submit 3x, one per model)
# =============================================================================
# Usage:
#   cd ~/logit-probe
#   sbatch --job-name=probe-llama --export=MODEL=llama run_probe_train.sbatch
#   sbatch --job-name=probe-phi   --export=MODEL=phi   run_probe_train.sbatch
#   sbatch --job-name=probe-qwen  --export=MODEL=qwen  run_probe_train.sbatch
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/users/joeunyook/logit-probe/probe_train_%x_%j.log

set -euo pipefail
cd /users/joeunyook/logit-probe

echo "============================================="
echo "P1 probe training — model=${MODEL}"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Started : $(date)"
echo "============================================="

~/bin/micromamba run -n ftdiag python analysis/probe_train.py \
    --models "${MODEL}" \
    --envs monty bandit stag_hunt ipd trust_game \
    --residuals-dir residuals \
    --tag P1_v2

echo ""
echo "============================================="
echo "Finished : $(date)"
echo "Plots    : $(ls plots/P1_v2/probe_${MODEL}_*.png 2>/dev/null | wc -l)"
echo "============================================="
