#!/bin/bash
# =============================================================================
# logit-probe — full pre-CoT sweep (3 models x 5 envs x 40 seeds), 1x GH200
# =============================================================================
# Usage:
#   cd ~/logit-probe
#   sbatch run_clariden.sbatch
# =============================================================================
#SBATCH --job-name=logit-probe-sweep
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --time=08:00:00
#SBATCH --output=/users/joeunyook/logit-probe/run_pre_40seeds_%j.log

set -euo pipefail
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
export HF_HUB_OFFLINE=1

cd /users/joeunyook/logit-probe

echo "============================================="
echo "logit-probe full sweep — CSCS Clariden (1x GH200)"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Started : $(date)"
echo "============================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

~/bin/micromamba run -n ftdiag python run_all.py \
    --models llama phi qwen \
    --envs monty bandit stag_hunt ipd trust_game \
    --seeds $(seq 0 39) --rounds 10 --no-post

echo ""
echo "============================================="
echo "Finished : $(date)"
echo "Results  : residuals/*.npy count -> $(ls residuals/*.npy 2>/dev/null | wc -l)"
echo "============================================="
