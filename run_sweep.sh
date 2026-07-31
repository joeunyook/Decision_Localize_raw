#!/bin/bash
set -euo pipefail
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
export HF_HUB_OFFLINE=1

cd /users/joeunyook/logit-probe

echo "============================================="
echo "logit-probe full sweep — CSCS Clariden (1x GH200)"
echo "Job ID  : ${SLURM_JOB_ID:-n/a}"
echo "Node    : $(hostname)"
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
