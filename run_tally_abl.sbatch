#!/bin/bash
# Tally-ablation control: same as canonical sweep but with # Tally line
# stripped from every history prompt. Tests whether linear decodability
# depends on the explicit summary or the observation sequence.
# Models: gemma2b, llama, phi, qwen (small/fast only)
# Usage:
#   sbatch --account=a0133 --qos=normal --job-name=tallyabl-<m>-<shard> \
#       --export=ALL,MODEL=<m>,ENVS=bandit:monty:trust_game,SEED_FROM=0,SEED_TO=29 \
#       run_tally_abl.sbatch
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/users/joeunyook/logit-probe/tallyabl_%x_%j.log

set -euo pipefail
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
export HF_HUB_OFFLINE=1

cd /users/joeunyook/logit-probe

ENVS=${ENVS//:/ }
SEED_FROM=${SEED_FROM:-0}
SEED_TO=${SEED_TO:-29}
OUT_ROOT=ABL_notally_v1

echo "============================================="
echo "logit-probe TALLY-ABL sweep — model=${MODEL} envs=${ENVS} seeds=${SEED_FROM}-${SEED_TO}"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Started : $(date)"
echo "============================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

~/bin/micromamba run -n ftdiag python run_all.py \
    --models "${MODEL}" \
    --envs ${ENVS} \
    --seeds $(seq "${SEED_FROM}" "${SEED_TO}") \
    --rounds 10 --no-post --no-tally \
    --out-root "${OUT_ROOT}" \
    --skip-existing

echo "--- done: ${MODEL} (${ENVS}) ---"
echo "Residuals: $(ls ${OUT_ROOT}/residuals/${MODEL}_*.npy 2>/dev/null | wc -l) files"
echo "Finished : $(date)"
