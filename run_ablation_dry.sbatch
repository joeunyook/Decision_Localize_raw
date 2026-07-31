#!/bin/bash
# =============================================================================
# logit-probe — prompt-framing ablation: "dry" (abstract/statistical) tone
# vs the canonical narrative-game prompt, pre-CoT only (--no-post, cheap).
# =============================================================================
# Same information content as the canonical forced-choice prompt (same payoffs,
# same hidden-type rates, same computed history tally) -- only the surrounding
# narrative flavor text is rewritten as a dry probability/decision-theory
# problem statement. See prompts/{env}_forced_dry.txt. Writes to a SEPARATE
# out-root (ABL_dry_v1, symlinked to scratch) and a separate results/plots tag
# (ABL_dry_v1) -- never touches P1_v4.
#
# Usage (per model, per env-shard):
#   sbatch --account=a0133 --qos=normal --job-name=abldry-<m>-<shard> \
#       --export=ALL,MODEL=<m>,ENVS=bandit:monty:ipd,SEED_FROM=0,SEED_TO=29 \
#       run_ablation_dry.sbatch
#   (add --gpus-per-node=2 for llama70 / dsllama70)
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/users/joeunyook/logit-probe/abldry_%x_%j.log

set -euo pipefail
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
export HF_HUB_OFFLINE=1

cd /users/joeunyook/logit-probe

ENVS=${ENVS//:/ }
SEED_FROM=${SEED_FROM:-0}
SEED_TO=${SEED_TO:-29}
TAG=ABL_dry_v1
OUT_ROOT=ABL_dry_v1

echo "============================================="
echo "logit-probe DRY-TONE ablation sweep — model=${MODEL} envs=${ENVS} seeds=${SEED_FROM}-${SEED_TO}"
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
    --forced-variant dry \
    --out-root "${OUT_ROOT}" \
    --skip-existing

echo ""
echo "--- sweep shard done for ${MODEL} (envs: ${ENVS}) ---"
echo "Residuals so far: $(ls ${OUT_ROOT}/residuals/${MODEL}_*.npy 2>/dev/null | wc -l) files"
echo "============================================="
echo "Finished : $(date)"
echo "============================================="
