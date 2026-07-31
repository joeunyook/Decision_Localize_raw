#!/bin/bash
# =============================================================================
# logit-probe — P3 post-CoT campaign, one model per node
# =============================================================================
# Runs run_all.py WITH CoT generation (no --no-post) at reduced scale
# (seeds 0-39) into the separate root P3_postcot/ so the canonical pre-CoT
# data in results/ + responses/ + residuals/ is never touched.
# Residual capture is OFF (--no-residuals): the P3 deliverable is the
# post-CoT decision/logit table, which needs only the JSONs (~MBs).
#
# NOTE: the login environment exports SBATCH_ACCOUNT=a0107 (dead account) --
# always pass --account=a0133 --qos=normal on the command line.
#
# Usage (after a smoke job has validated the path wiring):
#   cd ~/logit-probe
#   sbatch --account=a0133 --qos=normal --job-name=post3-gemma2b --export=ALL,MODEL=gemma2b run_post.sbatch
#   ... one per model; 70B models add --gpus-per-node=2 and (recommended)
#   split envs across two jobs via ENVS=a:b:c to stay inside the 12h wall:
#   sbatch --account=a0133 --qos=normal --gpus-per-node=2 --job-name=post3-llama70a \
#          --export=ALL,MODEL=llama70,ENVS=bandit:monty:ipd:stag_hunt run_post.sbatch
#   sbatch --account=a0133 --qos=normal --gpus-per-node=2 --job-name=post3-llama70b \
#          --export=ALL,MODEL=llama70,ENVS=trust_game:machine:chicken run_post.sbatch
#
# Resume after a cut/timeout: resubmit the same command; --skip-existing
# checks P3_postcot/ (via --out-root) at session granularity.
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/users/joeunyook/logit-probe/post_%x_%j.log

set -euo pipefail
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
export HF_HUB_OFFLINE=1

cd /users/joeunyook/logit-probe

SEED_FROM=${SEED_FROM:-0}
SEED_TO=${SEED_TO:-39}
OUT_ROOT=${OUT_ROOT:-P3_postcot}
ENVS=${ENVS:-"monty bandit stag_hunt ipd trust_game machine chicken"}
ENVS=${ENVS//:/ }
ROUNDS=${ROUNDS:-10}

echo "============================================="
echo "logit-probe P3 post-CoT — model=${MODEL} envs=${ENVS} seeds=${SEED_FROM}-${SEED_TO} out=${OUT_ROOT}"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Started : $(date)"
echo "============================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

~/bin/micromamba run -n ftdiag python run_all.py \
    --models "${MODEL}" \
    --envs ${ENVS} \
    --seeds $(seq "${SEED_FROM}" "${SEED_TO}") --rounds "${ROUNDS}" \
    --no-residuals \
    --out-root "${OUT_ROOT}" \
    --skip-existing

echo ""
echo "Sessions : $(ls ${OUT_ROOT}/responses/${MODEL}_*.json 2>/dev/null | wc -l) files for ${MODEL} in ${OUT_ROOT}/"
echo "Finished : $(date)"
