#!/bin/bash
# =============================================================================
# logit-probe — prompt-framing ablation, PHASE 2: dry-tone CoT generation.
# =============================================================================
# Same dry-tone rewrite as run_ablation_dry.sbatch (Phase 1, pre-CoT only),
# but WITH post-CoT generation on, using the new prompts/{env}_cot_dry.txt
# templates (spliced from the already-approved {env}_forced_dry.txt body +
# {env}_cot.txt's reasoning footer, narrative terms substituted to match).
# Writes to its OWN root ABL_dry_cot_v1 (NOT Phase 1's ABL_dry_v1). Reusing the
# Phase-1 root is a no-op because session_done() only checks file existence, so
# --skip-existing would skip every already-present pre-CoT seed and add no CoT.
# Seed-aligned with narrative P3_postcot (0-14 subset of 0-39) for direct compare.
#
# Seeds default to 0-14 (15 seeds -- judged sufficient for this ablation,
# half of Phase 1's 0-29 range) and remain a strict subset of P3_postcot's
# 0-39, so every round here is directly comparable, seed-for-seed, to the
# existing narrative-tone P3_postcot CoT data.
#
# Usage (per model, per env-shard -- mirrors run_post.sbatch's pattern):
#   sbatch --account=a0133 --qos=normal --job-name=abldry-cot-<m>-<shard> \
#       --export=ALL,MODEL=<m>,ENVS=bandit:monty:ipd run_ablation_dry_cot.sbatch
#   (add --gpus-per-node=2 for llama70 / dsllama70, and shard envs across
#   multiple jobs the same way run_post.sbatch's llama70/dsllama70 jobs did,
#   to stay inside the 12h wall)
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/users/joeunyook/logit-probe/abldrycot_%x_%j.log

set -euo pipefail
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
export HF_HUB_OFFLINE=1

cd /users/joeunyook/logit-probe

ENVS=${ENVS//:/ }
SEED_FROM=${SEED_FROM:-0}
SEED_TO=${SEED_TO:-14}
# Dry-CoT writes to its OWN root, separate from Phase-1's pre-CoT ABL_dry_v1 --
# exactly as baseline post-CoT (P3_postcot) is separate from pre-CoT (results/).
# Reusing ABL_dry_v1 was a no-op: session_done() only checks file existence, so
# --skip-existing skipped every Phase-1 seed and added zero CoT (caught by the
# 2026-07-24 debug sanity job). run_all.py's save_json auto-creates the subdirs.
OUT_ROOT=${OUT_ROOT:-ABL_dry_cot_v1}

echo "============================================="
echo "logit-probe DRY-TONE ablation, PHASE 2 (post-CoT) — model=${MODEL} envs=${ENVS} seeds=${SEED_FROM}-${SEED_TO}"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Started : $(date)"
echo "============================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

~/bin/micromamba run -n ftdiag python run_all.py \
    --models "${MODEL}" \
    --envs ${ENVS} \
    --seeds $(seq "${SEED_FROM}" "${SEED_TO}") --rounds 10 \
    --no-residuals \
    --forced-variant dry \
    --out-root "${OUT_ROOT}" \
    --skip-existing

echo ""
echo "Sessions : $(ls ${OUT_ROOT}/responses/${MODEL}_*.json 2>/dev/null | wc -l) files for ${MODEL} in ${OUT_ROOT}/"
echo "Finished : $(date)"
