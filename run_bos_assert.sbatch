#!/bin/bash
# =============================================================================
# BoS TREND ASSERTION -- small, fast, fully isolated.
# =============================================================================
# Question: does Bach-or-Stravinsky reproduce the reactive-opponent signature
# (belief high, belief-to-action mapping degraded) that Stag Hunt shows, so that
# BoS can replace IPD in the game-theoretic family?
#
# Scope: PRE-CoT ONLY (--no-post). No generation.
#
# ISOLATION (nothing here can touch a running campaign):
#   raw sessions  -> BOS_ASSERT/{results,responses,residuals}/   (--out-root)
#   probes/plots  -> results/BOS_ASSERT/ , plots/BOS_ASSERT/     (--tag)
# The canonical results/, responses/, residuals/, results/P1_v4/, P3_postcot/
# and ABL_dry_cot_v1/ trees are never opened for writing.
#
# PACKING: Slurm hands us the whole node (4x GH200) but the pipeline uses one
# GPU per process. MODEL_GROUPS assigns models to GPUs: space separates GPUs,
# comma separates models that share a GPU sequentially. Put the two smallest
# models together so each large model gets a GPU to itself.
#
# RESUME: --skip-existing reuses sessions already in BOS_ASSERT/, so raising
# SEEDS only runs the new seeds. Safe because the env, prompts and code are
# unchanged and sessions are deterministic in (env, seed).
#
# Usage (debug partition allows ONE job at a time per user):
#   sbatch --account=a0133 --qos=normal run_bos_assert.sbatch
#
# NOTE: the login environment exports SBATCH_ACCOUNT=a0107 (dead account, its
# association carries the `slowdown` QOS at priority 10), which overrides the
# directives below -- always pass --account=a0133 --qos=normal explicitly.
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --time=01:30:00
#SBATCH --job-name=bos-assert
#SBATCH --output=/users/joeunyook/logit-probe/bos_assert_%j.log

set -euo pipefail
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
export HF_HUB_OFFLINE=1

cd /users/joeunyook/logit-probe

OUT_ROOT=${OUT_ROOT:-BOS_ASSERT}
TAG=${TAG:-BOS_ASSERT}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
# one entry per GPU; comma-separated models inside an entry share that GPU
MODEL_GROUPS=${MODEL_GROUPS:-"qwen,llama phi gemma31 qwen27"}
ALL_MODELS=$(echo "${MODEL_GROUPS}" | tr ' ,' '\n\n' | sort -u | tr '\n' ' ')

# PREFLIGHT: every name must be a real key in the MODELS registry. Guards against
# an env var silently overriding MODEL_GROUPS -- the first version of this script
# used the name GROUPS, which bash pre-sets to the caller's Unix group IDs, so
# ${GROUPS:-default} never saw the default and one worker launched with
# `--models 1071`. Fail here, before the allocation is spent.
for M in ${ALL_MODELS}; do
    if ! grep -q "\"${M}\"" core/model_loader.py; then
        echo "PREFLIGHT FAIL: '${M}' is not a key in core/model_loader.py MODELS"
        echo "MODEL_GROUPS was: ${MODEL_GROUPS}"
        exit 1
    fi
done
echo "preflight ok: models = ${ALL_MODELS}"

echo "============================================="
echo "BoS trend assertion -- env=bos seeds=${SEEDS}"
echo "groups (gpu<-models): ${MODEL_GROUPS}"
echo "out-root=${OUT_ROOT}  tag=${TAG}  (pre-CoT only)"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Started : $(date)"
echo "============================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

mkdir -p "${OUT_ROOT}"

# --- one group per GPU, groups run concurrently, models within a group serially
gpu=0
pids=()
labels=()
for GROUP in ${MODEL_GROUPS}; do
    (
        for M in $(echo "${GROUP}" | tr ',' ' '); do
            echo "[gpu ${gpu}] start ${M} $(date +%T)"
            CUDA_VISIBLE_DEVICES=${gpu} \
            ~/bin/micromamba run -n ftdiag python run_all.py \
                --models "${M}" \
                --envs bos \
                --seeds ${SEEDS} --rounds 10 --no-post \
                --out-root "${OUT_ROOT}" \
                --skip-existing \
                > "${OUT_ROOT}/run_${M}.log" 2>&1
            echo "[gpu ${gpu}] done  ${M} $(date +%T)"
        done
    ) &
    pids+=($!)
    labels+=("${GROUP}")
    gpu=$((gpu + 1))
done

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[ok]   group ${labels[$i]} finished"
    else
        echo "[FAIL] group ${labels[$i]} exited non-zero -- see ${OUT_ROOT}/run_*.log"
        fail=1
    fi
done

echo ""
echo "--- per-model run logs (tail) ---"
for M in ${ALL_MODELS}; do
    echo "### ${M}"
    tail -n 6 "${OUT_ROOT}/run_${M}.log" 2>/dev/null || echo "(no log)"
done

if [ "${fail}" -ne 0 ]; then
    echo "!!! at least one worker failed; skipping probe training"
    exit 1
fi

echo ""
echo "--- sessions written ---"
for M in ${ALL_MODELS}; do
    echo "  ${M}: $(ls ${OUT_ROOT}/responses/${M}_bos_seed*.json 2>/dev/null | wc -l)"
done
echo ""

# --- belief probes (single invocation: probe_summary.json stays complete) -----
echo "--- training P1 probes (tag ${TAG}) ---"
~/bin/micromamba run -n ftdiag python analysis/probe_train.py \
    --models ${ALL_MODELS} \
    --envs bos \
    --residuals-dir "${OUT_ROOT}/residuals" \
    --tag "${TAG}" --save-weights

# --- the actual answer --------------------------------------------------------
echo ""
~/bin/micromamba run -n ftdiag python analysis/bos_assert_report.py \
    --out-root "${OUT_ROOT}" --tag "${TAG}" --models ${ALL_MODELS}

echo ""
echo "============================================="
echo "Finished : $(date)"
echo "============================================="
