#!/bin/bash
# =============================================================================
# GPU SMOKE TEST for the `pennies` scenario (Matching Pennies).
# =============================================================================
# Gate before committing ~15 twelve-hour production jobs. Runs the FULL pipeline
# on a tiny slice -- pre-CoT probe + residual capture + post-CoT generation
# together (neither --no-post nor --no-residuals) -- so it exercises every code
# path the production sweep will use: both prompt templates, the {answer_format}
# placeholder, the label-swap mapping, residual saving, generation and parsing.
#
# ISOLATION: writes ONLY to SMOKE_pennies/. Cannot touch results/, responses/,
# residuals/, results/P1_v4/, P3_postcot/, ABL_dry_cot_v1/ or any running job.
#
# PACKING: the node has 4x GH200 and the pipeline uses one GPU per process, so
# the models run concurrently, one per GPU.
#
# Usage (debug partition allows ONE job at a time per user):
#   sbatch --account=a0133 --qos=normal run_smoke_pennies.sbatch
#
# NOTE: do NOT name the model-group variable GROUPS -- bash pre-sets that to the
# caller's Unix group IDs, so ${GROUPS:-default} silently ignores the default
# and the models list becomes a numeric gid.
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --time=01:30:00
#SBATCH --job-name=smoke-pennies
#SBATCH --output=/users/joeunyook/logit-probe/smoke_pennies_%j.log

set -euo pipefail
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
export HF_HUB_OFFLINE=1

cd /users/joeunyook/logit-probe

export OUT_ROOT=${OUT_ROOT:-SMOKE_pennies}   # exported: the diagnostic block reads it from the environment
SEEDS=${SEEDS:-"0 1"}
ROUNDS=${ROUNDS:-5}
MODEL_GROUPS=${MODEL_GROUPS:-"gemma2b phi qwen gemma31"}

for M in ${MODEL_GROUPS}; do
    grep -q "\"${M}\"" core/model_loader.py || { echo "PREFLIGHT FAIL: unknown model '${M}'"; exit 1; }
done
python3 -c "
import sys; sys.path.insert(0,'.')
from envs import ENVS
assert 'pennies' in ENVS, 'pennies not registered'
print('preflight ok: pennies registered, models =', '${MODEL_GROUPS}')
"

echo "============================================="
echo "pennies GPU smoke -- models=${MODEL_GROUPS} seeds=${SEEDS} rounds=${ROUNDS}"
echo "out-root=${OUT_ROOT}   (full pipeline: pre-CoT + residuals + post-CoT)"
echo "Job ID : ${SLURM_JOB_ID}   Node: ${SLURM_NODELIST}   Started: $(date)"
echo "============================================="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

mkdir -p "${OUT_ROOT}"

gpu=0
pids=(); labels=()
for M in ${MODEL_GROUPS}; do
    (
        echo "[gpu ${gpu}] start ${M} $(date +%T)"
        CUDA_VISIBLE_DEVICES=${gpu} \
        ~/bin/micromamba run -n ftdiag python run_all.py \
            --models "${M}" --envs pennies \
            --seeds ${SEEDS} --rounds ${ROUNDS} \
            --out-root "${OUT_ROOT}" \
            > "${OUT_ROOT}/run_${M}.log" 2>&1
        echo "[gpu ${gpu}] done  ${M} $(date +%T)"
    ) &
    pids+=($!); labels+=("${M}"); gpu=$((gpu + 1))
done

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "[ok]   ${labels[$i]}"
    else echo "[FAIL] ${labels[$i]} -- see ${OUT_ROOT}/run_${labels[$i]}.log"; fail=1; fi
done

echo ""
for M in ${MODEL_GROUPS}; do
    echo "### ${M} (tail)"; tail -n 8 "${OUT_ROOT}/run_${M}.log" 2>/dev/null || echo "(no log)"
done

if [ "${fail}" -ne 0 ]; then echo "!!! a worker failed -- skipping diagnostics"; exit 1; fi

echo ""
echo "============================================="
echo "DIAGNOSTIC"
echo "============================================="
~/bin/micromamba run -n ftdiag python - <<'PYEOF'
import glob, json, os, re
import numpy as np
ROOT = os.environ.get("OUT_ROOT", "SMOKE_pennies")
ANSWER_RE = re.compile(r"Answer\s*:\s*([AB])\b|\\boxed\{([AB])\}")
ok = True
files = sorted(glob.glob(f"{ROOT}/responses/*_pennies_seed*.json"))
print(f"sessions written: {len(files)}")
if not files:
    raise SystemExit("FAIL: no sessions written")
for f in files:
    d = json.load(open(f))
    rs = d["rounds"]
    gaps = [r["delta_pre_sem"] for r in rs if r["delta_pre_sem"] is not None]
    finite = all(np.isfinite(g) for g in gaps)
    valid = sum(1 for r in rs if r.get("post_valid"))
    art = sum((r.get("cot_text") or "").count(c) for r in rs for c in ("Ġ", "Ċ"))
    orc = [r["oracle_p_act1"] for r in rs]
    stem = os.path.basename(f)[:-5]
    print(f"  {stem:26} rounds={len(rs)} preCoT_finite={finite} "
          f"post_valid={valid}/{len(rs)} decode_artifacts={art} "
          f"oracle=[{min(orc):.3f},{max(orc):.3f}]")
    if not finite or art:
        ok = False
    npy = f"{ROOT}/residuals/{stem}.npy"
    if os.path.exists(npy):
        print(f"  {'':26} residuals {np.load(npy).shape}")
    else:
        print(f"  {'':26} MISSING residuals"); ok = False
txt = (json.load(open(files[0]))["rounds"][-1].get("cot_text") or "")
print("\nsample CoT tail:", repr(txt[-220:]))
print("\nVERDICT:", "PASS -- safe to launch production" if ok else "CHECK MANUALLY")
PYEOF

echo ""
echo "Finished : $(date)"
