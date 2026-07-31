#!/bin/bash
# =============================================================================
# logit-probe — GPU smoke test for the R1-distill fixes (tokenizer patch,
# temperature=0.6 sampling, \boxed{} answer format).
#
# Runs the FULL run_all.py pipeline (pre-CoT forced-choice probe + residual
# capture + post-CoT generation, i.e. no --no-post / --no-residuals) for a
# small slice, then trains probes on that slice -- so this exercises all
# three measurement instruments: belief representation (residuals -> probe),
# pre-CoT action mapping, and post-CoT generation. Writes to a dedicated
# SMOKE_distill/ root, never touching real campaign data.
#
# Usage:
#   sbatch --account=a0133 --qos=normal --job-name=smoke-dsqwen7 \
#       --export=ALL,MODEL=dsqwen7 smoke_distill.sbatch
#   sbatch --account=a0133 --qos=normal --gpus-per-node=2 --job-name=smoke-dsllama70 \
#       --export=ALL,MODEL=dsllama70 smoke_distill.sbatch
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --time=02:00:00
#SBATCH --output=/users/joeunyook/logit-probe/smoke_distill_%x_%j.log

set -euo pipefail
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
export HF_HUB_OFFLINE=1

cd /users/joeunyook/logit-probe

ENVS=${ENVS:-"bandit monty trust_game"}
SEEDS=${SEEDS:-"0 1"}
ROUNDS=${ROUNDS:-5}
OUT_ROOT=SMOKE_distill

echo "============================================="
echo "logit-probe R1-distill smoke test — model=${MODEL} envs=${ENVS} seeds=${SEEDS} rounds=${ROUNDS}"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Started : $(date)"
echo "============================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

~/bin/micromamba run -n ftdiag python run_all.py \
    --models "${MODEL}" \
    --envs ${ENVS} \
    --seeds ${SEEDS} --rounds "${ROUNDS}" \
    --out-root "${OUT_ROOT}"

echo ""
echo "--- generation done, training probes on the smoke slice (sanity only) ---"
~/bin/micromamba run -n ftdiag python analysis/probe_train.py \
    --models "${MODEL}" \
    --envs ${ENVS} \
    --residuals-dir "${OUT_ROOT}/residuals" \
    --tag "${OUT_ROOT}" --save-weights

echo ""
echo "--- diagnostics ---"
~/bin/micromamba run -n ftdiag python3 -c "
import json, glob

model = '${MODEL}'
files = glob.glob(f'${OUT_ROOT}/responses/{model}_*_seed*.json')
print(f'response files found: {len(files)}')

n_rounds = 0
n_valid = 0
n_artifact = 0
n_near_cap = 0
n_boxed = 0
n_letter = 0
max_toks = 0
sample_tail = None

for fn in files:
    d = json.load(open(fn))
    for r in d['rounds']:
        n_rounds += 1
        txt = r.get('cot_text') or ''
        if r['post_valid']:
            n_valid += 1
        if 'Ġ' in txt or 'Ċ' in txt:
            n_artifact += 1
        if r['n_gen_tokens'] >= 2000:
            n_near_cap += 1
        if '\\\\boxed{' in txt:
            n_boxed += 1
        if 'Answer:' in txt:
            n_letter += 1
        max_toks = max(max_toks, r['n_gen_tokens'])
        sample_tail = txt[-200:] if txt else sample_tail

print(f'rounds sampled          : {n_rounds}')
print(f'post-CoT valid rate     : {100*n_valid/n_rounds:.1f}%'  if n_rounds else 'n/a')
print(f'Ġ/Ċ decode artifacts    : {n_artifact}  (must be 0)')
print(f'hit near 2048 cap       : {n_near_cap}  ({100*n_near_cap/n_rounds:.1f}%)' if n_rounds else 'n/a')
print(f'used \\\\boxed{{}} format   : {n_boxed}')
print(f'used Answer: format     : {n_letter}')
print(f'max gen_tokens seen     : {max_toks}')
print()
print('sample of last generated text (200 chars):')
print(repr(sample_tail))
print()
verdict = 'PASS' if (n_artifact == 0 and n_rounds > 0 and n_valid/n_rounds > 0.5) else 'CHECK MANUALLY'
print(f'VERDICT: {verdict}')
"

echo ""
echo "============================================="
echo "Finished : $(date)"
echo "============================================="
