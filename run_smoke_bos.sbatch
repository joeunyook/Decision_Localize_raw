#!/bin/bash
# =============================================================================
# logit-probe — SMOKE TEST for the new `bos` env (Bach or Stravinsky / Battle
# of the Sexes), on the DEBUG partition for a fast turnaround.
# =============================================================================
# Exercises the full per-round path on one small model: forced-choice pre-CoT
# probe under BOTH label mappings, residual capture (Probe 1), and post-CoT
# generation + answer parsing (Probe 3).
#
# Writes to its OWN root SMOKE_bos so --skip-existing cannot silently skip
# anything (session_done() only checks file existence -- the trap that made the
# 2026-07-24 dry-CoT job generate zero CoT).
#
# Usage:
#   sbatch --account=a0133 --partition=debug --qos=normal \
#       --job-name=smoke-bos run_smoke_bos.sbatch
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --time=01:30:00
#SBATCH --output=/users/joeunyook/logit-probe/smokebos_%x_%j.log

set -euo pipefail
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
export HF_HUB_OFFLINE=1

cd /users/joeunyook/logit-probe

MODEL=${MODEL:-gemma2b}
ENVS=${ENVS:-bos}
SEED_FROM=${SEED_FROM:-0}
SEED_TO=${SEED_TO:-1}
OUT_ROOT=${OUT_ROOT:-SMOKE_bos}

echo "============================================="
echo "logit-probe SMOKE — bos — model=${MODEL} envs=${ENVS} seeds=${SEED_FROM}-${SEED_TO}"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Out root: ${OUT_ROOT}"
echo "Started : $(date)"
echo "============================================="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

~/bin/micromamba run -n ftdiag python run_all.py \
    --models "${MODEL}" \
    --envs ${ENVS} \
    --seeds $(seq "${SEED_FROM}" "${SEED_TO}") --rounds 10 \
    --out-root "${OUT_ROOT}"

echo ""
echo "=============== diagnostics ==============="
~/bin/micromamba run -n ftdiag python - <<'PY'
import json, glob, os
root = os.environ.get("OUT_ROOT", "SMOKE_bos")
# NB: per-round records live in responses/, NOT results/ -- run_all.py splits the
# session dict (results/) from the rounds list (responses/). Reading results/ for
# a "rounds" key is a KeyError.
files = sorted(glob.glob(f"{root}/responses/*_bos_seed*.json"))
print(f"sessions written : {len(files)}")
if not files:
    print("FAIL: no session files -- --skip-existing may have skipped, or a crash")
    raise SystemExit(1)
nr = nv = 0
oracles, pres = [], []
for f in files:
    d = json.load(open(f))
    s = d["session"]
    print(f"\n-- {os.path.basename(f)}  hidden={s.get('hidden_param')}  act0={s.get('act0')} act1={s.get('act1')}")
    for r in d["rounds"]:
        nr += 1
        nv += bool(r.get("post_valid"))
        oracles.append(r["oracle_p_act1"])
        pres.append(r["delta_pre_sem"])
        print(f"   t={r['t']:2d} oracle={r['oracle_p_act1']:.3f} pre={r['delta_pre_sem']:+.2f} "
              f"post_valid={r.get('post_valid')} letter={r.get('parsed_letter')} toks={r.get('n_gen_tokens')}")
resid = glob.glob(f"{root}/residuals/*_bos_seed*.np*")
print(f"\nrounds           : {nr}")
print(f"post_valid       : {nv}/{nr}  ({100*nv/max(nr,1):.0f}%)")
print(f"residual files   : {len(resid)}")
print(f"oracle range     : {min(oracles):.3f} .. {max(oracles):.3f}")
print(f"oracle both sides: {'YES' if min(oracles)<0.5<max(oracles) else 'NO -- single-class, check env'}")
print(f"pre gap finite   : {'YES' if all(abs(p)<1e6 for p in pres) else 'NO -- inf/nan present'}")
PY
echo ""
echo "Finished : $(date)"
