#!/bin/bash
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/users/joeunyook/logit-probe/cpulayerbal_%j.log
# Full per-layer BALANCED LOSO curves for the 24 combos whose P1_v4 probes predate
# the balanced-accuracy field. Writes to tag P1_v4_layerbal -- canonical
# results/P1_v4/ is NOT modified. No --save-weights: directions stay frozen.
cd /users/joeunyook/logit-probe
# 24 concurrent sklearn processes each defaulted to ~130 BLAS threads on this
# 288-core node -> load average 1348, zero layers completed in 5 min. Pin them.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
for M in gemma2b llama phi qwen dsqwen7 qwen27 gemma31 llama70; do
  for E in bandit monty stag_hunt; do
    ~/bin/micromamba run -n ftdiag python analysis/probe_train.py \
      --residuals-dir residuals --tag P1_v4_layerbal --models $M --envs $E &
  done
done
wait
echo "done $(date)"
