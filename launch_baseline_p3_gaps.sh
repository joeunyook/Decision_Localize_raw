#!/bin/bash
# Baseline P3 post-CoT gap fills (run_post.sbatch, seeds 0-39, --skip-existing).
# Only missing seeds run. Gate: run AFTER canary 2891856 confirms env-on-GPU is good.
set -euo pipefail
cd /users/joeunyook/logit-probe
A="--account=a0133 --qos=normal"

# --- small/mid, 1 GPU ---
sbatch $A --job-name=post3-gemma2b-gap --export=ALL,MODEL=gemma2b,ENVS=machine:chicken run_post.sbatch
sbatch $A --job-name=post3-phi-gap     --export=ALL,MODEL=phi,ENVS=chicken            run_post.sbatch

# --- dsqwen7 P3 (bandit terminates; ipd/stag/trust/machine/chicken are the
#     documented non-terminators -- run all, count valid vs looped as a finding).
#     1-2 envs per job because looping rounds run to the 2048-token cap = slow. ---
sbatch $A --job-name=post3-dsq-1 --export=ALL,MODEL=dsqwen7,ENVS=bandit:ipd       run_post.sbatch
sbatch $A --job-name=post3-dsq-2 --export=ALL,MODEL=dsqwen7,ENVS=stag_hunt        run_post.sbatch
sbatch $A --job-name=post3-dsq-3 --export=ALL,MODEL=dsqwen7,ENVS=trust_game       run_post.sbatch
sbatch $A --job-name=post3-dsq-4 --export=ALL,MODEL=dsqwen7,ENVS=machine          run_post.sbatch
sbatch $A --job-name=post3-dsq-5 --export=ALL,MODEL=dsqwen7,ENVS=chicken          run_post.sbatch

# --- dsllama70 P3 tail (70B, 2 GPUs), split for wall safety ---
sbatch $A --gpus-per-node=2 --job-name=post3-dsl70-a --export=ALL,MODEL=dsllama70,ENVS=ipd:stag_hunt:trust_game run_post.sbatch
sbatch $A --gpus-per-node=2 --job-name=post3-dsl70-b --export=ALL,MODEL=dsllama70,ENVS=machine:chicken          run_post.sbatch

echo "baseline P3 gap jobs submitted."
squeue -u joeunyook -o "%.10i %.22j %.9T %.10M %R"
