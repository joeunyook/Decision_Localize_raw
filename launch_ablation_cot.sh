#!/bin/bash
# Ablation dry-tone CoT column (run_ablation_dry_cot.sbatch, seeds 0-14 default,
# --skip-existing, OUT_ROOT=ABL_dry_v1). Full 9 models x 7 envs. This is the
# entire dry-CoT table column (0 cot_text present today). Gate: run AFTER canary
# 2891856 (gemma2b bandit:monty seeds 0-1) confirms env + dry-CoT templates work.
set -euo pipefail
cd /users/joeunyook/logit-probe
A="--account=a0133 --qos=normal"
ALL7="bandit:monty:ipd:stag_hunt:trust_game:machine:chicken"

# --- small/fast, 1 GPU, all 7 envs in one job ---
sbatch $A --job-name=abldry-cot-gemma2b --export=ALL,MODEL=gemma2b,ENVS=$ALL7 run_ablation_dry_cot.sbatch
sbatch $A --job-name=abldry-cot-llama   --export=ALL,MODEL=llama,ENVS=$ALL7   run_ablation_dry_cot.sbatch
sbatch $A --job-name=abldry-cot-phi     --export=ALL,MODEL=phi,ENVS=$ALL7     run_ablation_dry_cot.sbatch
sbatch $A --job-name=abldry-cot-qwen    --export=ALL,MODEL=qwen,ENVS=$ALL7    run_ablation_dry_cot.sbatch

# --- dsqwen7 (7B, loopers slow even at 15 seeds), split 3/4 ---
sbatch $A --job-name=abldry-cot-dsq-a --export=ALL,MODEL=dsqwen7,ENVS=bandit:monty:ipd            run_ablation_dry_cot.sbatch
sbatch $A --job-name=abldry-cot-dsq-b --export=ALL,MODEL=dsqwen7,ENVS=stag_hunt:trust_game:machine:chicken run_ablation_dry_cot.sbatch

# --- 27-31B, 1 GPU, split for wall safety ---
sbatch $A --job-name=abldry-cot-qwen27-a  --export=ALL,MODEL=qwen27,ENVS=bandit:monty:ipd:stag_hunt run_ablation_dry_cot.sbatch
sbatch $A --job-name=abldry-cot-qwen27-b  --export=ALL,MODEL=qwen27,ENVS=trust_game:machine:chicken  run_ablation_dry_cot.sbatch
sbatch $A --job-name=abldry-cot-gemma31-a --export=ALL,MODEL=gemma31,ENVS=bandit:monty:ipd:stag_hunt run_ablation_dry_cot.sbatch
sbatch $A --job-name=abldry-cot-gemma31-b --export=ALL,MODEL=gemma31,ENVS=trust_game:machine:chicken  run_ablation_dry_cot.sbatch

# --- 70B, 2 GPUs, split ---
sbatch $A --gpus-per-node=2 --job-name=abldry-cot-llama70-a --export=ALL,MODEL=llama70,ENVS=bandit:monty:ipd:stag_hunt run_ablation_dry_cot.sbatch
sbatch $A --gpus-per-node=2 --job-name=abldry-cot-llama70-b --export=ALL,MODEL=llama70,ENVS=trust_game:machine:chicken  run_ablation_dry_cot.sbatch
sbatch $A --gpus-per-node=2 --job-name=abldry-cot-dsl70-a   --export=ALL,MODEL=dsllama70,ENVS=bandit:monty:ipd:stag_hunt run_ablation_dry_cot.sbatch
sbatch $A --gpus-per-node=2 --job-name=abldry-cot-dsl70-b   --export=ALL,MODEL=dsllama70,ENVS=trust_game:machine:chicken  run_ablation_dry_cot.sbatch

echo "ablation dry-CoT jobs submitted."
squeue -u joeunyook -o "%.10i %.22j %.9T %.10M %R"
