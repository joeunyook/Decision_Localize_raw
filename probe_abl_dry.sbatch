#!/bin/bash
#SBATCH --account=a0133
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --job-name=probe-abl-dry
#SBATCH --output=/users/joeunyook/logit-probe/probe_abl_dry_%j.log

set -euo pipefail
cd /users/joeunyook/logit-probe
echo "Started: $(date)"
~/bin/micromamba run -n ftdiag python analysis/probe_train.py     --residuals-dir ABL_dry_v1/residuals     --tag ABL_dry_v1
echo "Done: $(date)"
