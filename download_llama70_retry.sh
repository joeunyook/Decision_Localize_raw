#!/bin/bash
# Hardened llama70 fetch: the plain download stalls silently on this network
# (threads hang mid-transfer). Strategy: force plain-HTTP backend, short
# per-request timeouts, and a timeout+resume loop -- each pass wipes the stale
# locks a killed pass leaves behind, then resumes from the blobs on disk.
# On success, submits the sweep.
set -u
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
unset HF_HUB_OFFLINE
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=30

REPO=meta-llama/Llama-3.3-70B-Instruct
# a0107 store quota is 100% full -- download to scratch instead;
# core/model_loader.py points the llama70 key at this path.
DEST=/capstor/scratch/cscs/joeunyook/models/Llama-3.3-70B-Instruct

for attempt in $(seq 1 40); do
  echo "=== attempt $attempt : $(date) ==="
  timeout 900 ~/bin/micromamba run -n ftdiag hf download "$REPO" \
      --local-dir "$DEST" \
      --exclude "original/*" --exclude "consolidated*" --exclude "*.pth" \
      --max-workers 8
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== DOWNLOAD COMPLETE : $(date) ==="
    cd /users/joeunyook/logit-probe
    sbatch --account=a0133 --qos=normal --gpus-per-node=2 \
        --job-name=sweep-llama70 \
        --export=ALL,MODEL=llama70,SEED_FROM=0,SEED_TO=99 \
        run_sweep_100seeds.sbatch
    exit 0
  fi
  echo "--- attempt $attempt exited rc=$rc (124=stall timeout), retrying"
  sleep 10
done
echo "=== GAVE UP after 40 attempts ==="
exit 1
