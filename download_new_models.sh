#!/bin/bash
# Download the 2026-07-13 model additions into the shared offline HF cache.
# Run on a login node (compute nodes have no internet; jobs set HF_HUB_OFFLINE=1).
set -uo pipefail
source ~/.secrets
export HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache
unset HF_HUB_OFFLINE

MODELS=(
  google/gemma-2-2b-it
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  meta-llama/Llama-3.3-70B-Instruct
  deepseek-ai/DeepSeek-R1-Distill-Llama-70B
)

fail=0
for m in "${MODELS[@]}"; do
  echo "=== downloading $m ==="
  # exclude original/consolidated checkpoints; transformers only needs safetensors
  ~/bin/micromamba run -n ftdiag hf download "$m" \
      --exclude "original/*" --exclude "consolidated*" --exclude "*.pth" \
      || { echo "!!! FAILED: $m"; fail=1; }
done
echo ""
[ "$fail" -eq 0 ] && echo "ALL DOWNLOADS COMPLETE" || echo "SOME DOWNLOADS FAILED"
exit $fail
