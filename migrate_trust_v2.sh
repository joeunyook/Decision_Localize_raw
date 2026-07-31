#!/bin/bash
# =============================================================================
# trust_game v1 -> v2 migration (2026-07-19). Run ONCE, BEFORE any v2 job.
#
# Does two things:
#   1. Archives all v1 trust_game raw artifacts to archive/trust_v1/.
#      MANDATORY: the env name is unchanged, so without this the sweep's
#      --skip-existing silently keeps v1 data, and probe training would mix
#      v1/v2 residuals.
#   2. Scaffolds the new canonical tag P1_v4: copies the four unchanged envs'
#      probe/direction artifacts (results/P1_v3) and steering replays
#      (steer/P1_v3) into P1_v4, so after the trust_game v2 jobs finish,
#      P1_v4 is the complete 45-combo tag and P1_v3 stays frozen as the v1
#      snapshot. (probe_summary.json is not copied: it is rebuilt per job;
#      merge the per-combo JSONs for a full summary.)
#
# After this script, per model (see run_sweep_100seeds.sbatch header for the
# account/qos/gpu flags each model needs):
#   sbatch ... --export=ALL,MODEL=<m>,SEED_FROM=0,SEED_TO=99,ENVS=trust_game,TAG=P1_v4 run_sweep_100seeds.sbatch
#   sbatch ... --export=ALL,MODEL=<m>,ENVS=trust_game,TAG=P1_v4 run_steer.sbatch
# Then once, on any node (CPU-heavy but no GPU needed):
#   ~/bin/micromamba run -n ftdiag python analysis/by_t.py --tag P1_v4
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

ARCH=archive/trust_v1
if [ -e "$ARCH" ]; then
    echo "ABORT: $ARCH already exists -- migration appears to have run already."
    echo "       (Delete/rename it only if you are sure, then rerun.)"
    exit 1
fi

echo "== 1/3 archiving v1 trust_game raw artifacts to $ARCH =="
mkdir -p "$ARCH"/{results,responses,residuals,steer_P1_v3}
n_res=$(ls results/*_trust_game_seed*.json 2>/dev/null | wc -l)
n_rsp=$(ls responses/*_trust_game_seed*.json 2>/dev/null | wc -l)
n_rsd=$(ls residuals/*_trust_game_seed* 2>/dev/null | wc -l)
n_str=$(ls steer/P1_v3/*_trust_game_seed*.json 2>/dev/null | wc -l)
[ "$n_rsd" -gt 0 ] || { echo "ABORT: no v1 trust_game residuals found -- wrong directory?"; exit 1; }
mv results/*_trust_game_seed*.json    "$ARCH"/results/
mv responses/*_trust_game_seed*.json  "$ARCH"/responses/
mv residuals/*_trust_game_seed*       "$ARCH"/residuals/
mv steer/P1_v3/*_trust_game_seed*.json "$ARCH"/steer_P1_v3/
echo "   moved: $n_res results, $n_rsp responses, $n_rsd residual files, $n_str steer replays"

left=$( (ls results/*_trust_game_seed* responses/*_trust_game_seed* \
            residuals/*_trust_game_seed* steer/P1_v3/*_trust_game_seed* \
         2>/dev/null || true) | wc -l)
[ "$left" -eq 0 ] || { echo "ABORT: $left trust_game v1 files still in place"; exit 1; }

echo "== 2/3 scaffolding canonical tag P1_v4 from the four unchanged envs =="
mkdir -p results/P1_v4 steer/P1_v4 plots/P1_v4
for env in bandit monty ipd stag_hunt; do
    cp results/P1_v3/probe_*_${env}.json     results/P1_v4/
    cp results/P1_v3/direction_*_${env}.npz  results/P1_v4/
    cp steer/P1_v3/*_${env}_seed*.json       steer/P1_v4/
    cp plots/P1_v3/probe_*_${env}.png        plots/P1_v4/ 2>/dev/null || true
done
echo "   P1_v4 now holds: $(ls results/P1_v4/probe_*.json | wc -l) probe JSONs," \
     "$(ls results/P1_v4/direction_*.npz | wc -l) directions," \
     "$(ls steer/P1_v4/*.json | wc -l) steer replays (36/36/1440 expected)"

echo "== 3/3 sanity: smoke test on the v2 env =="
~/bin/micromamba run -n ftdiag python smoke_test.py > /dev/null \
    && echo "   smoke test PASSED" \
    || { echo "ABORT: smoke test failed -- fix before launching jobs"; exit 1; }

echo ""
echo "Migration complete. Launch per model (trust_game v2 + machine + chicken"
echo "in one job; flags per run_sweep_100seeds.sbatch header):"
echo "  sbatch --account=a0133 --qos=normal --job-name=sweep3-<m> \\"
echo "      --export=ALL,MODEL=<m>,SEED_FROM=0,SEED_TO=99,ENVS=trust_game:machine:chicken,TAG=P1_v4 run_sweep_100seeds.sbatch"
echo "  sbatch --account=a0133 --qos=normal --job-name=steer3-<m> \\"
echo "      --export=ALL,MODEL=<m>,ENVS=trust_game:machine:chicken,TAG=P1_v4 run_steer.sbatch   # after its sweep job"
echo "  (llama70/dsllama70: add --gpus-per-node=2)"
echo "Then regenerate the unified by-t tables:"
echo "  ~/bin/micromamba run -n ftdiag python analysis/by_t.py --tag P1_v4"
