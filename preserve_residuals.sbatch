#!/bin/bash
# =============================================================================
# logit-probe -- preserve purge-exposed residuals from SCRATCH to a0133 STORE
# =============================================================================
# WHY: /capstor/scratch auto-purges files untouched for 30 days. The oldest
# access time under residuals/ was 2026-07-11, so that tree begins purging
# around 2026-08-10. The a0133 project store has no purge policy.
#
# This job COPIES. It deletes nothing, and it never touches the scratch source.
# rsync is resume-safe: if the job is killed, resubmit the same command.
#
#   cd ~/logit-probe && sbatch --account=a0133 --qos=normal preserve_residuals.sbatch
#
# *** PASS --account=a0133 ON THE COMMAND LINE. ***  The login environment sets
# SBATCH_ACCOUNT=a0107, and Slurm environment variables OVERRIDE #SBATCH
# directives in the script (only command-line flags beat them). a0107 does not
# allow the `normal` QOS, so a plain `sbatch preserve_residuals.sbatch` fails
# with the misleading error "Invalid qos specification".
# =============================================================================
#SBATCH --account=a0133
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=preserve-residuals
#SBATCH --output=/users/joeunyook/logit-probe/preserve_residuals_%j.log

set -uo pipefail

SRC=/capstor/scratch/cscs/joeunyook/logit-probe
DST=/capstor/store/cscs/swissai/a0133/joeunyook/logit-probe-preserved
MAN=/users/joeunyook/logit-probe/MANIFESTS

echo "==============================================================="
echo "PRESERVE RESIDUALS  scratch -> a0133 store"
echo "started : $(date -Is)   job $SLURM_JOB_ID on $(hostname)"
echo "source  : $SRC   (NOT modified, NOT deleted)"
echo "dest    : $DST"
echo "==============================================================="

mkdir -p "$DST"

# -a preserves times/perms and does NOT dereference symlinks (never use -L:
# the archive tree contains a self-referential symlink; keep the habit here).
# --partial + resumability means a killed job can simply be resubmitted.
for TREE in residuals ABL_dry_v1; do
  echo ""
  echo "--- copying $TREE ---"
  echo "    started $(date -Is)"
  rsync -a --partial --info=progress2,stats2 \
        "$SRC/$TREE/" "$DST/$TREE/"
  RC=$?
  echo "    rsync exit=$RC  finished $(date -Is)"
  if [ $RC -ne 0 ]; then
    echo "    *** RSYNC FAILED for $TREE -- resubmit this same script ***"
  fi
done

echo ""
echo "==============================================================="
echo "VERIFICATION against the manifests committed in the GitHub repo"
echo "==============================================================="
OVERALL_OK=1
for TREE in residuals ABL_dry_v1; do
  echo ""
  echo "--- $TREE ---"
  SRC_N=$(find "$SRC/$TREE" -type f 2>/dev/null | wc -l)
  DST_N=$(find "$DST/$TREE" -type f 2>/dev/null | wc -l)
  MAN_N=$(grep -vc '^#' "$MAN/$TREE.manifest" 2>/dev/null || echo 0)
  echo "    files in manifest : $MAN_N"
  echo "    files on scratch  : $SRC_N"
  echo "    files on store    : $DST_N"
  echo "    bytes on store    : $(du -sh "$DST/$TREE" 2>/dev/null | cut -f1)"

  # size+path comparison against the manifest, which is the authoritative
  # record of what existed at archive time.
  ( cd "$DST/$TREE" && find . -type f -printf '%s %p\n' | sort -k2 ) > "/tmp/dst_$TREE.$$"
  grep -v '^#' "$MAN/$TREE.manifest" | sort -k2 > "/tmp/man_$TREE.$$"
  DIFFN=$(diff "/tmp/man_$TREE.$$" "/tmp/dst_$TREE.$$" | grep -c '^[<>]' || true)
  echo "    manifest vs store, differing lines: $DIFFN"
  if [ "$DIFFN" -eq 0 ]; then
    echo "    STATUS: EXACT MATCH -- every file present at its recorded size"
  else
    OVERALL_OK=0
    echo "    STATUS: MISMATCH -- first 10 differences:"
    diff "/tmp/man_$TREE.$$" "/tmp/dst_$TREE.$$" | grep '^[<>]' | head -10 \
      | sed 's/^/        /'
    echo "    ( '<' = in manifest but missing from store; '>' = extra on store )"
  fi
  rm -f "/tmp/dst_$TREE.$$" "/tmp/man_$TREE.$$"
done

echo ""
echo "==============================================================="
if [ "$OVERALL_OK" -eq 1 ]; then
  echo "RESULT: SUCCESS -- both trees preserved and verified."
  echo "The scratch copies are untouched and may now be left to purge."
else
  echo "RESULT: INCOMPLETE -- see mismatches above. Resubmit this script;"
  echo "rsync will copy only what is missing."
fi
echo "store usage now:"
du -sh "$DST" 2>/dev/null
echo "finished: $(date -Is)"
echo "==============================================================="
