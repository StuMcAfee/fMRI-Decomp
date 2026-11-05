#!/bin/bash
# Submit the wave stage for every run in a list file as an LSF job array (at most 8 at a time), then the
# cross-participant summary once every element has ended. Run from inside the repository:
#   bash scripts/submit_waves_batch.sh runs.txt
# runs.txt: one mbme_tedana run folder per line (made with --keep-intermediates; paths may contain spaces).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f "${1:-}" ] || { echo "usage: bash scripts/submit_waves_batch.sh <run-list file>" >&2; exit 2; }
COMMIT="$(git rev-parse --short HEAD)"
mkdir -p logs
LIST="$PWD/logs/waves_runs_$COMMIT.txt"
grep -v '^[[:space:]]*$' "$1" > "$LIST"                      # drop blank lines so line numbers = array indices
N="$(wc -l < "$LIST")"
echo "runs: $N (copied to $LIST); code: git $COMMIT"
missing=0
while IFS= read -r run; do
    [ -f "$run/work/echo-1_desc-mc_bold.nii.gz" ] || { echo "  no motion-corrected echoes: $run"; missing=$((missing + 1)); }
done < "$LIST"
[ "$missing" -eq 0 ] || echo "  $missing run(s) lack work/echo-k_desc-mc_bold.nii.gz: those elements will stop in preflight"
bsub -q standard -n 8 -R "rusage[mem=2000] span[hosts=1]" -env "all" -J "wavesb[1-$N]%8" \
     -o "logs/waves_batch_%J_%I.log" "bash scripts/waves_batch.lsf '$LIST'"
bsub -q standard -n 1 -R "rusage[mem=4000]" -env "all" -J wavesb_summary -w "ended(wavesb)" \
     -o "logs/waves_summary_%J.log" "bash scripts/waves_batch_summary.lsf $COMMIT"
echo "summary will be in outputs/waves_batch/$COMMIT/ (waves_batch.md, waves_batch.tsv, waves_batch_motion.png)"
