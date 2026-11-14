#!/bin/bash
# Route B step 3 for several participants: one LSF job each (FSL registration and region labels).
# Run from inside the repository, on the login node:
#   bash scripts/submit_anatomy.sh <T1 list file> [subject ...]        
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
T1LIST="${1:-}"; shift || true
[ -f "$T1LIST" ] || { echo "usage: bash scripts/submit_anatomy.sh <T1 list file> [subject ...]" >&2; exit 2; }
RUNS=("$@"); [ ${#RUNS[@]} -gt 0 ] || { echo "usage: bash scripts/submit_anatomy.sh <T1 list file> <subject> [...]" >&2; exit 2; }
mkdir -p logs
for r in "${RUNS[@]}"; do
    bsub -q standard -n 1 -R "rusage[mem=8000]" -env "all" -J "anat_$r" -o "logs/anatomy_%J_$r.log" \
         "bash scripts/anatomy_run.lsf $r '$(cd "$(dirname "$T1LIST")" && pwd)/$(basename "$T1LIST")'"
done
echo "per participant: outputs/anatomy/<run>/qc_200_7.png and qc_t1_to_mni.png (check by eye), regions_200_7.tsv"
