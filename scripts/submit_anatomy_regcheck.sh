#!/bin/bash
# R2 registration check for several participants, one LSF job each:
#   bash scripts/submit_anatomy_regcheck.sh [subject ...]     
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS=("$@"); [ ${#RUNS[@]} -gt 0 ] || { echo "usage: bash scripts/submit_anatomy_regcheck.sh <subject> [...]" >&2; exit 2; }
mkdir -p logs
for r in "${RUNS[@]}"; do
    bsub -q standard -n 1 -R "rusage[mem=8000]" -env "all" -J "regcheck_$r" -o "logs/regcheck_%J_$r.log" "bash scripts/anatomy_regcheck.lsf $r"
done
echo "per participant: outputs/anatomy/<run>/regcheck.md and regcheck.png; logs: logs/regcheck_<job>_<subject>.log"
