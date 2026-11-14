#!/bin/bash
# R5: route B for every participant (or the ones named), one LSF job each, with the common region set if it exists.
# Run from inside the repository, on the login node:
#   bash scripts/submit_route_b.sh [--atlas 200_7] [--after-anatomy] [subject ...]     (default: every run with a front end)
# --after-anatomy: each job waits for that participant's anat_<subject> job (submit_anatomy.sh) to finish.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${MEJOINT_PY:-$HOME/miniconda3_new/envs/mejoint/bin/python}"
ATLAS="200_7"; AFTER=""
RUNS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --atlas) ATLAS="$2"; shift 2 ;;
        --after-anatomy) AFTER=1; shift ;;
        *) RUNS+=("$1"); shift ;;
    esac
done
if [ ${#RUNS[@]} -eq 0 ]; then
    mapfile -t RUNS < <(PYTHONPATH="$PWD/src" "$PY" -c "
from mejoint import io, route_b_batch
print('\n'.join(route_b_batch.list_runs('outputs/phase2', io.load_config())))")
fi
COMMON="outputs/route_b/common_ids_${ATLAS}.txt"
[ -f "$COMMON" ] && echo "common region set: $COMMON" || { echo "no common region set yet ($COMMON): full usable sets only"; COMMON=""; }
mkdir -p logs
for r in "${RUNS[@]}"; do
    s="${r#sub-}"; s="${s%%_*}"
    DEP=(); [ -n "$AFTER" ] && DEP=(-w "done(anat_$s)")
    bsub -q standard -n 4 -R "rusage[mem=8000] span[hosts=1]" -env "all" "${DEP[@]}" -J "rb_$s" -o "logs/route_b_${s}_%J.log" \
         "bash scripts/route_b_run.lsf $r $ATLAS $COMMON"
done
echo "when all have finished: $PY -m mejoint.cli route-b-summary   (PYTHONPATH=src)"
