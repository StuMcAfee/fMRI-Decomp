#!/bin/bash
# FreeSurfer recon-all for every participant with a front end and a T1 (one LSF job each, 4 cores), plus extra
# runs given as name=T1 (e.g. the pilot):
#   bash scripts/submit_freesurfer.sh [--t1-list file] [name=/path/T1.nii[.gz] ...]
# T1s are found as for the anatomy (regions.t1_list / t1_root in the config, or --t1-list). Excluded subjects
# (outputs/site.yaml) are skipped. Run from inside the repository, on the login node.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f outputs/site.yaml ] || { echo "STOP: outputs/site.yaml not found (subject exclusions live there)" >&2; exit 1; }
PY="${MEJOINT_PY:-$HOME/miniconda3_new/envs/mejoint/bin/python}"
T1LIST=""; EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --t1-list) T1LIST="$2"; shift 2 ;;
        *=*) EXTRA+=("$1"); shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
mkdir -p logs
mapfile -t PAIRS < <(PYTHONPATH="$PWD/src" "$PY" - "$T1LIST" <<'PYEOF'
import sys
from mejoint import anatomy, io, route_b_batch
cfg = io.load_config(); G = cfg.get("regions", {})
for run in route_b_batch.list_runs("outputs/phase2", cfg):
    t1 = anatomy.find_t1(run, sys.argv[1] or G.get("t1_list"), G.get("t1_root"))
    print(f"{run}={t1}" if t1 else f"{run}=")
PYEOF
)
n=0
for p in "${PAIRS[@]}" "${EXTRA[@]}"; do
    name="${p%%=*}"; t1="${p#*=}"
    if [ -z "$t1" ]; then echo "no T1: $name (skipped)"; continue; fi
    bsub -q standard -n 4 -R "rusage[mem=4000] span[hosts=1]" -env "all" -J "fs_${name:0:40}" -o "logs/freesurfer_${name}_%J.log" \
         "bash scripts/freesurfer_run.lsf $name $t1"
    n=$((n + 1))
done
echo "submitted $n recon-all jobs; outputs in outputs/freesurfer/<run>/"
