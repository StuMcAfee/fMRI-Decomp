#!/bin/bash
# The two-stage parcel pipeline for every participant with an icosahedral export (or the ones named):
#   bash scripts/submit_parcel_pipeline.sh [run ...]
# The pilot (pilot_me3) gets its own config automatically. Run from inside the repository, on the login node.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
    for d in outputs/route_b_surf/*/ico1002_7/parcel_channels.npz; do [ -f "$d" ] && RUNS+=("$(basename "$(dirname "$(dirname "$d")")")"); done
fi
mkdir -p logs
for r in "${RUNS[@]}"; do
    CFG=""; [ "$r" = "pilot_me3" ] && CFG="outputs/pilot/pilot_me3/config_pass2.yaml"
    bsub -q standard -n 4 -R "rusage[mem=6000] span[hosts=1]" -env "all" -J "pp_${r:0:40}" -o "logs/parcel_pipeline_${r}_%J.log" \
         "bash scripts/parcel_pipeline.lsf $r $CFG"
done
echo "submitted ${#RUNS[@]} parcel pipeline jobs; reports in outputs/parcel_pipeline/<run>/report.html"
