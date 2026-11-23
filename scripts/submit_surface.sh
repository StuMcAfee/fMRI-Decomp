#!/bin/bash
# Surface pipeline S2-S4 and the icosahedral parcel ICA for every participant whose recon-all has finished:
# one surface_labels job per run, then (when it finishes successfully) one parcel_ica job on the 1876 icosahedral
# parcels. Run from inside the repository, on the login node:
#   bash scripts/submit_surface.sh [--no-ica] [run ...]      (default: every finished run in outputs/freesurfer)
# The pilot (pilot_me3) is given its own config automatically. Finished outputs are not checked: a rerun redoes them.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f outputs/site.yaml ] || { echo "STOP: outputs/site.yaml not found (subject exclusions and coil set live there)" >&2; exit 1; }
ICA=1; RUNS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-ica) ICA=0; shift ;;
        *) RUNS+=("$1"); shift ;;
    esac
done
if [ ${#RUNS[@]} -eq 0 ]; then
    for d in outputs/freesurfer/*/; do
        n="$(basename "$d")"
        case "$n" in fsaverage*) continue ;; esac
        RUNS+=("$n")
    done
fi
mkdir -p logs
n=0
for r in "${RUNS[@]}"; do
    if [ ! -f "outputs/freesurfer/$r/scripts/recon-all.done" ] || [ -f "outputs/freesurfer/$r/scripts/recon-all.error" ]; then
        echo "recon-all not finished: $r (skipped)"; continue
    fi
    CFG=""; [ "$r" = "pilot_me3" ] && CFG="outputs/pilot/pilot_me3/config_pass2.yaml"
    if [ ! -f "outputs/anatomy/$r/t12epi.mat" ]; then echo "no volumetric anatomy for $r (run scripts/anatomy_run.lsf first; skipped)"; continue; fi
    J="surf_${r:0:40}"
    bsub -q standard -n 4 -R "rusage[mem=8000] span[hosts=1]" -env "all" -J "$J" -o "logs/surface_${r}_%J.log" \
         "bash scripts/surface_labels.lsf $r $CFG"
    if [ "$ICA" = 1 ]; then
        bsub -q standard -n 4 -R "rusage[mem=6000] span[hosts=1]" -env "all" -w "done($J)" -J "pp_${r:0:40}" -o "logs/parcel_pipeline_${r}_%J.log" \
             "bash scripts/parcel_pipeline.lsf $r $CFG"
    fi
    n=$((n + 1))
done
echo "submitted surface jobs for $n runs$([ "$ICA" = 1 ] && echo ', each followed by its parcel pipeline job')"
