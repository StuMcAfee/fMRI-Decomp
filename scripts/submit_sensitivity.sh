#!/bin/bash
# Second-pass sensitivity analyses: one LSF job per participant and variant.
# Run from inside the repository, on the login node:
#   bash scripts/submit_sensitivity.sh [--variants "band_008 gsr censor atlas_400_17 gm05 equal_weighting motion_lowpass6"] <subject> [...]
# Then: PYTHONPATH=src ~/miniconda3_new/envs/mejoint/bin/python -m mejoint.cli sensitivity-summary
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VARIANTS="band_008 gsr censor atlas_400_17 gm05 equal_weighting motion_lowpass6"
RUNS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --variants) VARIANTS="$2"; shift 2 ;;
        *) RUNS+=("$1"); shift ;;
    esac
done
[ ${#RUNS[@]} -gt 0 ] || { echo "usage: bash scripts/submit_sensitivity.sh [--variants \"...\"] <subject> [...]" >&2; exit 2; }
mkdir -p logs
for v in $VARIANTS; do
    [ -f "configs/sensitivity/$v.yaml" ] || { echo "no configs/sensitivity/$v.yaml" >&2; exit 2; }
    for r in "${RUNS[@]}"; do
        s="${r#sub-}"; s="${s%%_*}"
        bsub -q standard -n 4 -R "rusage[mem=8000] span[hosts=1]" -env "all" -J "sens_${v}_$s" -o "logs/sens_${v}_${s}_%J.log" \
             "bash scripts/sensitivity_run.lsf $r $v"
    done
done
echo "submitted $(( $(echo $VARIANTS | wc -w) * ${#RUNS[@]} )) jobs"
