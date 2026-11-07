#!/bin/bash
# Phase 5 injection calibration: one LSF job per participant and amplitude level, then a pooled summary once all
# have ended. Run from inside the repository, on the login node:
#   bash scripts/submit_inject.sh [--layout focal|network] [--levels "strong weak"] [--seed 1] [--clean] <subject or run folder name> [...]
# e.g.  bash scripts/submit_inject.sh P04 P12
#       bash scripts/submit_inject.sh --layout network P04 P12
# A subject number selects its run folder in OUT_keep (sub-<number>_*). The original run must already have been
# through the whole pipeline (front end, tissue classes, sweep), as in scripts/submit_pipeline_batch.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ANALYSIS="/research_jude/rgs01_jude/dept/DI/tir/vision_lab/stu/CSIMemphis/Analysis/Monitoring Visit"
MBME="${MBME:-$ANALYSIS/mbme_tedana.py}"
OUTK="${OUTK:-$ANALYSIS/OUT_keep}"
LEVELS="strong weak"; SEED=1; CLEAN=""; LAYOUT="focal"
RUNS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --levels) LEVELS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --layout) LAYOUT="$2"; shift 2 ;;
        --clean) CLEAN="clean"; shift ;;
        *) RUNS+=("$1"); shift ;;
    esac
done
[ ${#RUNS[@]} -gt 0 ] || { echo "usage: bash scripts/submit_inject.sh [--layout focal|network] [--levels \"strong weak\"] [--seed 1] [--clean] <subject> [...]" >&2; exit 2; }
[ -f "$MBME" ] || { echo "mbme_tedana.py not found: $MBME" >&2; exit 2; }
mkdir -p logs
IDS=()
for r in "${RUNS[@]}"; do
    if [ -d "$OUTK/$r" ]; then RD="$OUTK/$r"; else RD="$(ls -d "$OUTK"/sub-"$r"_* 2>/dev/null | head -n 1 || true)"; fi
    [ -n "$RD" ] && [ -f "$RD/result.json" ] || { echo "no run folder for $r in $OUTK; skipped" >&2; continue; }
    NAME="$(basename "$RD")"
    [ -f "outputs/phase2/$NAME/mc/analysis_mask.nii.gz" ] || { echo "$NAME: no front end in outputs/phase2/$NAME/mc (run the pipeline first); skipped" >&2; continue; }
    for L in $LEVELS; do
        out="$(bsub -q standard -n 8 -R "rusage[mem=4000] span[hosts=1]" -env "all" -J "inject_${LAYOUT}_${r}_${L}" \
               -o "logs/inject_%J_${LAYOUT}_${r}_${L}.log" "bash scripts/inject_run.lsf '$RD' '$MBME' $L seed=$SEED layout=$LAYOUT $CLEAN")"
        echo "$out"
        id="$(echo "$out" | sed -n 's/.*<\([0-9]*\)>.*/\1/p' | head -n 1)"
        [ -n "$id" ] && IDS+=("ended($id)")
    done
done
[ ${#IDS[@]} -gt 0 ] || { echo "nothing submitted" >&2; exit 1; }
DEP="$(IFS='&'; echo "${IDS[*]}" | sed 's/&/ \&\& /g')"
PY="${MEJOINT_PY:-$HOME/miniconda3_new/envs/mejoint/bin/python}"
bsub -q standard -n 1 -R "rusage[mem=8000]" -env "all" -J inject_summary -w "$DEP" -o "logs/inject_summary_%J.log" \
     "cd '$PWD' && PYTHONPATH='$PWD/src' '$PY' -m mejoint.cli inject-summary --root outputs/inject"
echo "logs: logs/inject_<job>_${LAYOUT}_<subject>_<level>.log; per run: outputs/inject/<run>/[network_]<level>_s$SEED/recovery.md; pooled: outputs/inject/inject_summary.md"
