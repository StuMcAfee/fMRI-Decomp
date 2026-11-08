#!/bin/bash
# Background-corrected eta: one LSF job per participant (native components at each k, computing k = 150 if missing,
# and that participant's injections), then the pooled summaries. Run from inside the repository, on the login node:
#   bash scripts/submit_eta_background.sh [--k "50 90 150"] [subject ...]      (default subjects: P04 P12)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
KS="50 90 150"; RUNS=()
while [ $# -gt 0 ]; do case "$1" in --k) KS="$2"; shift 2 ;; *) RUNS+=("$1"); shift ;; esac; done
[ ${#RUNS[@]} -gt 0 ] || RUNS=(P04 P12)
PY="${MEJOINT_PY:-$HOME/miniconda3_new/envs/mejoint/bin/python}"
mkdir -p logs
IDS=()
for r in "${RUNS[@]}"; do
    out="$(bsub -q standard -n 8 -R "rusage[mem=4000] span[hosts=1]" -env "all" -J "etabg_$r" -o "logs/eta_background_%J_$r.log" \
           "bash scripts/eta_background_run.lsf $r $KS")"
    echo "$out"
    id="$(echo "$out" | sed -n 's/.*<\([0-9]*\)>.*/\1/p' | head -n 1)"
    [ -n "$id" ] && IDS+=("ended($id)")
done
DEP="$(IFS='&'; echo "${IDS[*]}" | sed 's/&/ \&\& /g')"
KARGS=""; for k in $KS; do KARGS="$KARGS --k $k"; done
bsub -q standard -n 1 -R "rusage[mem=8000]" -env "all" -J etabg_summary -w "$DEP" -o "logs/eta_background_summary_%J.log" \
     "cd '$PWD' && PYTHONPATH='$PWD/src' '$PY' -m mejoint.cli inject-summary --root outputs/inject && PYTHONPATH='$PWD/src' '$PY' -m mejoint.cli eta-background-summary $KARGS --runs ${RUNS[*]}"
echo "native: outputs/eta_background/eta_background_summary.md / .png; injections: outputs/inject/inject_summary.md"
