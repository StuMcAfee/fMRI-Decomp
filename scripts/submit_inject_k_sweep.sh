#!/bin/bash
# Injection at higher k for every injection folder that has a front end: one LSF job each (8 cores, 32 GB), then
# the pooled summary (it lists every k separately). Run from inside the repository, on the login node:
#   bash scripts/submit_inject_k_sweep.sh [k ...]        (default: 90 150)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
KS="${*:-90 150}"
PY="${MEJOINT_PY:-$HOME/miniconda3_new/envs/mejoint/bin/python}"
mkdir -p logs
IDS=()
for d in outputs/inject/*/*_s*/; do
    d="${d%/}"
    [ -f "$d/frontend_mc/c0.npy" ] || { echo "$d: no injected front end; skipped"; continue; }
    tag="$(basename "$(dirname "$d")" | cut -d_ -f1)_$(basename "$d")"
    out="$(bsub -q standard -n 8 -R "rusage[mem=4000] span[hosts=1]" -env "all" -J "injk_$tag" -o "logs/inject_k_%J_$tag.log" \
           "bash scripts/inject_k_sweep.lsf '$d' $KS")"
    echo "$out"
    id="$(echo "$out" | sed -n 's/.*<\([0-9]*\)>.*/\1/p' | head -n 1)"
    [ -n "$id" ] && IDS+=("ended($id)")
done
[ ${#IDS[@]} -gt 0 ] || { echo "nothing submitted" >&2; exit 1; }
DEP="$(IFS='&'; echo "${IDS[*]}" | sed 's/&/ \&\& /g')"
bsub -q standard -n 1 -R "rusage[mem=8000]" -env "all" -J inject_k_summary -w "$DEP" -o "logs/inject_k_summary_%J.log" \
     "cd '$PWD' && PYTHONPATH='$PWD/src' '$PY' -m mejoint.cli inject-summary --root outputs/inject"
echo "per folder: outputs/inject/<run>/<folder>/recovery_k<k>.md; pooled (every k): outputs/inject/inject_summary.md"
