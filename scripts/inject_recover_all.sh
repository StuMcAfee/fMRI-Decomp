#!/bin/bash
# Re-run the recovery step (no preprocessing, no new decomposition) for every injection folder under
# outputs/inject/<run>/<level>_s<seed>/, then the pooled summary. Uses the saved front end, decomposition,
# te_check.json and the mbme_tedana run of each injection. Run from inside the repository (login node is fine for
# a few runs; each takes a few minutes):
#   bash scripts/inject_recover_all.sh [k, default 50]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${MEJOINT_PY:-$HOME/miniconda3_new/envs/mejoint/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
K="${1:-50}"
for IDIR in outputs/inject/*/*_s*/; do
    IDIR="${IDIR%/}"
    NAME="$(basename "$(dirname "$IDIR")")"
    FE1="$IDIR/frontend_mc"; KD="$IDIR/decompose/k$K"
    [ -f "$FE1/c0.npy" ] && [ -f "$KD/components.json" ] || { echo "$IDIR: no front end or k$K decomposition; skipped"; continue; }
    IRUN="$(ls -d "$IDIR"/OUT/*/ 2>/dev/null | head -n 1)"; IRUN="${IRUN%/}"
    ARGS=(); [ -n "$IRUN" ] && ARGS+=(--mbme-run "$IRUN")
    BASEK="outputs/phase5/$NAME/phase4_sweep_mc/k$K"; [ -f "$BASEK/time_courses.tsv" ] && ARGS+=(--baseline-k-dir "$BASEK")
    echo "== $IDIR $(date)"
    "$PY" -m mejoint.cli inject-recover --inject-dir "$IDIR" --frontend-dir "$FE1" --k-dir "$KD" "${ARGS[@]}" > "$IDIR/recover_rerun.log" 2>&1 \
        && echo "   ok: $IDIR/recovery.md" || { echo "   FAILED, see $IDIR/recover_rerun.log"; tail -5 "$IDIR/recover_rerun.log"; }
done
"$PY" -m mejoint.cli inject-summary --root outputs/inject > /dev/null && echo "pooled: outputs/inject/inject_summary.md"
