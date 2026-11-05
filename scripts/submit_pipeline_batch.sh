#!/bin/bash
# The whole pipeline for every participant in an echo-1 list: one LSF job per participant (job array, at most
# 8 at a time; 8 cores, 32 GB each), then the cross-participant wave summary once all have ended.
# Run from inside the repository, on the login node:
#   bash scripts/submit_pipeline_batch.sh <echo-1 list> [mbme_tedana.py] [OUT_keep folder]
# The list is the mbme_tedana input (one echo-1 NIfTI per line; quotes, blank lines and #comments allowed).
# Subjects excluded in configs/default.yaml (subjects.exclude) are dropped before submission.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SRC="${1:-}"
ANALYSIS="/research_jude/rgs01_jude/dept/DI/tir/vision_lab/stu/CSIMemphis/Analysis/Monitoring Visit"
MBME="${2:-$ANALYSIS/mbme_tedana.py}"
OUTK="${3:-$ANALYSIS/OUT_keep}"
TPY="${TEDANA_PY:-$HOME/miniconda3_new/envs/tedana/bin/python}"
PY="${MEJOINT_PY:-$HOME/miniconda3_new/envs/mejoint/bin/python}"
[ -f "$SRC" ] || { echo "usage: bash scripts/submit_pipeline_batch.sh <echo-1 list> [mbme_tedana.py] [OUT_keep]" >&2; exit 2; }
[ -f "$MBME" ] || { echo "mbme_tedana.py not found: $MBME" >&2; exit 2; }
COMMIT="$(git rev-parse --short HEAD)"
mkdir -p logs
LIST="$PWD/logs/pipeline_list_$COMMIT.txt"
# clean list: unquoted, no blanks or comments (so line numbers = mbme_tedana --index), excluded subjects dropped
PYTHONPATH="$PWD/src" "$PY" - "$SRC" "$LIST" <<'PYEOF'
import re, sys
from pathlib import Path
from mejoint import io
excl = {str(k) for k in (io.load_config().get("subjects", {}).get("exclude") or {})}
keep = []
for line in Path(sys.argv[1]).read_text().splitlines():
    s = line.strip().strip('"').strip("'").strip()
    if not s or s.startswith("#"):
        continue
    subj = next((p for p in Path(s).parts if re.fullmatch(r"\d+", p)), "")
    if subj in excl:
        print(f"  excluded (configs/default.yaml): {subj}")
        continue
    keep.append(s)
Path(sys.argv[2]).write_text("\n".join(keep) + "\n")
print(f"  participants to run: {len(keep)}")
PYEOF
N="$(wc -l < "$LIST")"
echo "list: $LIST ($N lines); code: git $COMMIT; mbme_tedana: $MBME; output: $OUTK"
# fast dry run of mbme_tedana (echo files and echo times for every line; no processing)
echo "== mbme_tedana check"
"$TPY" "$MBME" check --list "$LIST" --out "$PWD/logs/mbme_check_$COMMIT" --echo-times 16.20 31.22 46.24 61.26 \
    || { echo "mbme_tedana check failed: fix the list before submitting (output above)" >&2; exit 1; }
for d in "$OUTK"/sub-*/; do [ -f "$d/work/echo-1_desc-mc_bold.nii.gz" ] && echo "  already preprocessed (kept): $(basename "$d")"; done
bsub -q standard -n 8 -R "rusage[mem=4000] span[hosts=1]" -env "all" -J "mejoint_full[1-$N]%8" \
     -o "logs/pipeline_%J_%I.log" "bash scripts/pipeline_batch.lsf '$LIST' '$MBME' '$OUTK'"
bsub -q standard -n 1 -R "rusage[mem=4000]" -env "all" -J mejoint_full_summary -w "ended(mejoint_full)" \
     -o "logs/pipeline_summary_%J.log" "bash scripts/waves_batch_summary.lsf $COMMIT"
echo "logs: logs/pipeline_<job>_<element>.log; wave summary: outputs/waves_batch/$COMMIT/"
