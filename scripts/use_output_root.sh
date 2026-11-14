#!/bin/bash
# Put every mejoint output in one folder outside the repository (user request, 2026-11-09):
#   bash scripts/use_output_root.sh [folder]
# default folder: /research_jude/rgs01_jude/dept/DI/tir/vision_lab/stu/CSIMemphis/Analysis/MEJOINT_QC
# The repository's outputs/ folder is moved into that folder (its contents, entry by entry) and replaced by a link
# named outputs pointing there, so every script and command keeps using outputs/... unchanged, and earlier results
# (front ends, sweeps, injections) stay available as inputs to later jobs. Run once, on the login node, with no
# jobs running. Nothing is overwritten: if an entry already exists in the folder, the script stops and says which.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DEST="${1:-/research_jude/rgs01_jude/dept/DI/tir/vision_lab/stu/CSIMemphis/Analysis/MEJOINT_QC}"
DEST="${DEST%/}"
if [ -L outputs ]; then
    echo "outputs is already a link to: $(readlink outputs)"; exit 0
fi
mkdir -p "$DEST"
if [ -d outputs ]; then
    echo "outputs/ holds: $(du -sh outputs | cut -f1)"
    for e in outputs/* outputs/.[!.]*; do
        [ -e "$e" ] || continue
        n="$(basename "$e")"
        [ -e "$DEST/$n" ] && { echo "STOP: $DEST/$n already exists; move or rename it, then run again (nothing changed for it)" >&2; exit 1; }
    done
    if [ "$(stat -c %d outputs)" != "$(stat -c %d "$DEST")" ]; then
        echo "outputs/ and $DEST are on different file systems: this copies, then deletes (can take a while)"
    fi
    for e in outputs/* outputs/.[!.]*; do
        [ -e "$e" ] || continue
        echo "  moving $(basename "$e")"
        mv "$e" "$DEST/"
    done
    rmdir outputs
fi
ln -s "$DEST" outputs
echo "outputs -> $DEST"
