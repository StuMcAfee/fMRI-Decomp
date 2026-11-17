#!/bin/bash
# Surface-pipeline resources (login node, needs internet), into outputs/atlases_fs32k/ (research storage, not git):
#   - fs_LR 32k spheres for resampling FreeSurfer surfaces (HCP pipelines, standard_mesh_atlases):
#       {L,R}.sphere.32k_fs_LR.surf.gii and resample_fsaverage/fs_LR-deformed_to-fsaverage.{L,R}.sphere.32k_fs_LR.surf.gii
#   - Icosahedron1002.{L,R}.label.gii and the fs32k medial-wall masks (DiedrichsenLab Functional_Fusion, tpl-fs32k)
#   - Schaefer2018_200Parcels_7Networks_order.dlabel.nii on fs_LR 32k (CBIG)
#   - the cortico-cerebellar connectivity model Nettekoven_2024_Fusion_L2.pdconn.nii (DiedrichsenLab
#     cortico_cereb_connectivity; stored with Git LFS, so fetched from media.githubusercontent.com) and the SUIT 3 mm
#     grey-matter mask
# Usage: bash scripts/fetch_fs32k_resources.sh        (rerun safely: existing files are kept)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
O="outputs/atlases_fs32k"; mkdir -p "$O"
HCP="https://raw.githubusercontent.com/Washington-University/HCPpipelines/master/global/templates/standard_mesh_atlases"
FF="https://raw.githubusercontent.com/DiedrichsenLab/Functional_Fusion/main/Functional_Fusion/Atlases"
CBIG="https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti"
CCC="https://media.githubusercontent.com/media/DiedrichsenLab/cortico_cereb_connectivity/main/data"
get() {   # get <url> <file name>: download unless present; refuse Git LFS pointers (a few hundred bytes of text)
    local url="$1" f="$O/$2"
    if [ -s "$f" ] && ! grep -q "git-lfs.github.com" "$f" 2>/dev/null; then echo "have  $2"; return; fi
    curl -sSfL --retry 3 -o "$f.part" "$url" || { echo "FAILED to download $url" >&2; exit 1; }
    if grep -q "git-lfs.github.com" "$f.part" 2>/dev/null; then echo "FAILED: $2 is a Git LFS pointer, not the file ($url)" >&2; exit 1; fi
    mv "$f.part" "$f"; echo "got   $2 ($(du -h "$f" | cut -f1))"
}
for H in L R; do
    get "$HCP/$H.sphere.32k_fs_LR.surf.gii" "$H.sphere.32k_fs_LR.surf.gii"
    get "$HCP/resample_fsaverage/fs_LR-deformed_to-fsaverage.$H.sphere.32k_fs_LR.surf.gii" "fs_LR-deformed_to-fsaverage.$H.sphere.32k_fs_LR.surf.gii"
    get "$FF/tpl-fs32k/Icosahedron1002.$H.label.gii" "Icosahedron1002.$H.label.gii"
    get "$FF/tpl-fs32k/tpl-fs32k_hemi-${H}_mask.label.gii" "tpl-fs32k_hemi-${H}_mask.label.gii"
done
get "$CBIG/Schaefer2018_200Parcels_7Networks_order.dlabel.nii" "Schaefer2018_200Parcels_7Networks_order.dlabel.nii"
get "$CCC/Nettekoven_2024_Fusion_L2.pdconn.nii" "Nettekoven_2024_Fusion_L2.pdconn.nii"
get "$FF/tpl-SUIT/tpl-SUIT_res-3_gmcmask.nii" "tpl-SUIT_res-3_gmcmask.nii"
# the connectivity model as published (sha256 of the Git LFS object, checked 2026-11-17)
echo "1d45bf2fa4b2bb7b4e67cb4bd72f4e9006aedf27f89d4e16415c8b5605d21e1c  $O/Nettekoven_2024_Fusion_L2.pdconn.nii" | sha256sum -c -
echo "resources in $PWD/$O"
