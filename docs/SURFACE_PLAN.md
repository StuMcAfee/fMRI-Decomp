# Surface-based cortical regions and the lesion-network analysis (design A)

User decision, 2026-11-17: adopt the Diedrichsen-lab convention for the cortex (FreeSurfer surfaces on fs_LR 32k,
the 1876-parcel icosahedral tessellation used by the cortico-cerebellar connectivity models) so that route B's
BOLD-like signals can be used directly with those models; use the models for a lesion-network analysis (design A).
The cerebellum stays out of the outcomes (CLAUDE.md section 1): no cerebellar BOLD is measured.

## 1. Verified facts (2026-11-17)

- `DiedrichsenLab/cerebellar_atlases`: cerebellar atlases only, volumetric (SUIT, MNI152NLin6AsymC,
  MNI152NLin2009cSymC; NIfTI, plus GIFTI cerebellar flat-map projections); CC BY unless the source paper says otherwise.
- `DiedrichsenLab/cortico_cereb_connectivity` (MIT): connectivity weights as CIFTI `.pdconn.nii` in `data/` (Git
  LFS); README: cortex "Icosahedron1002", cerebellum "SUIT3", "The cerebellum is voxel wise". Downloaded
  `Nettekoven_2024_Fusion_L2.pdconn.nii` (sha256 1d45bf2f..., matches the LFS record): 6918 cerebellar voxels (SUIT 3 mm
  grid 47 x 32 x 29) x 1876 cortical parcels (fs32k vertices); 44 % of the weights negative (ridge regression).
- Nettekoven et al. 2024 (Nat Commun 15:8376), Methods: cortex "projected onto individual surfaces, which are aligned
  to the symmetric freesurfer32LR template"; "1876 parcels using a regular icosahedron"; activity "averaged across
  all voxels in each cerebral ROI" (matrix X); cerebellar activity predicted as Y = XW (ridge); no smoothing.
- `Functional_Fusion/Atlases/tpl-fs32k`: `Icosahedron1002.{L,R}.label.gii` (938 parcels per hemisphere, 0 = medial
  wall). Checked: left label k = model parcel cortex_left_k, right label k = model parcel 938 + k (all 1876 match).
- Schaefer 200/7 on fs_LR 32k: CBIG `Schaefer2018_200Parcels_7Networks_order.dlabel.nii` (64,984 vertices, ids 1-200).
- FreeSurfer to fs_LR 32k: resample through the subject's `sphere.reg` (aligned to fsaverage) and the HCP
  `fs_LR-deformed_to-fsaverage.{L,R}.sphere.32k_fs_LR.surf.gii` (HCPpipelines standard_mesh_atlases).
- Cluster: FreeSurfer modules 6.0-7.4.1; Connectome Workbench not installed as a module, but
  `connectome-workbench-cli` is on conda-forge (version 2.2.1 listed).

## 2. Pipeline

| step | script | output |
|---|---|---|
| resources | `scripts/fetch_fs32k_resources.sh` (login node) | `outputs/atlases_fs32k/` |
| S1 recon-all | `scripts/submit_freesurfer.sh` -> `freesurfer_run.lsf` | `outputs/freesurfer/<run>/` |
| S2 surfaces on fs_LR 32k, moved to EPI space | `scripts/surface_labels.lsf` | `outputs/surface/<run>/` |
| S3 Schaefer 200/7 and Icosahedron1002 on the EPI cortical ribbon; region labels (grey matter, CSF > 0.3, thalamus / striatum as before) | same | `outputs/anatomy_surf/<run>/` |
| S4 route B with surface-based Schaefer parcels (sensitivity); icosahedral parcel time courses in the model's parcel order | same | `outputs/route_b_surf/<run>/{200_7, ico1002_7}` |

The functional data are never resampled; the surfaces only decide which EPI voxels form each parcel. The surface-based
Schaefer parcels are first a sensitivity analysis next to the current volumetric ones (per-parcel Dice is reported);
whether they replace them is a later decision, because the regions define the primary.

Precision: about 24,000 cortical ribbon voxels per participant over 1876 parcels is about 13 voxels per icosahedral
parcel, below route B's 20-voxel minimum; per-parcel BOLD-like signals are noisy (separation cost). The export keeps
every parcel (no threshold) because applying model weights is linear and pools over many parcels; per-parcel values
are not interpreted. Route B's own measures stay on Schaefer 200/7.

## 3. Design A: lesion-network analysis (definitions proposed for approval)

Inputs: each patient's cerebellar lesion map (and, separately, dose map) in SUIT 3 mm, from the separate efforts.

1. **Model**: primary `Nettekoven_2024_Fusion_L2` (average of the task-based models; best in the authors'
   evaluation); sensitivity `Nettekoven_2024_HCP_L2` (fitted on resting-state data, like ours).
2. **Disconnection of cortical parcel p in patient i**: D_ip = sum_v L_iv W+_vp / sum_v W+_vp, with L_iv the
   lesioned fraction of cerebellar voxel v (0-1) and W+ the positive part of the weights: the share of the parcel's
   positive cerebellar connectivity that targets damaged tissue. Sensitivity: |W| instead of W+. For dose maps, the
   dose-weighted version (L = dose / prescribed dose) is a separate secondary score.
3. **Aggregation to Schaefer 200/7**: each Schaefer parcel's D as the vertex-weighted mean over the icosahedral
   parcels it overlaps on fs_LR 32k; network-level D as the mean over a network's parcels.
4. **Tests (to be fixed before any look at the relation)**: (a) within patients, across cortical parcels, whether
   parcel-level BOLD-like measures (coupling with the parcel's own network, and with the rest of cortex after removing
   the brain-wide pattern) are lower where D is higher, expressed against the controls once available; (b) across
   patients, network-level D against network-level integration. Both secondary; the primary outcome is unchanged.

Decided (user, 2026-11-17): model `Nettekoven_2024_Fusion_L2` primary (HCP_L2 sensitivity) and positive weights
(|W| sensitivity). Still open: the outcome measures in (4), to be fixed before any look at the relation, and the
availability and timing of SUIT lesion / dose maps.
