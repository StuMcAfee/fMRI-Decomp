# R5: route B over all participants (draft for approval, 2026-11-11)

Route B passed R3 (injection validation, revalidated on the CSF-excluded labels) and R4 (pilot review, P04 and
P12). R5 applies the same analysis, unchanged, to all 15 participants and produces the tables for the group
analysis with the cerebellar factors and, later, the controls. Nothing here changes a measure; the open choices are
listed in section 7.

## 1. Inputs that already exist

- Front ends for all 15 runs (`outputs/phase2/<run>/mc`, from the whole-pipeline batch of 2026-11-06, motion-corrected
  echoes kept in `OUT_keep`), with the 24-term motion regression basis.
- Data-only tissue classes for all runs (`outputs/tissue/<run>/mc`).
- Anatomy for P04 and P12 only.

## 2. Steps

1. **Preflight (one command, login node).** For each of the 15 runs: front end present; echo files named in its
   `done.json` reachable; T1 found (list file, else the `regions.t1_root` search); number of kept volumes and any
   gaps from `volumes_used.tsv`; coil set (head-only or head-neck). Output: one table, so missing inputs are seen
   before any job is submitted.
2. **Anatomy for the 13 remaining participants** (`scripts/submit_anatomy.sh`, one job each, ~2-5 min). Reference
   image: the static T2* map (as accepted in R2). Labels: Schaefer 200 / 7 (and 400 / 17), cortex and deep regions
   without voxels of CSF probability > 0.3, all-voxel cortex for the sensitivity analysis.
3. **Registration gate (user), as in R2.** For each participant: `qc_epi_to_t1.png`, `regcheck` table and figure,
   `qc_subcortical.png`, `qc_200_7.png`. Participants whose registration fails are rerun with another reference or
   excluded by the user; none goes forward unchecked. Patients with large cavities, shunts or enlarged ventricles are
   flagged for a closer look.
4. **Common region set.** After anatomy: the cortical parcels usable in every participant (section 7, R5-2). Written
   to `outputs/route_b/common_ids_200_7.txt`; route B then reports the full usable set and the common set.
5. **Route B for all 15** (`route_b_run.lsf`, one job each, ~10-20 min with the all-voxel sensitivity), at a common
   number of volumes (section 7, R5-1).
6. **Per-participant sensitivity by injection** (CLAUDE.md: the primary is reported with each participant's
   measured sensitivity). Two injections per participant, as in R3 at resting strength: independent networks and one
   shared network. Reported: change in the participation ratio and in the version without the first pattern, and
   share recovery. 26 injection jobs plus their front ends (the R3 job scripts, now over all participants).
7. **Group tables** (new `mejoint route-b-summary`):
   - **participant table** (one row each): participation ratio (primary), version without the first pattern,
     global share, segregation, eigenvalues above null, each on the common set and the full set; the sensitivity
     rows (all-voxel cortex, reliability >= 0.5); cortical CSF fraction (covariate); median and 5th-percentile
     reliability; number of usable parcels; volumes used; motion (mean and maximum framewise displacement after
     respiratory filtering, fraction of frames above 0.2 mm); coil set; injection sensitivity;
   - **region table** (one row per participant and region): BOLD-like, S0-like and leftover shares with intervals,
     reliability, voxels before and after CSF exclusion, energy-below-noise flags;
   - **deep-region table**: thalamus and striatum coupling with each network, with intervals and the
     measurable / not-measurable flag (section 7, R5-4).
8. **Cohort checks** (they need many participants): QC-FC (correlation across participants between motion and each
   edge, and with the participation ratio); distance dependence of that relation (parcel centroids in MNI space);
   participation ratio against volumes used, coil set and cortical CSF fraction.
9. **Review gate (user)** on the tables and cohort checks before any group statistics.

## 3. Sensitivity analyses

- **First pass (built in, run in step 5):** all-voxel cortex; cortex without regions below reliability 0.5;
  Schaefer 400 / 17.
- **Second pass (need front-end reruns, scheduled after the review gate):** band-passed motion projection instead of
  the 24-term regression; censoring at filtered FD > 0.2 mm; global signal regression; equal echo weighting; band
  limits; grey-matter threshold.

## 4. Compute

About 13 anatomy jobs, 15 route B jobs, 26 injection jobs (each with a front end), standard queue, 4 cores and
8 GB per core. The injections dominate (as in R3).

## 5. Outputs

`outputs/anatomy/<run>/`, `outputs/route_b/<run>/200_7/` (as in R4), `outputs/inject/<run>/regions_*`, and
`outputs/route_b/summary/` (participant, region and deep-region tables as TSV, a markdown summary and cohort figures).

## 6. What R5 does not do

No group statistics, no patient-control comparison (controls not yet acquired), no use of clinical variables in any
step that defines regions or measures (CLAUDE.md section 1).

## 7. Decisions for the user

- **R5-1 Common number of volumes.** The primary must be computed at a common number of volumes. Kept volumes are
  800 in most runs and 797, 797, 777 and 751 in four (P12, P14, P10, P05; exact assignment from
  `volumes_used.tsv` in the preflight). Run P10 has a 19-volume gap inside the run. Recommendation: truncate every
  run to its first 751 volumes for the integration measures (shares use all volumes); for P10, use the longest
  gap-free stretch if it reaches 751 volumes, otherwise report it separately and leave it out of the primary
  (band-pass filtering and circular shifts assume an unbroken time series).
- **R5-2 Common region set.** Recommendation: the parcels usable in every participant define the set for group
  comparison (the participation ratio depends on which and how many regions enter); the full usable set is reported
  alongside. If one participant removes many parcels (a cavity, dropout), the user decides whether to exclude that
  participant from the common set rather than shrink it for everyone.
- **R5-3 Per-participant injection sensitivity.** Recommendation: yes, two injections per participant (independent
  and shared, resting strength), as in step 6.
- **R5-4 Thalamus and striatum at group level.** Per participant, coupling below reliability 0.5 is not measurable
  (5 of 8 deep regions in the pilots). Recommendation: report per participant with the flag, and at group level a
  pooled estimate over the measurable participants (inverse-variance weighted mean of Fisher-z values from the
  bootstrap intervals) with the count of participants contributing.
- **R5-5 Sensitivity schedule.** Recommendation: first pass as in section 3 within R5; second pass after the review
  gate.
